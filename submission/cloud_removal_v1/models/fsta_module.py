"""
FSTA-SNN: Frequency-Based Spatial-Temporal Attention Module
============================================================
Reference: FSTA-SNN (AAAI 2025) — https://arxiv.org/pdf/2501.14744
           DarkIR  (CVPR 2025) — https://arxiv.org/pdf/2412.13443

Three plug-and-play modules for SNN image restoration:

1. TemporalAmplitudeAttention (TA)
   — Per-time-step amplitude modulation.
   The existing TimeAttention in model.py uses global 3D pooling to
   produce a single scalar per (T, C) pair.  TA is different: it
   first pools (H, W) → (1, 1) for each (T, C) independently, then
   runs a learned FC over the T axis so that every time step gets its
   own weight, capturing inter-temporal dependencies more precisely.

2. DCTSpatialAttention (SA)
   — Full-spectrum spatial attention via 2D FFT magnitude modulation.
   Standard spatial attention (avg/max pool → 7×7 conv → sigmoid) only
   sees low-frequency (DC-like) global statistics.  SA computes the 2D
   FFT of the time-averaged feature, selectively weights each frequency
   component with a tiny 1×1 MLP, reconstructs via iFFT (preserving
   phase), and produces a spatial attention map from the enriched
   frequency-domain representation.  This allows the network to learn
   which spatial frequency bands to enhance or suppress.

3. FSTAModule
   — Serial TA → SA pipeline with a learnable scale factor.
   The blend gate is sigmoid(scale) with scale initialised to 0, so the
   module starts HALF-open (sigmoid(0) = 0.5), not as an identity; an
   exact identity init would need scale → −∞ or a sigmoid-free gate
   (compare FreMLPBlock.gamma, which multiplies directly and therefore
   IS identity at init).

4. FreMLPBlock
   — Frequency-domain MLP from DarkIR.
   Operates on the 2D FFT magnitude of SNN spike tensors to provide
   global-receptive-field feature enhancement in O(N log N).  Especially
   useful in encoder stages for low-light / rainy-image restoration.

All modules accept and return tensors in the SNN format [T, B, C, H, W].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Temporal Amplitude Attention
# ---------------------------------------------------------------------------

class TemporalAmplitudeAttention(nn.Module):
    """
    Temporal Amplitude Attention (TA).

    Modulates the amplitude (importance weight) of each time step in the
    SNN spike tensor.  Key insight from FSTA-SNN: different time steps
    carry different amounts of task-relevant information, but their
    frequency distributions are highly similar across steps.  Therefore,
    amplitude re-weighting (not frequency manipulation) is the right
    inductive bias for the temporal axis.

    Pipeline (input [T, B, C, H, W]):
        1.  Transpose to [B, T, C, H, W].
        2.  AdaptiveAvgPool3d / AdaptiveMaxPool3d over (H, W) → [B, T, C, 1, 1].
        3.  Learnable α/β mixing: combined = α·avg + β·max.
        4.  Reshape to [B, C, T] and apply Linear(T→T) to capture
            inter-timestep correlations.
        5.  Mean over C dimension → per-step gate [B, T].
        6.  Sigmoid + broadcast-multiply with input, add residual.

    Args:
        channels: Number of feature channels C.
        step:     Number of time steps T.
    """

    def __init__(self, channels: int, step: int = 4):
        super().__init__()
        self.step = step
        self.channels = channels

        # Pool spatial dims, keep T and C independent
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # [B, T, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        # Learnable mixing weights (initialised to equal contribution)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

        # FC captures inter-timestep relationships
        self.fc_t = nn.Linear(step, step, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            out: [T, B, C, H, W]  amplitude-modulated + residual
        """
        T, B, C, H, W = x.shape

        # [T, B, C, H, W] → [B, T, C, H, W]  for Pool3d
        x_bt = x.permute(1, 0, 2, 3, 4)

        avg_out = self.avg_pool(x_bt)  # [B, T, C, 1, 1]
        max_out = self.max_pool(x_bt)  # [B, T, C, 1, 1]

        # Adaptive mixing
        pooled = self.alpha * avg_out + self.beta * max_out   # [B, T, C, 1, 1]
        pooled = pooled.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, C, T]

        # Inter-step linear: [B, C, T] → [B, C, T]
        t_weights = self.fc_t(pooled)                          # [B, C, T]

        # Per-step scalar gate: mean over C → [B, T]
        t_gate = self.sigmoid(t_weights.mean(dim=1))           # [B, T]

        # Broadcast to [T, B, 1, 1, 1]
        t_gate = t_gate.permute(1, 0).view(T, B, 1, 1, 1)

        # Amplitude modulation with additive residual
        return x * t_gate + x


# ---------------------------------------------------------------------------
# 2.  DCT Spatial Attention
# ---------------------------------------------------------------------------

class DCTSpatialAttention(nn.Module):
    """
    Frequency-domain Spatial Attention (SA).

    NOTE on naming: despite the class name, this module uses a 2D real
    FFT (torch.fft.rfft2), NOT a DCT.  The name is kept for state_dict /
    checkpoint stability; treat "DCT" here as a historical misnomer.

    Captures full-spectrum spatial frequency information rather than only
    the DC (mean) statistics used by standard avg/max-pool spatial attention.

    Pipeline (input [T, B, C, H, W]):
        1.  Temporal mean → [B, C, H, W].
        2.  2D real FFT → complex [B, C, H, W//2+1].
        3.  Decompose: magnitude |F|, phase ∠F.
        4.  Frequency-selective MLP on |F| (1×1 Conv, no spatial mixing
            in frequency domain):  learns per-channel, per-frequency weights.
        5.  Reconstruct via iFFT (phase preserved) → [B, C, H, W].
        6.  7×7 conv → spatial attention map [B, 1, H, W].
        7.  Multiply with each time step + additive residual.

    Why this is superior to pooling-based spatial attention:
        • Standard: sees only spatial mean/max — equivalent to DC coefficient.
        • DCT-SA: operates on ALL frequency components simultaneously.
        • The network can learn to suppress rain-streak frequencies (mid-high)
          while preserving structural frequencies (low) — task-adaptive.

    Args:
        channels:  Number of feature channels C.
        reduction: Channel reduction ratio inside the frequency MLP.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 16)

        # 1×1 conv MLP operating on FFT magnitude [B, C, H, W//2+1]
        # (1×1 = channel-wise; spatial dims are frequency coordinates)
        self.freq_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # After iFFT reconstruction, produce spatial attention map
        self.spatial_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            out: [T, B, C, H, W]  spatially-attended + additive residual
        """
        T, B, C, H, W = x.shape

        # Step 1: temporal mean → [B, C, H, W]
        x_avg = x.mean(0)

        # Step 2: 2D FFT → [B, C, H, W//2+1] complex
        x_fft = torch.fft.rfft2(x_avg, norm='backward')

        # Step 3: magnitude + phase
        mag = x_fft.abs()    # [B, C, H, W//2+1]
        pha = x_fft.angle()  # [B, C, H, W//2+1]

        # Step 4: frequency-selective weighting on magnitude
        mag_weight   = self.freq_mlp(mag)          # [B, C, H, W//2+1] ∈ [0,1]
        mag_attended = mag * mag_weight

        # Step 5: reconstruct via iFFT (preserve original phase)
        real     = mag_attended * torch.cos(pha)
        imag     = mag_attended * torch.sin(pha)
        x_fft_mod = torch.complex(real, imag)
        x_spatial = torch.fft.irfft2(x_fft_mod, s=(H, W), norm='backward')  # [B, C, H, W]

        # Step 6: spatial attention map
        attn_map = self.spatial_head(x_spatial)    # [B, 1, H, W]

        # Step 7: apply to every time step (broadcast [1, B, 1, H, W])
        out = x * attn_map.unsqueeze(0)
        return out + x


# ---------------------------------------------------------------------------
# 3.  FSTA Module (plug-and-play)
# ---------------------------------------------------------------------------

class FSTAModule(nn.Module):
    """
    FSTA (Frequency-Based Spatial-Temporal Attention) Module.

    Drop-in plug-and-play attention block for any SNN architecture.
    Combines TA and SA in series, gated by sigmoid(scale) with scale
    initialised to 0 — i.e. the gate starts at 0.5 (half-open), NOT at
    identity: at init the output is x + 0.5·(TA→SA(x) − x).

    Architecture::

        x ──► TemporalAmplitudeAttention ──► DCTSpatialAttention ──► y
        │                                                              │
        └─────────────────── x + sigmoid(s) · (y − x) ───────────────┘

    The learnable scale s allows the network to gradually open the gate
    during training, a technique used in many residual-attention papers.

    Args:
        channels: Feature channels C.
        T:        Number of time steps.
    """

    def __init__(self, channels: int, T: int = 4):
        super().__init__()
        self.ta = TemporalAmplitudeAttention(channels=channels, step=T)
        self.sa = DCTSpatialAttention(channels=channels)
        # scale=0 → sigmoid(scale)=0.5: the blend gate starts half-open
        # (NOT identity — see class docstring).
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            out: [T, B, C, H, W]
        """
        out = self.ta(x)                              # temporal amplitude gate
        out = self.sa(out)                            # FFT spatial gate
        # Learnable blend: gate starts at sigmoid(0)=0.5, adapts with training
        return x + (out - x) * torch.sigmoid(self.scale)


# ---------------------------------------------------------------------------
# 4.  FreMLP Block  (DarkIR, CVPR 2025)
# ---------------------------------------------------------------------------

class FreMLPBlock(nn.Module):
    """
    Frequency-domain MLP Block (FreMLP).

    Enhances features by operating in the 2D Fourier domain:
        1.  2D real FFT of the input feature.
        2.  Modulate the magnitude spectrum with a lightweight 1×1 MLP.
        3.  Reconstruct via iFFT (phase untouched).
        4.  Multiply with the original feature (amplitude × gate).
        5.  Learnable residual scale (initialised to 0).

    Benefits in image restoration:
        • Global receptive field in O(N log N) — equivalent to a full
          non-local operation but far cheaper.
        • Magnitude modulation = frequency-selective enhancement without
          disturbing spatial structure (phase preserved).
        • Natural complement to the spatial convolutions in SRB: those
          model local correlations, FreMLPBlock models global frequency
          patterns (rain streaks have characteristic frequencies).

    Reference: DarkIR (CVPR 2025) — https://arxiv.org/pdf/2412.13443

    Args:
        channels: Input/output channel count C.
        expand:   Channel expansion ratio inside the frequency MLP.
    """

    def __init__(self, channels: int, expand: int = 2):
        super().__init__()

        # Frequency-domain MLP on magnitude [B, C, H, W//2+1]
        self.freq_mlp = nn.Sequential(
            nn.Conv2d(channels, expand * channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * channels, channels, kernel_size=1, bias=False),
        )

        # Layer norm (channel-last GroupNorm equivalent for 2D spatial tensors)
        self.norm = nn.GroupNorm(1, channels)

        # Learnable scale initialised to 0 (safe insertion into trained nets)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _freq_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FreMLP to a single spatial tensor [B, C, H, W].

        Returns frequency-modulated feature of the same shape.
        """
        _, _, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='backward')  # [B, C, H, W//2+1] complex

        mag = x_fft.abs()    # magnitude spectrum
        pha = x_fft.angle()  # phase (structural information)

        # Modulate magnitude, preserve phase
        mag_mod = self.freq_mlp(mag)

        real = mag_mod * torch.cos(pha)
        imag = mag_mod * torch.sin(pha)
        x_out = torch.fft.irfft2(torch.complex(real, imag), s=(H, W), norm='backward')
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            out: [T, B, C, H, W]
        """
        T, B, C, H, W = x.shape

        # Merge T and B for efficient batched processing
        x_flat = x.reshape(T * B, C, H, W)

        # Normalise before frequency processing (stabilises magnitudes)
        x_normed = self.norm(x_flat)

        # Frequency-domain enhancement gate
        x_freq = self._freq_forward(x_normed)       # [T*B, C, H, W]

        # Amplitude-gate: original × frequency-enhanced
        enhanced = x_flat * x_freq

        # Residual with learnable scale (starts at 0 → identity)
        out = x_flat + enhanced * self.gamma

        return out.reshape(T, B, C, H, W)


# ---------------------------------------------------------------------------
# 5.  FSTA Spatial-Only Module  (for use inside TRF-SNN)
# ---------------------------------------------------------------------------

class FSTASpatialOnly(nn.Module):
    """
    Spatial-only variant of FSTAModule for use inside TRF-SNN blocks.

    Inside ResonantResidualBlock, TemporalResonanceAttention (TRA) already
    performs frequency-domain temporal attention.  Stacking TA on top creates
    redundant gradients competing for the same T axis.

    This module keeps only DCTSpatialAttention (SA), which is orthogonal:
      - SA  → 2-D spatial frequency of the time-averaged feature map
      - TRA → 1-D temporal frequency of the spike sequence

    Together they cover both axes without overlap or redundancy.

    Args:
        channels: Feature channels C.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.sa = DCTSpatialAttention(channels=channels)
        # scale=0 → sigmoid(scale)=0.5: blend gate starts half-open
        # (NOT identity — same caveat as FSTAModule).
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            out: [T, B, C, H, W]
        """
        out = self.sa(x)
        return x + (out - x) * torch.sigmoid(self.scale)


# ---------------------------------------------------------------------------
# Quick sanity check (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    T, B, C, H, W = 4, 2, 48, 32, 32
    x = torch.randn(T, B, C, H, W)

    ta  = TemporalAmplitudeAttention(channels=C, step=T)
    sa  = DCTSpatialAttention(channels=C)
    fsta = FSTAModule(channels=C, T=T)
    fsta_sp = FSTASpatialOnly(channels=C)
    fre  = FreMLPBlock(channels=C)

    print("TemporalAmplitudeAttention:", ta(x).shape)
    print("DCTSpatialAttention:       ", sa(x).shape)
    print("FSTAModule:                ", fsta(x).shape)
    print("FSTASpatialOnly:           ", fsta_sp(x).shape)
    print("FreMLPBlock:               ", fre(x).shape)
    print("All outputs match input shape:", all(
        m(x).shape == x.shape for m in [ta, sa, fsta, fsta_sp, fre]
    ))
