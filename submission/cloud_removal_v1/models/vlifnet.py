"""
VLIFNet — ported from system1_VLIFNet/model.py (LovelyC-kx/qun_bme).

Changes vs. the upstream file (kept minimal):

1.  `from fsta_module import …`  →  `from .fsta_module import …`
    (package-relative import).

2.  Module-level `_BACKEND` so the caller picks 'torch' or 'cupy'.
    v1 defaults to 'torch' because cupy wheels can fail to build on
    user machines; the numerics are identical.

3.  `build_vlifnet(...)` / `set_vlifnet_backend(...)` factory helpers.
    build_vlifnet() does NOT force .cuda(); caller decides device.
    Asserts T==4 because SUNet_Level1_Block / SRB / FSTAModule /
    MultiDimensionalAttention hard-code T=4 internally (see comments
    on lines marked "T=4 hardcoded by upstream").

The spiking neuron code (MultiSpike4, mem_update), the SRB dual-group
frequency filter, GatedSkipFusion, FSTA and FreMLPBlock are numerically
identical to upstream.
"""

from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import functional, layer
from . import _sj_compat  # noqa: F401  must run BEFORE any spikingjelly class is used
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsta_module import FSTAModule, FreMLPBlock

# ---------------------------------------------------------------------------
# Module-level knobs
# ---------------------------------------------------------------------------
_BACKEND = "torch"    # v1 default; set_vlifnet_backend('cupy') to switch
v_th = 0.15
alpha = 1 / (2 ** 0.5)
decay = 0.25          # MultiSpike4 / mem_update decay constant


def set_vlifnet_backend(backend: str) -> None:
    """Set backend used by LIFNode layers instantiated *after* this call."""
    global _BACKEND
    assert backend in ("torch", "cupy"), f"backend must be 'torch' or 'cupy', got {backend}"
    _BACKEND = backend


# ---------------------------------------------------------------------------
# BN variant selection (v2 Phase-1 P1.2: SC-16d ablation support)
# ---------------------------------------------------------------------------
# Default "tdbn" = spikingjelly ThresholdDependentBatchNorm2d (what FLSNN
# and Zheng 2021 use). The alternative "bn2d" applies standard nn.BatchNorm2d
# across [T*B, C, H, W] flattened tensors, recovering the baseline used in
# FedBN/SiloBN/HarmoFL/FedWon papers. Switching between them enables the
# SC-16d ablation (v2_comprehensive_literature.md §16.3) — the clean binary
# test of Claim C16 (TDBN makes FedBN's cross-client alignment redundant).

_BN_VARIANT = "tdbn"    # set by set_vlifnet_bn_variant(...) from run_smoke


def set_vlifnet_bn_variant(variant: str) -> None:
    """Select BN variant for all _make_bn() calls that follow.

    Must be called BEFORE constructing VLIFNet (i.e. before build_vlifnet).
    Changing this after construction has no effect on already-built modules.
    """
    global _BN_VARIANT
    assert variant in ("tdbn", "bn2d"), \
        f"bn_variant must be 'tdbn' or 'bn2d', got {variant}"
    _BN_VARIANT = variant


class StandardBN2dWrapper(nn.Module):
    """nn.BatchNorm2d with a 5-D [T, B, C, H, W] interface.

    Flattens T and B dimensions before BN, unflattens after — this matches
    the "tdBN-style pooling over timesteps" used in FLSNN's Fig 5 setup and
    in the standard tdBN definition (Zheng 2021 Eq 5 BEFORE the alpha·V_th
    scaling was introduced). Concretely:

        [T, B, C, H, W]
        -> reshape [T*B, C, H, W]
        -> nn.BatchNorm2d (normalises over T*B, H, W; per-channel affine γ, β)
        -> reshape [T, B, C, H, W]

    This matches the call-interface of spikingjelly's
    ThresholdDependentBatchNorm2d exactly, so it can be swapped in via
    _make_bn() without touching the forward() methods of the using blocks.

    The `alpha`, `v_th` kwargs are accepted for API compatibility and
    silently ignored — standard BN has no threshold-dependence.
    """

    def __init__(self, num_features: int, alpha: float = 1.0,
                 v_th: float = 1.0, affine: bool = True,
                 eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                                 affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        assert x.dim() == 5, \
            f"StandardBN2dWrapper expects 5-D [T,B,C,H,W]; got {tuple(x.shape)}"
        T, B, C, H, W = x.shape
        flat = x.reshape(T * B, C, H, W)
        out = self.bn(flat)
        return out.reshape(T, B, C, H, W)


def _make_bn(num_features: int, alpha: float, v_th: float,
             affine: bool = True) -> nn.Module:
    """Factory for TDBN vs standard-BN, controlled by _BN_VARIANT.

    Keyword signature matches spikingjelly.layer.ThresholdDependentBatchNorm2d
    so existing VLIFNet sub-blocks can switch variant transparently.
    """
    if _BN_VARIANT == "tdbn":
        return layer.ThresholdDependentBatchNorm2d(
            num_features=num_features, alpha=alpha, v_th=v_th, affine=affine)
    elif _BN_VARIANT == "bn2d":
        return StandardBN2dWrapper(
            num_features=num_features, alpha=alpha, v_th=v_th, affine=affine)
    else:
        raise ValueError(f"unknown _BN_VARIANT={_BN_VARIANT}")


# ---------------------------------------------------------------------------
# Backbone variant (v2 Phase-1 P1.3: ANN-vs-SNN ablation for FLSNN §VI-B match)
# ---------------------------------------------------------------------------
# Default "snn" = MultiSpike4 quantization + LIFNode (the model used in all
# v1 and v2 results). The alternative "ann" replaces all spiking activations
# with plain ReLU, letting us run an ANN-backbone baseline within the same
# pipeline (§VI-C opposite to FLSNN Fig 6 ANN vs SNN comparison).

_BACKBONE_VARIANT = "snn"   # set by set_vlifnet_backbone(...) from run_smoke


def set_vlifnet_backbone(variant: str) -> None:
    """Select backbone variant for all neuron factories that follow.

    Must be called BEFORE constructing VLIFNet (i.e. before build_vlifnet).
    Changes do not retroactively affect already-constructed modules.
    """
    global _BACKBONE_VARIANT
    assert variant in ("snn", "ann"), \
        f"backbone must be 'snn' or 'ann', got {variant}"
    _BACKBONE_VARIANT = variant


class ReLUMemUpdate(nn.Module):
    """ANN replacement for mem_update: elementwise ReLU on [T,B,C,H,W].

    mem_update implements a soft-reset LIF neuron with MultiSpike4 output
    that iterates through the T dimension. The ANN counterpart collapses
    that temporal integration into an elementwise nonlinearity — each
    timestep passes through independently. Shape is preserved.
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)


def _make_lif_or_relu(v_threshold: float = v_th,
                      decay_input: bool = False) -> nn.Module:
    """Factory for LIFNode vs ReLU, controlled by _BACKBONE_VARIANT.

    For SNN: spikingjelly LIFNode with multi-step mode and current backend.
    For ANN: nn.ReLU (shape-agnostic, works on multi-step [T,B,C,H,W] input).
    """
    if _BACKBONE_VARIANT == "snn":
        return LIFNode(v_threshold=v_threshold, backend=_BACKEND,
                       step_mode="m", decay_input=decay_input)
    elif _BACKBONE_VARIANT == "ann":
        return nn.ReLU(inplace=False)
    else:
        raise ValueError(f"unknown _BACKBONE_VARIANT={_BACKBONE_VARIANT}")


def _make_mem_update() -> nn.Module:
    """Factory for mem_update (SNN) vs ReLUMemUpdate (ANN)."""
    if _BACKBONE_VARIANT == "snn":
        return mem_update()
    elif _BACKBONE_VARIANT == "ann":
        return ReLUMemUpdate()
    else:
        raise ValueError(f"unknown _BACKBONE_VARIANT={_BACKBONE_VARIANT}")


# ---------------------------------------------------------------------------
# 4-level quantised spike
# ---------------------------------------------------------------------------
class MultiSpike4(nn.Module):
    """Multi-level spike: quantises [0, 4] to {0, 0.25, 0.5, 0.75, 1.0}.

    Hard window gradient (ReLU-like, clipped at 0 and 4) with 1/4 scaling
    to match the forward quantisation step.
    """

    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            quantized = torch.round(torch.clamp(input, min=0, max=4))
            return quantized / 4.0

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input / 4.0

    def forward(self, x):
        return self.quant4.apply(x)


class mem_update(nn.Module):
    """Soft-reset LIF neuron with MultiSpike4 output.  Iterates over the
    leading time dimension T of a [T, B, C, H, W] tensor."""

    def __init__(self):
        super().__init__()
        self.qtrick = MultiSpike4()

    def forward(self, x):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


# ---------------------------------------------------------------------------
# PixelShuffle + LIF (spatial high-frequency extractor)
# ---------------------------------------------------------------------------
class PixelShuffleLIFBlock(nn.Module):
    def __init__(self, in_channels, downsample_factor=2):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.r = downsample_factor

        self.pixel_unshuffle = nn.PixelUnshuffle(downsample_factor)
        self.lif_node = _make_mem_update()

        self.channel_adjust = layer.Conv2d(
            in_channels * (downsample_factor ** 2),
            in_channels,
            kernel_size=1,
            bias=False,
            step_mode="m",
        )
        functional.set_step_mode(self.channel_adjust, step_mode="m")

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.reshape(T * B, C, H, W)
        x_downsampled = self.pixel_unshuffle(x_flat)
        x_reshaped = x_downsampled.reshape(T, B, C * self.r ** 2, H // self.r, W // self.r)
        x_patch_separated = x_reshaped.reshape(T, B, C, self.r ** 2, H // self.r, W // self.r)
        x_temporal = x_patch_separated.permute(0, 3, 1, 2, 4, 5).contiguous()
        x_temporal = x_temporal.reshape(T * self.r ** 2, B, C, H // self.r, W // self.r)
        x_lif_output = self.lif_node(x_temporal)
        x_reorg = x_lif_output.reshape(T, self.r ** 2, B, C, H // self.r, W // self.r)
        x_reorg = x_reorg.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_reorg = x_reorg.reshape(T, B, C * self.r ** 2, H // self.r, W // self.r)
        x_flat_reorg = x_reorg.reshape(T * B, C * self.r ** 2, H // self.r, W // self.r)
        x_upsampled = F.interpolate(x_flat_reorg, size=(H, W), mode="bilinear", align_corners=False)
        x_restored = x_upsampled.reshape(T, B, C * self.r ** 2, H, W)
        x_final = self.channel_adjust(x_restored)
        return x_final


# ---------------------------------------------------------------------------
# 3D channel attention over time
# ---------------------------------------------------------------------------
class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


# ---------------------------------------------------------------------------
# Level-1 multi-scale block
# ---------------------------------------------------------------------------
class SUNet_Level1_Block(nn.Module):
    """SRB → PixelShuffle unshuffle (time expanded to T*r²=16) →
    TimeAttention → Conv3d time-compress back to T → 2 × [LIF-Conv-TDBN-MDAttn]
    → bilinear upsample → skip-add → MDAttn → FreMLP.
    """

    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")

        self.downsample_factor = 2
        self.r = 2

        self.initial_residual = Spiking_Residual_Block(dim=dim)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downsample_factor)
        self.lif_node = _make_mem_update()
        # T=4 hardcoded by upstream: T*r^2 = 16
        self.time_attention = TimeAttention(in_planes=16, ratio=4)
        # T=4 hardcoded by upstream: compress T*r^2 → T along the time axis
        self.temporal_compress = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(4, 1, 1),
                      stride=(4, 1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.lif_1 = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.conv_1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn_1 = _make_bn(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        # T=4 hardcoded by upstream
        self.attn_1 = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

        self.lif_2 = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.conv_2 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn_2 = _make_bn(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        self.attn_2 = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
        self.fre_mlp = FreMLPBlock(channels=dim, expand=2)

    def forward(self, x):
        residual_out = self.initial_residual(x)
        skip_features = residual_out.clone()

        T, B, C, H, W = residual_out.shape
        x_flat = residual_out.reshape(T * B, C, H, W)
        x_downsampled = self.pixel_unshuffle(x_flat)
        x_reshaped = x_downsampled.reshape(T, B, C * self.r ** 2, H // self.r, W // self.r)
        x_patch_separated = x_reshaped.reshape(T, B, C, self.r ** 2, H // self.r, W // self.r)
        x_temporal = x_patch_separated.permute(0, 3, 1, 2, 4, 5).contiguous()
        downsampled = x_temporal.reshape(T * self.r ** 2, B, C, H // self.r, W // self.r)
        downsampled = self.lif_node(downsampled)

        downsampled_transposed = downsampled.transpose(0, 1)   # [B, T*r², C, h, w]
        time_att = self.time_attention(downsampled_transposed)
        attended = downsampled_transposed * time_att
        attended = attended.transpose(0, 1)
        attended_3d = attended.permute(1, 2, 0, 3, 4)          # [B, C, T*r², h, w]
        compressed = self.temporal_compress(attended_3d)       # [B, C, T,     h, w]
        out = compressed.permute(2, 0, 1, 3, 4)                # [T, B, C, h, w]

        out = self.lif_1(out); out = self.conv_1(out); out = self.bn_1(out); out = self.attn_1(out)
        out = self.lif_2(out); out = self.conv_2(out); out = self.bn_2(out); out = self.attn_2(out)

        T, B, C, H, W = out.shape
        out_flat = out.reshape(T * B, C, H, W)
        upsampled_flat = self.upsample(out_flat)
        upsampled = upsampled_flat.reshape(T, B, C, upsampled_flat.shape[2], upsampled_flat.shape[3])

        combined = upsampled + skip_features
        final_out = self.final_attn(combined)
        final_out = self.fre_mlp(final_out)
        return final_out


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif",
                 LayerNorm_type="WithBias", bias=False, T=4):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Spiking Residual Block  (dual-group temporal + spatial frequency)
# ---------------------------------------------------------------------------
class Spiking_Residual_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")

        # Group 1: temporal high-frequency filter (LIF)
        self.lif_1 = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.conv1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn1 = _make_bn(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        self.high_freq_scale_1 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_1  = nn.Parameter(torch.ones(1))

        # Group 2: spatial high-frequency filter (PixelShuffle)
        self.lif_2 = PixelShuffleLIFBlock(in_channels=dim, downsample_factor=2)
        self.conv2 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn2 = _make_bn(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True)
        self.high_freq_scale_2 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_2  = nn.Parameter(torch.ones(1))

        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            _make_bn(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
        )

        # T=4 hardcoded by upstream
        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
        self.fsta = FSTAModule(channels=dim, T=4)
        # Cross-scale gate initialised to 0 → sigmoid gate starts at 0.5 but
        # the multiplicative path is x_h_1 * x_h_2, so the gate gradually
        # opens from zero.  See upstream comment.
        self.cross_scale_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Group 1
        x_h_1 = self.lif_1(x)
        x_l_1 = x - x_h_1
        combined_features_1 = (self.high_freq_scale_1 * x_h_1
                                + self.low_freq_scale_1 * x_l_1
                                + x * x_h_1)
        out = self.conv1(combined_features_1)
        out = self.bn1(out)

        # Group 2
        x_h_2 = self.lif_2(out)
        x_l_2 = out - x_h_2
        x_cross = torch.sigmoid(self.cross_scale_gate) * x_h_1 * x_h_2
        combined_features_2 = (self.high_freq_scale_2 * x_h_2
                                + self.low_freq_scale_2 * x_l_2
                                + out * x_h_2
                                + x_cross)
        out = self.conv2(combined_features_2)
        out = self.bn2(out)

        shortcut = torch.clone(x)
        out = out + self.shortcut(shortcut)
        out = self.attn(out) + shortcut
        out = self.fsta(out)
        return out


class DownSampling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.lif = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.conv = layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode="m", bias=False)
        self.bn = _make_bn(alpha=alpha, v_th=v_th, num_features=dim * 2, affine=True)

    def forward(self, x):
        return self.bn(self.conv(self.lif(x)))


class UpSampling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale_factor = 2
        self.lif = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.conv = layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode="m", bias=False)
        self.bn = _make_bn(alpha=alpha, v_th=v_th, num_features=dim // 2, affine=True)

    def forward(self, input):
        T, B, C, H, W = input.shape
        x_flat = input.reshape(T * B, C, H, W)
        x_up = F.interpolate(x_flat, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        out = x_up.reshape(T, B, C, H * self.scale_factor, W * self.scale_factor)
        return self.bn(self.conv(self.lif(out)))


class GatedSkipFusion(nn.Module):
    """Sigmoid-gated fusion of decoder (dec) and encoder skip (enc)."""

    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.gate_conv = layer.Conv2d(dim * 2, dim, kernel_size=1, bias=True, step_mode="m")
        self.lif = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.bn = _make_bn(num_features=dim, alpha=alpha, v_th=v_th)

    def forward(self, dec, enc):
        combined = torch.cat([dec, enc], dim=2)
        gate = torch.sigmoid(self.gate_conv(combined))
        fused = gate * dec + (1.0 - gate) * enc
        return self.bn(self.lif(fused))


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------
class VLIFNet(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=24,
                 en_num_blocks=(2, 2, 4, 4), de_num_blocks=(2, 2, 2, 2),
                 bias=False, T=4, use_refinement=False):
        super().__init__()
        functional.set_backend(self, backend=_BACKEND)
        functional.set_step_mode(self, step_mode="m")

        self.T = T
        self.use_refinement = use_refinement
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, T=T)
        self.encoder_level1 = SUNet_Level1_Block(dim=int(dim * 1))

        self.down1_2 = DownSampling(dim)
        self.encoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for _ in range(en_num_blocks[1])])

        self.down2_3 = DownSampling(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for _ in range(en_num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for _ in range(de_num_blocks[2])])

        self.up3_2 = UpSampling(int(dim * 2 ** 2))
        self.skip_fusion_level2 = GatedSkipFusion(dim=int(dim * 2 ** 1))
        self.decoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for _ in range(de_num_blocks[1])])

        self.up2_1 = UpSampling(int(dim * 2 ** 1))
        self.skip_fusion_level1 = GatedSkipFusion(dim=int(dim * 2 ** 0))
        self.decoder_level1 = SUNet_Level1_Block(dim=int(dim * 2 ** 0))
        self.additional_sunet_level1 = SUNet_Level1_Block(dim=int(dim * 2 ** 0))

        # Auxiliary output heads (v3 feature; parameters created but unused in v1)
        self.aux_head_level2 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=1)
        self.aux_head_level3 = nn.Conv2d(int(dim * 2 ** 2), out_channels, kernel_size=1)

        if self.use_refinement:
            self.refinement_blocks = nn.Sequential(*[
                Spiking_Residual_Block(dim=int(dim * 2 ** 0)) for _ in range(4)
            ])

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=int(dim * 2 ** 0), out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, inp_img, return_aux=False):
        # short keeps the ORIGINAL 4D input (used as the global residual); the
        # forward body broadcasts to 5D with the time dim.
        assert inp_img.dim() == 4, (
            f"VLIFNet expects [B, C, H, W] input for v1; got shape {tuple(inp_img.shape)}")
        short = inp_img.clone()
        inp_img = inp_img.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.skip_fusion_level2(inp_dec_level2, out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.skip_fusion_level1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.additional_sunet_level1(out_dec_level1)

        if self.use_refinement:
            out_dec_level1 = self.refinement_blocks(out_dec_level1)

        main_out = self.output(out_dec_level1.mean(0)) + short

        if return_aux:
            H, W = short.shape[-2], short.shape[-1]
            aux2 = self.aux_head_level2(
                F.interpolate(out_enc_level2.mean(0), size=(H, W),
                              mode="bilinear", align_corners=False)
            ) + short
            aux3 = self.aux_head_level3(
                F.interpolate(out_enc_level3.mean(0), size=(H, W),
                              mode="bilinear", align_corners=False)
            ) + short
            return main_out, aux2, aux3

        return main_out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_vlifnet(dim=24, en_num_blocks=(2, 2, 4, 4), de_num_blocks=(2, 2, 2, 2),
                  T=4, use_refinement=False,
                  inp_channels=3, out_channels=3,
                  backend="torch",
                  bn_variant="tdbn",
                  backbone="snn") -> VLIFNet:
    """Preferred VLIFNet constructor.

    * Does NOT force .cuda() — caller decides device.
    * Asserts T == 4.  Four upstream sub-blocks (TimeAttention(in_planes=16),
      Conv3d(kernel=(4,1,1)), MultiDimensionalAttention(T=4), FSTAModule(T=4))
      hard-code T=4 in their constructors.  Supporting T != 4 is scheduled
      for v3; surfacing the constraint early avoids an opaque reshape error
      at first forward.
    * `bn_variant`: either "tdbn" (default, threshold-dependent BN — what
      FLSNN and Zheng 2021 use) or "bn2d" (standard nn.BatchNorm2d wrapped
      to accept [T,B,C,H,W] input — for SC-16d ablation of Claim C16).
    * `backbone`: "snn" (default, LIF + MultiSpike4 quantisation) or "ann"
      (ReLU replacing every spike activation — for FLSNN §VI-B ANN-vs-SNN
      comparison inside our federated pipeline).
    """
    assert T == 4, (
        f"VLIFNet is hard-coded for T=4 in several sub-blocks (TimeAttention, "
        f"Conv3d time-compress, MultiDimensionalAttention, FSTAModule). "
        f"Got T={T}.  Supporting T!=4 is scheduled for v3.")
    set_vlifnet_backend(backend)
    # IMPORTANT: set BN and backbone variants BEFORE constructing VLIFNet.
    # _make_bn() and _make_lif_or_relu() read their module-level flags at
    # construction time, so these calls must happen first.
    set_vlifnet_bn_variant(bn_variant)
    set_vlifnet_backbone(backbone)
    net = VLIFNet(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        en_num_blocks=list(en_num_blocks),
        de_num_blocks=list(de_num_blocks),
        T=T,
        use_refinement=use_refinement,
    )
    # Belt-and-braces: LIFNode instances created before set_vlifnet_backend()
    # took effect (if any) are updated; no-op on 'torch'.
    functional.set_backend(net, backend=backend)
    functional.set_step_mode(net, step_mode="m")
    return net
