"""spikingjelly 0.0.0.0.14 compatibility shims.

Fixes two bugs that prevent VLIFNet from running on the current PyPI
release of spikingjelly.  Applied by being imported at module-load time
from ``vlifnet.py`` BEFORE any spikingjelly class is used.

Bug 1: ThresholdDependentBatchNorm{1,2,3}d forgot to override
       _check_input_dim — seq_to_ann_forward raises NotImplementedError
       on every torch version.
Bug 2: MultiDimensionalAttention inherits only from MultiStepModule
       (not from nn.Module), so instances are not callable and their
       learnable parameters are orphaned (never registered with the
       parent Module).  We replace the class with a proper nn.Module
       CBAM-style attention over [T, B, C, H, W].

Both fixes are applied unconditionally at import; re-importing this
module is idempotent.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer as _sjlayer


# ---------- TDBN: _check_input_dim overrides -------------------------------

def _tdbn_check_1d(self, x):
    if x.dim() not in (2, 3):
        raise ValueError(f"expected 2D or 3D input (got {x.dim()}D)")


def _tdbn_check_2d(self, x):
    if x.dim() != 4:
        raise ValueError(f"expected 4D input (got {x.dim()}D)")


def _tdbn_check_3d(self, x):
    if x.dim() != 5:
        raise ValueError(f"expected 5D input (got {x.dim()}D)")


if hasattr(_sjlayer, "ThresholdDependentBatchNorm1d"):
    _sjlayer.ThresholdDependentBatchNorm1d._check_input_dim = _tdbn_check_1d
if hasattr(_sjlayer, "ThresholdDependentBatchNorm2d"):
    _sjlayer.ThresholdDependentBatchNorm2d._check_input_dim = _tdbn_check_2d
if hasattr(_sjlayer, "ThresholdDependentBatchNorm3d"):
    _sjlayer.ThresholdDependentBatchNorm3d._check_input_dim = _tdbn_check_3d


# ---------- MultiDimensionalAttention replacement --------------------------

class _MDAttention(nn.Module):
    """Drop-in replacement for spikingjelly's buggy MultiDimensionalAttention.

    Three-axis attention (channel + spatial + temporal), CBAM-style.
    Input / output shape: [T, B, C, H, W].

    This implementation differs from upstream in exact numerics but keeps
    the same shape contract and same set of learnable parameters
    (reduction_t, reduction_c, kernel_size, T, C), so a state_dict from
    a future-fixed spikingjelly would need a re-training rather than a
    drop-in load — acceptable for v1 (no pre-trained checkpoints exist).
    """

    def __init__(self, T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=32):
        super().__init__()
        c_mid = max(1, C // reduction_c)
        t_mid = max(1, T // reduction_t)
        self.c_mlp = nn.Sequential(
            nn.Linear(C, c_mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_mid, C, bias=False),
        )
        pad = kernel_size // 2
        self.s_conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.t_mlp = nn.Sequential(
            nn.Linear(T, t_mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(t_mid, T, bias=False),
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Channel SE attention
        c_pool = x.mean(dim=(0, 3, 4))               # [B, C]
        c_att = torch.sigmoid(self.c_mlp(c_pool))    # [B, C]
        x = x * c_att.view(1, B, C, 1, 1)
        # Spatial (CBAM) attention
        s_avg = x.mean(dim=2, keepdim=True)          # [T, B, 1, H, W]
        s_max = x.amax(dim=2, keepdim=True)
        s_cat = torch.cat([s_avg, s_max], dim=2).reshape(T * B, 2, H, W)
        s_att = torch.sigmoid(self.s_conv(s_cat)).reshape(T, B, 1, H, W)
        x = x * s_att
        # Temporal attention
        t_pool = x.mean(dim=(2, 3, 4)).mean(dim=1)   # [T]
        t_att = torch.sigmoid(self.t_mlp(t_pool))    # [T]
        x = x * t_att.view(T, 1, 1, 1, 1)
        return x


_sjlayer.MultiDimensionalAttention = _MDAttention
