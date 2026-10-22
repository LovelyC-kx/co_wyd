"""VLIFNet model package — vendored from system1_VLIFNet, plus a plain
ANN U-Net baseline (plain_unet.py) used for the §IV ablation.

The plain U-Net deliberately depends on no spikingjelly modules so that
importing it works on a torch-only environment (e.g. unit tests on CPU).
"""

from .vlifnet import (
    VLIFNet,
    build_vlifnet,
    set_vlifnet_backend,
    MultiSpike4,
    mem_update,
    Spiking_Residual_Block,
    SUNet_Level1_Block,
    GatedSkipFusion,
    DownSampling,
    UpSampling,
    OverlapPatchEmbed,
    TimeAttention,
    PixelShuffleLIFBlock,
)
from .plain_unet import PlainUNet, build_plain_unet

__all__ = [
    "VLIFNet",
    "build_vlifnet",
    "set_vlifnet_backend",
    "MultiSpike4",
    "mem_update",
    "Spiking_Residual_Block",
    "SUNet_Level1_Block",
    "GatedSkipFusion",
    "DownSampling",
    "UpSampling",
    "OverlapPatchEmbed",
    "TimeAttention",
    "PixelShuffleLIFBlock",
    "PlainUNet",
    "build_plain_unet",
]
