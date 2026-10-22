"""
Plain ANN U-Net baseline for cloud removal — used as the "no-attention,
no-spike, no-frequency-module" reference in Tab 1.

Design choices (matched to VLIFNet for fair capacity comparison):
  * dim = 24 base channels
  * encoder block counts [2, 2, 4]  (3 levels, doubling channels at each
    downsample)  — VLIFNet's U-Net is also 3 levels
  * decoder block counts [2, 2, 2]  symmetric
  * conv block = 3x3 Conv → BN → ReLU → 3x3 Conv → BN → residual
  * skip fusion = additive (cheaper than VLIFNet's gated fusion;
    deliberate — this is the *baseline*, not the proposed model)
  * final output = Conv(3) + global residual to input image

Resulting parameter count at dim=24 / [2,2,4]/[2,2,2]: 1,311,603 ≈ 1.31 M
(measured; VLIFNet is 2,302,901 ≈ 2.30 M).  The two backbones are therefore
capacity-comparable only in the loose sense (same width/depth skeleton);
VLIFNet carries ~1.75x the parameters, and PSNR/SSIM comparisons should
say "capacity-comparable", not "matched".
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """3x3 → BN → ReLU → 3x3 → BN → + skip → ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class DownBlock(nn.Module):
    """Stride-2 conv that doubles channels."""

    def __init__(self, in_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * 2, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_c * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class UpBlock(nn.Module):
    """Bilinear up + 3x3 conv that halves channels."""

    def __init__(self, in_c: int):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c // 2, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_c // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ---------------------------------------------------------------------------
# Plain U-Net
# ---------------------------------------------------------------------------

class PlainUNet(nn.Module):
    """3-level plain ANN U-Net for cloud removal.

    Forward: [B, 3, H, W] → [B, 3, H, W].  H, W must be divisible by 4
    (two stride-2 downsamples).
    """

    def __init__(self,
                 inp_channels: int = 3,
                 out_channels: int = 3,
                 dim: int = 24,
                 en_blocks: Sequence[int] = (2, 2, 4),
                 de_blocks: Sequence[int] = (2, 2, 2)):
        super().__init__()
        assert len(en_blocks) == 3 and len(de_blocks) == 3, \
            "PlainUNet uses a 3-level U-Net to match VLIFNet's depth"

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoder_level1 = nn.Sequential(*[ResidualBlock(dim) for _ in range(en_blocks[0])])
        self.down1_2 = DownBlock(dim)
        self.encoder_level2 = nn.Sequential(*[ResidualBlock(dim * 2) for _ in range(en_blocks[1])])
        self.down2_3 = DownBlock(dim * 2)
        self.encoder_level3 = nn.Sequential(*[ResidualBlock(dim * 4) for _ in range(en_blocks[2])])

        # Decoder (symmetric)
        self.decoder_level3 = nn.Sequential(*[ResidualBlock(dim * 4) for _ in range(de_blocks[2])])
        self.up3_2 = UpBlock(dim * 4)
        self.decoder_level2 = nn.Sequential(*[ResidualBlock(dim * 2) for _ in range(de_blocks[1])])
        self.up2_1 = UpBlock(dim * 2)
        self.decoder_level1 = nn.Sequential(*[ResidualBlock(dim) for _ in range(de_blocks[0])])

        # Output head
        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"PlainUNet expects [B, C, H, W]; got {tuple(x.shape)}"
        H, W = x.shape[-2:]
        assert H % 4 == 0 and W % 4 == 0, \
            f"PlainUNet requires H, W divisible by 4; got {H}x{W}"

        short = x

        # Encoder
        e1 = self.patch_embed(x)
        e1 = self.encoder_level1(e1)
        e2 = self.down1_2(e1)
        e2 = self.encoder_level2(e2)
        e3 = self.down2_3(e2)
        e3 = self.encoder_level3(e3)

        # Decoder
        d3 = self.decoder_level3(e3)
        d2 = self.up3_2(d3) + e2
        d2 = self.decoder_level2(d2)
        d1 = self.up2_1(d2) + e1
        d1 = self.decoder_level1(d1)

        return self.output(d1) + short


def build_plain_unet(dim: int = 24,
                     en_blocks: Sequence[int] = (2, 2, 4),
                     de_blocks: Sequence[int] = (2, 2, 2),
                     inp_channels: int = 3,
                     out_channels: int = 3) -> PlainUNet:
    """Factory matching the build_vlifnet convention."""
    return PlainUNet(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        en_blocks=tuple(en_blocks),
        de_blocks=tuple(de_blocks),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    net = build_plain_unet(dim=24)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    x = torch.randn(2, 3, 64, 64)
    y = net(x)
    print(f"PlainUNet  params = {n_params:,}  ({n_params / 1e6:.2f} M)")
    print(f"input  shape = {tuple(x.shape)}")
    print(f"output shape = {tuple(y.shape)}")
    assert y.shape == x.shape, "output shape must match input"
    print("OK.")
