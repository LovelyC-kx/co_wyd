"""
PSNR / SSIM evaluation for the v1 cloud-removal pipeline.

Three evaluation modes:

    center_patch  (v1 default)
        Centre-crop each test image to `eval_patch_size`² and score one
        forward pass.  Matches the training distribution (VLIFNet was
        trained on 64×64 patches) and fits V100 16 GB for any test res.

    sliding
        Overlapping sliding-window inference with raised-cosine blending.
        Memory-safe for arbitrary resolutions; slower.

    fullimage
        Forward on the native image.  Will OOM VLIFNet dim=24 at 512²
        on V100 16 GB; only use when GPU has sufficient RAM.

Metrics:
    PSNR in dB (clamp-to-[0, 1] before computation)
    SSIM 11-tap Gaussian-window, σ = 1.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from .task import SSIMLoss


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    psnr_per_image: List[float] = field(default_factory=list)
    ssim_per_image: List[float] = field(default_factory=list)

    @property
    def mean_psnr(self) -> float:
        return float(np.mean(self.psnr_per_image)) if self.psnr_per_image else float("nan")

    @property
    def mean_ssim(self) -> float:
        return float(np.mean(self.ssim_per_image)) if self.ssim_per_image else float("nan")

    def summary(self) -> str:
        n = len(self.psnr_per_image)
        return (f"n={n}  PSNR mean={self.mean_psnr:.4f}  "
                f"SSIM mean={self.mean_ssim:.4f}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _torch_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR (dB) for pairs in [0, 1], shape [C, H, W].  The MSE floor
    clamps PSNR at ~100 dB and avoids a GPU sync for the zero-MSE
    branch that a Python-level `if` would introduce."""
    pred   = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = (pred - target).pow(2).mean().clamp(min=1e-10)
    return 20.0 * torch.log10(1.0 / mse.sqrt())


_SSIM_LOSS = SSIMLoss()


def _torch_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred   = pred.clamp(0.0, 1.0).unsqueeze(0)
    target = target.clamp(0.0, 1.0).unsqueeze(0)
    with torch.no_grad():
        ssim_loss = _SSIM_LOSS(pred, target)
    return 1.0 - ssim_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _center_crop(x: torch.Tensor, p: int) -> torch.Tensor:
    if x.dim() == 4:
        _, _, H, W = x.shape
    elif x.dim() == 3:
        _, H, W = x.shape
    else:
        raise ValueError(f"_center_crop expects 3D/4D tensor, got {x.shape}")
    assert H >= p and W >= p, f"image too small ({H}x{W}) for centre crop {p}"
    top, left = (H - p) // 2, (W - p) // 2
    if x.dim() == 4:
        return x[:, :, top:top + p, left:left + p]
    return x[:, top:top + p, left:left + p]


def _raised_cosine_window(H: int, W: int, device, dtype) -> torch.Tensor:
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    wy = 0.5 - 0.5 * torch.cos(2 * math.pi * (y + 0.5) / H)
    wx = 0.5 - 0.5 * torch.cos(2 * math.pi * (x + 0.5) / W)
    return wy.unsqueeze(1) * wx.unsqueeze(0)


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_centerpatch(model: nn.Module,
                         loader: DataLoader,
                         patch_size: int,
                         device: Optional[torch.device] = None,
                         ) -> EvalResult:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    res = EvalResult()
    for batch in loader:
        cloudy, clear = batch
        cloudy = cloudy.to(device, non_blocking=True)
        clear  = clear.to(device,  non_blocking=True)
        cloudy = _center_crop(cloudy, patch_size)
        clear  = _center_crop(clear,  patch_size)
        functional.reset_net(model)
        pred = model(cloudy)
        functional.reset_net(model)
        for i in range(pred.shape[0]):
            res.psnr_per_image.append(_torch_psnr(pred[i], clear[i]).item())
            res.ssim_per_image.append(_torch_ssim(pred[i], clear[i]).item())
    return res


@torch.no_grad()
def evaluate_fullimage(model: nn.Module,
                       loader: DataLoader,
                       device: Optional[torch.device] = None,
                       ) -> EvalResult:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    res = EvalResult()
    for batch in loader:
        cloudy, clear = batch
        cloudy = cloudy.to(device, non_blocking=True)
        clear  = clear.to(device,  non_blocking=True)
        functional.reset_net(model)
        pred = model(cloudy)
        functional.reset_net(model)
        for i in range(pred.shape[0]):
            res.psnr_per_image.append(_torch_psnr(pred[i], clear[i]).item())
            res.ssim_per_image.append(_torch_ssim(pred[i], clear[i]).item())
    return res


@torch.no_grad()
def _sliding_forward(model: nn.Module, img: torch.Tensor,
                     window: int, stride: int,
                     device: torch.device) -> torch.Tensor:
    C, H, W = img.shape
    assert H >= window and W >= window, \
        f"sliding window {window} larger than image {H}×{W}"
    out  = torch.zeros_like(img)
    norm = torch.zeros((1, H, W), device=device, dtype=img.dtype)
    ys = list(range(0, max(1, H - window + 1), stride))
    xs = list(range(0, max(1, W - window + 1), stride))
    if ys[-1] != H - window:
        ys.append(H - window)
    if xs[-1] != W - window:
        xs.append(W - window)
    w = _raised_cosine_window(window, window, device, img.dtype)
    for y0 in ys:
        for x0 in xs:
            patch = img[:, y0:y0 + window, x0:x0 + window].unsqueeze(0)
            functional.reset_net(model)
            pred = model(patch)[0]
            functional.reset_net(model)
            out [:, y0:y0 + window, x0:x0 + window] += pred * w
            norm[:, y0:y0 + window, x0:x0 + window] += w
    return out / norm.clamp(min=1e-8)


@torch.no_grad()
def evaluate_sliding(model: nn.Module,
                     loader: DataLoader,
                     window: int = 64,
                     stride: int = 32,
                     device: Optional[torch.device] = None,
                     ) -> EvalResult:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    res = EvalResult()
    for batch in loader:
        cloudy, clear = batch
        assert cloudy.shape[0] == 1, (
            f"evaluate_sliding requires batch_size=1; got {cloudy.shape[0]}")
        cloudy = cloudy[0].to(device, non_blocking=True)
        clear  = clear[0].to(device,  non_blocking=True)
        pred = _sliding_forward(model, cloudy, window, stride, device)
        res.psnr_per_image.append(_torch_psnr(pred, clear).item())
        res.ssim_per_image.append(_torch_ssim(pred, clear).item())
    return res


# ---------------------------------------------------------------------------
# Plane-by-plane evaluation + cross-plane averaging
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_per_plane(constellation,
                       loader: DataLoader,
                       *,
                       mode: str = "center_patch",
                       patch_size: int = 64,
                       window: int = 64,
                       stride: int = 32,
                       device: Optional[torch.device] = None,
                       ) -> List[EvalResult]:
    assert mode in ("center_patch", "fullimage", "sliding")
    results: List[EvalResult] = []
    for p in range(constellation.num_planes):
        model = constellation.planes[p][0].model
        if mode == "center_patch":
            results.append(evaluate_centerpatch(model, loader, patch_size, device))
        elif mode == "sliding":
            one = DataLoader(loader.dataset, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)
            results.append(evaluate_sliding(model, one, window, stride, device))
        else:
            results.append(evaluate_fullimage(model, loader, device))
    return results


def average_eval_results(results: List[EvalResult]) -> EvalResult:
    """Element-wise mean across planes (same test set)."""
    if not results:
        return EvalResult()
    n = len(results[0].psnr_per_image)
    out = EvalResult()
    for i in range(n):
        out.psnr_per_image.append(float(np.mean([r.psnr_per_image[i] for r in results])))
        out.ssim_per_image.append(float(np.mean([r.ssim_per_image[i] for r in results])))
    return out
