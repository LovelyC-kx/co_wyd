"""Shared Y-channel PSNR/SSIM on float predictions (not saved PNGs).

YCbCr Y follows MATLAB/BT.601 studio-range coefficients. Border=0 and valid
11x11 Gaussian SSIM are declared choices absent from the THz-Net manuscript.
"""
import math
import numpy as np
from scipy.ndimage import correlate
import torch


def _numpy_rgb(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.ndim == 3:
            image = image.transpose(1, 2, 0)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError('Expected finite HWC RGB or CHW torch RGB')
    return image


def rgb_to_y(image):
    image = _numpy_rgb(image)
    return (image @ np.array([65.481, 128.553, 24.966]) + 16.0) / 255.0


def y_metrics(prediction, target, border=0):
    pred = _numpy_rgb(prediction)
    ref = _numpy_rgb(target)
    if pred.shape != ref.shape or border < 0:
        raise ValueError('Matching shapes and nonnegative border required')
    # Clamp only prediction at evaluation; training MSE never clamps.
    pred = rgb_to_y(pred.clip(0, 1))
    ref = rgb_to_y(ref)
    if border:
        pred = pred[border:-border, border:-border]
        ref = ref[border:-border, border:-border]
    if min(pred.shape, default=0) < 11:
        raise ValueError('SSIM needs at least 11 pixels after metric border crop')
    mse = np.mean((pred - ref) ** 2)
    psnr = float('inf') if mse == 0 else -10 * math.log10(float(mse))
    axis = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-axis ** 2 / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    window = np.outer(kernel, kernel)

    def smooth(image):
        return correlate(image, window, mode='reflect')[5:-5, 5:-5]

    mu_p, mu_r = smooth(pred), smooth(ref)
    var_p = smooth(pred * pred) - mu_p ** 2
    var_r = smooth(ref * ref) - mu_r ** 2
    cov = smooth(pred * ref) - mu_p * mu_r
    c1, c2 = .01 ** 2, .03 ** 2
    ssim = ((2 * mu_p * mu_r + c1) * (2 * cov + c2) /
            ((mu_p ** 2 + mu_r ** 2 + c1) * (var_p + var_r + c2))).mean()
    return {'psnr_y': psnr, 'ssim_y': float(ssim)}


def json_scores(scores):
    # Standards-compliant JSON; identity baseline has +inf PSNR, represented as a string.
    return {key: (value if math.isfinite(value) else str(value))
            for key, value in scores.items()}
