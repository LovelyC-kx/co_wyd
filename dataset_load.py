"""Shared DIV2K HR -> Gaussian PSF -> sampling -> noise pipeline.

Returns native LR and clean HR. Model-specific resizing belongs in sr_runtime,
so replacing a model cannot change the training pairs or validation noise.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from thz_protocol import Protocol, file_sha256


def read_rgb(path):
    with Image.open(path) as image:
        array = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def degrade(hr, protocol, generator):
    """Gaussian convolution, antialiased bicubic downsampling, then AWGN.

    Torch bicubic is an explicit sampler choice, not the author/MATLAB sampler.
    Reflection padding and crop-before-blur are also implementation choices.
    Float RGB HR is in [0,1]; noisy LR is NOT clipped by default.
    """
    _, h, w = hr.shape
    if h % protocol.scale or w % protocol.scale:
        raise ValueError('HR dimensions must be divisible by scale')
    radius = protocol.kernel_size // 2
    axis = torch.arange(-radius, radius + 1, dtype=hr.dtype, device=hr.device)
    kernel_1d = torch.exp(-axis.square() / (2 * protocol.blur_sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).expand(3, 1, -1, -1).contiguous()
    if min(h, w) <= radius:
        raise ValueError('Image/patch too small for PSF reflection padding')
    blurred = F.conv2d(F.pad(hr[None], (radius,) * 4, mode='reflect'),
                       kernel, groups=3)
    lr = F.interpolate(blurred, size=(h // protocol.scale, w // protocol.scale),
                       mode='bicubic', align_corners=False, antialias=True)[0]
    if protocol.noise_std:
        noise = torch.randn(lr.shape, generator=generator, dtype=lr.dtype)
        lr = lr + protocol.noise_std * noise
    return lr.clamp(0, 1) if protocol.clip_lr else lr


class DIV2KDataset(Dataset):
    def __init__(self, root, split, protocol: Protocol, allow_partial=False):
        if split not in ('train', 'valid'):
            raise ValueError('DIV2K split must be train or valid')
        self.root = Path(root)
        self.split = split
        self.protocol = protocol
        self.epoch = 0
        folder = self.root / f'DIV2K_{split}_HR'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing {folder}; download/extract HR archives, not LR archives')
        self.files = sorted(folder.glob('*.png'))
        expected = {f'{i:04d}.png' for i in
                    (range(1, 801) if split == 'train' else range(801, 901))}
        actual = {path.name for path in self.files}
        if not self.files or not actual <= expected:
            raise ValueError(f'Wrong DIV2K IDs in {folder}; expected '
                             + ('0001..0800' if split == 'train' else '0801..0900'))
        if not allow_partial and actual != expected:
            raise ValueError(f'{folder}: expected {len(expected)} HR PNGs, got {len(actual)}. '
                             '--allow-partial is for smoke tests only')
        self.complete = actual == expected

    def __len__(self):
        return len(self.files)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        path = self.files[index]
        # Model/dropout RNG and worker scheduling cannot change the pairs.
        # Validation always uses epoch 0 and gives fixed full-image pairs.
        epoch = self.epoch if self.split == 'train' else 0
        sequence = np.random.SeedSequence([self.protocol.seed, epoch, int(path.stem),
                                          0 if self.split == 'train' else 1])
        crop_seed, noise_seed = sequence.generate_state(2, dtype=np.uint64)
        rng = np.random.default_rng(crop_seed)
        generator = torch.Generator().manual_seed(int(noise_seed))
        hr = read_rgb(path)
        _, h, w = hr.shape
        if self.split == 'train':
            patch = self.protocol.hr_patch_size
            if min(h, w) < patch:
                raise ValueError(f'{path}: smaller than HR patch {patch}; no implicit resizing')
            top = int(rng.integers(0, h - patch + 1))
            left = int(rng.integers(0, w - patch + 1))
            hr = hr[:, top:top + patch, left:left + patch]
            if self.protocol.augment:
                if rng.random() < .5:
                    hr = hr.flip(-1)
                if rng.random() < .5:
                    hr = hr.flip(-2)
                hr = hr.rot90(int(rng.integers(4)), (-2, -1))
        else:
            scale = self.protocol.scale
            hr = hr[:, :h - h % scale, :w - w % scale]
        hr = hr.contiguous()
        return {'lr': degrade(hr, self.protocol, generator), 'hr': hr, 'name': path.name}

    def manifest(self):
        return {'split': self.split, 'complete': self.complete,
                'files': [{'name': path.name, 'sha256': file_sha256(path)}
                          for path in self.files]}


class PairedImageDataset(Dataset):
    """Real/native LR inputs, with optional measured HR references by stem.

    No synthetic degradation is applied. Without a target, this is inference-only.
    Raw signed detector arrays need an independently specified calibration.
    """
    def __init__(self, input_dir, scale, target_dir=None):
        self.scale = scale
        self.inputs = self._images(input_dir)
        self.targets = self._images(target_dir) if target_dir else None
        if self.targets is not None and self.targets.keys() != self.inputs.keys():
            raise ValueError('Input and target filename stems must match exactly')
        self.names = sorted(self.inputs)

    @staticmethod
    def _images(folder):
        paths = sorted(path for path in Path(folder).iterdir()
                       if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        images = {path.stem: path for path in paths}
        if not images or len(images) != len(paths):
            raise ValueError(f'No images or duplicate filename stems in {folder}')
        for path in paths:
            with Image.open(path) as image:
                if image.mode not in ('RGB', 'L', 'RGBA', 'P'):
                    raise ValueError(f'{path}: convert/calibrate raw or high-bit-depth data explicitly')
        return images

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        lr = read_rgb(self.inputs[name])
        sample = {'lr': lr, 'name': name + '.png'}
        if self.targets is not None:
            hr = read_rgb(self.targets[name])
            if hr.shape[-2:] != tuple(v * self.scale for v in lr.shape[-2:]):
                raise ValueError(f'{name}: target must have scale× native LR height/width')
            sample['hr'] = hr
        return sample
