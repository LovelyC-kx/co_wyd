"""Model boundary only: native LR -> optional bicubic HR -> uploaded network.

Future models can share dataset_load unchanged using factory + input_mode.
"""
import importlib
import torch
from torch import nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional


def build_model(spec, device, backend='torch'):
    module_name, factory_name = spec['factory'].split(':')
    factory = getattr(importlib.import_module(module_name), factory_name)
    network = factory(**spec['kwargs'])
    if not isinstance(network, nn.Module):
        raise TypeError('Model factory must return torch.nn.Module')
    network.to(device)
    functional.set_step_mode(network, step_mode='m')
    if backend == 'cupy' and device.type != 'cuda':
        raise ValueError('CuPy backend requires CUDA; use --backend torch on CPU')
    functional.set_backend(network, backend=backend)
    return network


class SRAdapter(nn.Module):
    def __init__(self, network, scale, input_mode='bicubic', spatial_multiple=8):
        super().__init__()
        if input_mode not in ('bicubic', 'native') or spatial_multiple < 1:
            raise ValueError('Invalid model interface')
        self.network = network
        self.scale = scale
        self.input_mode = input_mode
        self.spatial_multiple = spatial_multiple

    def reset(self):
        functional.reset_net(self.network)

    def prepare(self, lr):
        if self.input_mode == 'bicubic':
            return F.interpolate(lr, scale_factor=self.scale, mode='bicubic',
                                 align_corners=False, antialias=False)
        return lr

    def forward_prepared(self, image):
        h, w = image.shape[-2:]
        multiple = self.spatial_multiple
        pad_h, pad_w = -h % multiple, -w % multiple
        # Replication also works for images smaller than the padding size.
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
        restored = self.network(image)
        factor = 1 if self.input_mode == 'bicubic' else self.scale
        expected = (image.shape[0], 3, image.shape[-2] * factor, image.shape[-1] * factor)
        if tuple(restored.shape) != expected:
            raise ValueError(f'Model output {tuple(restored.shape)} != expected {expected}; '
                             'check bicubic/native input mode')
        return restored[..., :h * factor, :w * factor]

    def forward(self, lr):
        return self.forward_prepared(self.prepare(lr))


@torch.no_grad()
def restore(adapter, lr, tile_size=192, overlap=32):
    """Tiled/full evaluation. Tiles measured in model-input pixels (HR for VLIF).

    State is reset for every independent tile/image. Uniform overlap averaging
    is fixed, recorded and shared by validation/test. Tiling is not in the paper.
    """
    if tile_size < 0 or overlap < 0 or (tile_size and overlap >= tile_size):
        raise ValueError('Require tile_size=0 or 0 <= overlap < tile_size')
    image = adapter.prepare(lr)
    h, w = image.shape[-2:]
    if not tile_size or (h <= tile_size and w <= tile_size):
        adapter.reset()
        try:
            return adapter.forward_prepared(image)
        finally:
            adapter.reset()

    def positions(length):
        if length <= tile_size:
            return [0]
        return list(range(0, length - tile_size, tile_size - overlap)) + [length - tile_size]

    factor = 1 if adapter.input_mode == 'bicubic' else adapter.scale
    total = image.new_zeros(image.shape[0], 3, h * factor, w * factor)
    count = image.new_zeros(1, 1, h * factor, w * factor)
    for top in positions(h):
        for left in positions(w):
            tile = image[..., top:top + tile_size, left:left + tile_size]
            adapter.reset()
            try:
                prediction = adapter.forward_prepared(tile)
            finally:
                adapter.reset()
            y, x = top * factor, left * factor
            hh, ww = prediction.shape[-2:]
            total[..., y:y + hh, x:x + ww] += prediction
            count[..., y:y + hh, x:x + ww] += 1
    return total / count


@torch.no_grad()
def evaluate(adapter, dataset, device, tile_size=192, overlap=32, border=0,
             workers=0, max_images=None, save_dir=None):
    from pathlib import Path
    from PIL import Image
    import numpy as np
    from torch.utils.data import DataLoader
    from metrics import y_metrics

    adapter.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(loader):
        if max_images is not None and index >= max_images:
            break
        lr = batch['lr'].to(device)
        restored = restore(adapter, lr, tile_size, overlap).clamp(0, 1)
        row = {'name': batch['name'][0]}
        if 'hr' in batch:
            target = batch['hr'][0]
            row.update(y_metrics(restored[0], target, border))
            bicubic = F.interpolate(lr, scale_factor=adapter.scale, mode='bicubic',
                                    align_corners=False, antialias=False)[0]
            row.update({'bicubic_' + key: value for key, value in
                        y_metrics(bicubic, target, border).items()})
        if save_dir is not None:
            array = restored[0].cpu().permute(1, 2, 0).numpy()
            Image.fromarray(np.rint(array * 255).astype(np.uint8)).save(
                Path(save_dir) / batch['name'][0])
        rows.append(row)
    if not rows:
        raise ValueError('Evaluation selected zero images')
    keys = [key for key in rows[0] if key != 'name']
    means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    return means, rows
