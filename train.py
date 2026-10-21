"""Train the uploaded VLIF model on the declared THz-Net-style DIV2K protocol."""
import argparse
from dataclasses import replace
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_load import DIV2KDataset
from metrics import json_scores
from sr_runtime import SRAdapter, build_model, evaluate
from thz_protocol import Protocol, file_sha256, fingerprint, write_json


DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'thznet_div2k.json'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--data-root', required=True, type=Path)
    result.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    result.add_argument('--output', type=Path, default=Path('runs/vlif_div2k_x2'))
    result.add_argument('--scale', type=int, choices=(2, 3, 4))
    result.add_argument('--epochs', type=int)
    result.add_argument('--batch-size', type=int)
    result.add_argument('--patch-size', type=int, help='HR patch size; manuscript leaves HR/LR convention unspecified')
    result.add_argument('--lr', type=float)
    result.add_argument('--blur-sigma', type=float, help='Chosen PSF sigma in HR pixels; NOT specified in paper')
    result.add_argument('--noise-std', type=float, help='Chosen noise std in normalized [0,1] units; NOT specified in paper')
    result.add_argument('--workers', type=int, default=2)
    result.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    result.add_argument('--backend', choices=('torch', 'cupy'), default='torch')
    result.add_argument('--model-factory', default='model:model')
    result.add_argument('--model-kwargs', default='{}', help='JSON object of model constructor arguments')
    result.add_argument('--model-input', choices=('bicubic', 'native'), default='bicubic')
    result.add_argument('--spatial-multiple', type=int, default=8)
    result.add_argument('--tile-size', type=int, default=192, help='Model-input pixels; 0 disables tiling')
    result.add_argument('--overlap', type=int, default=32)
    result.add_argument('--val-every', type=int, default=1)
    result.add_argument('--resume', type=Path)
    result.add_argument('--allow-partial', action='store_true', help='Smoke tests only; incomplete data are marked in manifest')
    result.add_argument('--max-train-steps', type=int, help='Smoke tests only')
    result.add_argument('--max-val-images', type=int, help='Smoke tests only')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('val_every', 'spatial_multiple'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('max_train_steps', 'max_val_images'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.workers < 0 or args.tile_size < 0 or args.overlap < 0 or (
            args.tile_size and args.overlap >= args.tile_size):
        raise ValueError('Invalid workers/tile-size/overlap')
    overrides = {field: getattr(args, arg) for arg, field in (
        ('scale', 'scale'), ('epochs', 'epochs'), ('batch_size', 'batch_size'),
        ('patch_size', 'hr_patch_size'), ('lr', 'learning_rate'),
        ('blur_sigma', 'blur_sigma'), ('noise_std', 'noise_std'))
        if getattr(args, arg) is not None}
    protocol = replace(Protocol.load(args.config), **overrides)
    spec = {'factory': args.model_factory, 'kwargs': json.loads(args.model_kwargs),
            'input_mode': args.model_input, 'spatial_multiple': args.spatial_multiple}
    if not isinstance(spec['kwargs'], dict):
        raise ValueError('--model-kwargs must be a JSON object')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Output directory is nonempty; choose a new run or use --resume')
    device = torch.device(args.device)
    random.seed(protocol.seed)
    torch.manual_seed(protocol.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_set = DIV2KDataset(args.data_root, 'train', protocol, args.allow_partial)
    valid_set = DIV2KDataset(args.data_root, 'valid', protocol, args.allow_partial)
    datasets = {'train': train_set.manifest(), 'valid': valid_set.manifest()}
    source_dir = Path(__file__).parent
    data_code = {name: file_sha256(source_dir / name) for name in
                 ('dataset_load.py', 'thz_protocol.py')}
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'Pillow', 'scipy', 'spikingjelly', 'einops')}
    data_hash = fingerprint({'protocol': protocol.to_dict(), 'datasets': datasets,
                             'data_code': data_code, 'versions': versions})
    network = build_model(spec, device, args.backend)
    adapter = SRAdapter(network, protocol.scale, spec['input_mode'], spec['spatial_multiple'])
    optimizer = torch.optim.Adam(network.parameters(), lr=protocol.learning_rate,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    criterion = nn.MSELoss()
    start_epoch, best_psnr = 1, -float('inf')
    model_source = Path(inspect.getfile(type(network)))
    execution = {'tile_size': args.tile_size, 'overlap': args.overlap,
                 'val_every': args.val_every, 'max_train_steps': args.max_train_steps,
                 'max_val_images': args.max_val_images}
    source_hashes = {name: file_sha256(source_dir / name) for name in
                     ('train.py', 'test.py', 'sr_runtime.py', 'metrics.py')}
    source_hashes['model_source'] = file_sha256(model_source)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if (checkpoint['data_fingerprint'] != data_hash or checkpoint['model_spec'] != spec
                or checkpoint['execution'] != execution or checkpoint['source_hashes'] != source_hashes):
            raise ValueError('Resume requires identical data, protocol, source, model and execution settings')
        network.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['best_psnr_y']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if device.type == 'cuda' and checkpoint['cuda_rng_state']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    if start_epoch > protocol.epochs:
        raise ValueError('Checkpoint has already completed the configured epochs')
    manifest = {
        'protocol': protocol.to_dict(), 'model_spec': spec, 'versions': versions,
        'data_fingerprint': data_hash, 'datasets': datasets, 'data_code': data_code,
        'source_hashes': source_hashes, 'execution': execution,
        'device': str(device), 'backend': args.backend,
        'parameter_count': sum(p.numel() for p in network.parameters()),
        'smoke_test': bool(args.allow_partial or args.max_train_steps or args.max_val_images),
        'author_pipeline_exact': False,
        'unpublished_author_details': ['PSF sigma/size', 'noise std', 'sampling implementation',
                                      'patch HR/LR convention', 'Y conversion/SSIM/border details'],
        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'manifest.json', manifest)
    print('Author degradation values are unpublished. This is the declared protocol, not an exact numerical reproduction.')
    print(f'Data fingerprint: {data_hash}; parameters: {manifest["parameter_count"]:,}', flush=True)
    for epoch in range(start_epoch, protocol.epochs + 1):
        start = time.monotonic()
        train_set.set_epoch(epoch)
        generator = torch.Generator().manual_seed(protocol.seed + epoch)
        loader = DataLoader(train_set, batch_size=protocol.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=False,
                            generator=generator, pin_memory=device.type == 'cuda')
        adapter.train()
        total_loss, seen, steps = 0., 0, 0
        for index, batch in enumerate(loader):
            if args.max_train_steps is not None and index >= args.max_train_steps:
                break
            lr, hr = batch['lr'].to(device), batch['hr'].to(device)
            adapter.reset()
            optimizer.zero_grad(set_to_none=True)
            try:
                prediction = adapter(lr)
                if prediction.shape != hr.shape:
                    raise ValueError('Prediction and clean HR target shapes differ')
                loss = criterion(prediction, hr)  # RGB MSE, no clamp and no SSIM loss.
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                optimizer.step()
            finally:
                adapter.reset()
            total_loss += float(loss.detach()) * lr.shape[0]
            seen += lr.shape[0]
            steps += 1
        row = {'epoch': epoch, 'mse_rgb': total_loss / seen,
               'train_images': seen, 'steps': steps, 'lr': optimizer.param_groups[0]['lr']}
        is_best = False
        if epoch % args.val_every == 0 or epoch == protocol.epochs:
            means, rows = evaluate(adapter, valid_set, device, args.tile_size,
                                   args.overlap, protocol.metric_border, args.workers,
                                   args.max_val_images)
            row.update(json_scores(means))
            write_json(args.output / 'validation_latest.json', {
                'epoch': epoch, 'data_fingerprint': data_hash,
                'mean': json_scores(means),
                'images': [dict(name=item['name'], **json_scores(
                    {key: value for key, value in item.items() if key != 'name'})) for item in rows]})
            if means['psnr_y'] > best_psnr:
                best_psnr = means['psnr_y']
                is_best = True
        row['seconds'] = time.monotonic() - start
        with (args.output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        checkpoint = {
            'epoch': epoch, 'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(), 'best_psnr_y': best_psnr,
            'protocol': protocol.to_dict(), 'model_spec': spec,
            'data_fingerprint': data_hash, 'datasets': datasets, 'data_code': data_code, 'execution': execution,
            'source_hashes': source_hashes, 'versions': versions,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
        }
        torch.save(checkpoint, args.output / 'model_last.pth')
        if is_best:
            torch.save(checkpoint, args.output / 'model_best.pth')
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
