"""Evaluate held-out DIV2K or restore real native LR images with a saved protocol."""
import argparse
from pathlib import Path
import torch
from dataset_load import DIV2KDataset, PairedImageDataset
from metrics import json_scores
from sr_runtime import SRAdapter, build_model, evaluate
from thz_protocol import Protocol, file_sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-root', type=Path, help='Re-evaluate the 100 DIV2K validation HR images')
    group.add_argument('--input-dir', type=Path, help='Real/native LR images; no synthetic degradation applied')
    parser.add_argument('--target-dir', type=Path, help='Optional measured HR references for real inputs')
    parser.add_argument('--output', type=Path, default=Path('results/vlif_div2k_x2'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--backend', choices=('torch', 'cupy'), default='torch')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--tile-size', type=int, help='Defaults to checkpoint validation setting')
    parser.add_argument('--overlap', type=int, help='Defaults to checkpoint validation setting')
    parser.add_argument('--allow-partial', action='store_true', help='Smoke tests only')
    parser.add_argument('--max-images', type=int, help='Smoke tests only')
    args = parser.parse_args(argv)
    if args.target_dir and not args.input_dir:
        parser.error('--target-dir is only valid with --input-dir')
    if args.workers < 0 or (args.max_images is not None and args.max_images < 1):
        parser.error('workers must be nonnegative and max-images positive')
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    protocol = Protocol(**checkpoint['protocol'])
    spec = checkpoint['model_spec']
    device = torch.device(args.device)
    # Detect changed model/runtime code rather than silently loading into a new architecture.
    import inspect
    network = build_model(spec, device, args.backend)
    source_dir = Path(__file__).parent
    expected_sources = checkpoint['source_hashes']
    for key, path in [('model_source', Path(inspect.getfile(type(network)))),
                      ('sr_runtime.py', source_dir / 'sr_runtime.py'),
                      ('metrics.py', source_dir / 'metrics.py')]:
        if file_sha256(path) != expected_sources[key]:
            raise ValueError(f'{key} differs from the training checkpoint; use the matching source version')
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    adapter = SRAdapter(network, protocol.scale, spec['input_mode'], spec['spatial_multiple'])
    execution = checkpoint['execution']
    tile = execution['tile_size'] if args.tile_size is None else args.tile_size
    overlap = execution['overlap'] if args.overlap is None else args.overlap
    if args.data_root:
        for name, expected in checkpoint['data_code'].items():
            if file_sha256(source_dir / name) != expected:
                raise ValueError(f'{name} differs from the training data pipeline')
        dataset = DIV2KDataset(args.data_root, 'valid', protocol, args.allow_partial)
        data_manifest = dataset.manifest()
        if data_manifest != checkpoint['datasets']['valid']:
            raise ValueError('Validation files differ from checkpoint manifest')
        mode = 'DIV2K synthetic held-out validation (not the real THz test set)'
    else:
        dataset = PairedImageDataset(args.input_dir, protocol.scale, args.target_dir)
        data_manifest = {'inputs': [{'name': path.name, 'sha256': file_sha256(path)}
                                    for path in dataset.inputs.values()]}
        if dataset.targets:
            data_manifest['targets'] = [{'name': path.name, 'sha256': file_sha256(path)}
                                        for path in dataset.targets.values()]
        mode = 'Real/native LR with HR references' if args.target_dir else 'Real/native LR inference only; no PSNR/SSIM'
    means, rows = evaluate(adapter, dataset, device, tile, overlap,
                           protocol.metric_border, args.workers, args.max_images,
                           args.output / 'images')
    report = {'mode': mode, 'protocol': protocol.to_dict(),
              'training_data_fingerprint': checkpoint['data_fingerprint'],
              'evaluation_data': data_manifest, 'model_spec': spec,
              'tile_size': tile, 'overlap': overlap,
              'tiling_matches_training_validation': tile == execution['tile_size'] and overlap == execution['overlap'],
              'image_count': len(rows), 'max_images': args.max_images,
              'metrics_on': 'float predictions before PNG quantization, Y channel',
              'mean': json_scores(means),
              'images': [dict(name=row['name'], **json_scores(
                  {key: value for key, value in row.items() if key != 'name'})) for row in rows]}
    write_json(args.output / 'metrics.json', report)
    print(mode)
    print(json_scores(means))
    print(f'Results: {args.output / "metrics.json"}')


if __name__ == '__main__':
    main()
