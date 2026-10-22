"""Infer real signed THz matrices, including the key OPJU + wyd01 export pair."""
import argparse
import inspect
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from measurement_io import load_measurement, project_preview, SignalMapping
from model_provenance import verify_model_sources
from sr_runtime import SRAdapter, build_model, restore
from thz_protocol import Protocol, file_sha256, write_json


def save_matrix(folder, name, array):
    np.save(folder / (name + '.npy'), np.asarray(array))
    np.savetxt(folder / (name + '.tsv'), np.asarray(array), delimiter='\t', fmt='%.17g')


def gray_preview(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, help='Header-free numeric .xls/.tsv/.csv/.txt, or 2-D .npy')
    parser.add_argument('--opju', type=Path, help='Origin project; known key project auto-locates wyd01(1).xls')
    parser.add_argument('--origin-sheet', help='Read numeric values using Origin on Windows, e.g. [Book1]Sheet1')
    parser.add_argument('--origin-sheet-type', choices=('w', 'm'), default='w')
    parser.add_argument('--origin-matrix-index', type=int, default=0)
    parser.add_argument('--normalization', choices=('minmax', 'affine'), required=True,
                        help='Explicit input mapping; neither choice asserts physical intensity calibration')
    parser.add_argument('--low', type=float, help='Lower mapping bound, required for affine')
    parser.add_argument('--high', type=float, help='Upper mapping bound, required for affine')
    parser.add_argument('--polarity', choices=('positive', 'negative'), default='positive')
    parser.add_argument('--weights', type=Path, help='Shared protocol checkpoint from either model')
    parser.add_argument('--inspect-only', action='store_true', help='Export/inspect input without running a network')
    parser.add_argument('--output', type=Path, default=Path('results/key_measurement'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--backend', choices=('torch', 'cupy'), default='torch')
    parser.add_argument('--tile-size', type=int, help='Default: same as checkpoint validation')
    parser.add_argument('--overlap', type=int, help='Default: same as checkpoint validation')
    args = parser.parse_args(argv)
    if not args.inspect_only and not args.weights:
        parser.error('--weights is required unless --inspect-only is used')
    if args.inspect_only and args.weights:
        parser.error('--inspect-only does not use a checkpoint; omit --weights')
    if args.normalization == 'affine' and (args.low is None or args.high is None):
        parser.error('affine requires --low and --high')
    if args.normalization == 'minmax' and (args.low is not None or args.high is not None):
        parser.error('minmax derives bounds from this input; omit --low/--high')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output directory is nonempty; choose a new output to avoid mixing results')
    raw, metadata = load_measurement(args.matrix, args.opju, args.origin_sheet,
                                     args.origin_sheet_type, args.origin_matrix_index)
    bounds = (float(raw.min()), float(raw.max())) if args.normalization == 'minmax' else (args.low, args.high)
    mapping = SignalMapping(*bounds, polarity=args.polarity, method=args.normalization)
    normalized_unclipped = mapping.forward(raw)
    normalized = normalized_unclipped.clip(0, 1)
    clipped = int(np.count_nonzero(normalized_unclipped != normalized))
    report = {'input': metadata, 'mapping': mapping.to_dict(), 'input_clipped_count': clipped,
              'input_shape': list(raw.shape), 'reference_available': False,
              'metrics': None, 'metrics_reason': 'No independent HR reference was supplied',
              'mode': 'inspect only' if args.inspect_only else 'measured native LR inference',
              'synthetic_degradation_applied': False,
              'normalized_channels': 'one signal copied to three RGB channels; no extra spectral information',
              'physical_axes': 'not supplied; coordinates below are pixel-center indices',
              'code_sha256': {name: file_sha256(Path(__file__).parent / name) for name in
                              ('infer_measurement.py', 'measurement_io.py')}}
    if not args.inspect_only:
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
        protocol = Protocol(**checkpoint['protocol'])
        spec = checkpoint['model_spec']
        verify_model_sources(checkpoint)
        device = torch.device(args.device)
        network = build_model(spec, device, args.backend)
        for key, path in [('model_source', Path(inspect.getfile(type(network)))),
                          ('sr_runtime.py', Path(__file__).parent / 'sr_runtime.py')]:
            if file_sha256(path) != checkpoint['source_hashes'][key]:
                raise ValueError(f'{key} differs from the checkpoint; use its matching source')
        network.load_state_dict(checkpoint['state_dict'], strict=True)
        adapter = SRAdapter(network, protocol.scale, spec['input_mode'], spec['spatial_multiple']).eval()
        lr = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        tile = checkpoint['execution']['tile_size'] if args.tile_size is None else args.tile_size
        overlap = checkpoint['execution']['overlap'] if args.overlap is None else args.overlap
        with torch.no_grad():
            restored = restore(adapter, lr, tile, overlap).cpu().numpy()[0].transpose(1, 2, 0)
            bicubic = F.interpolate(lr, scale_factor=protocol.scale, mode='bicubic',
                                    align_corners=False, antialias=False)[0, 0].cpu().numpy()
        if not np.isfinite(restored).all():
            raise FloatingPointError('Model produced non-finite values')
        gray = restored @ np.array([.299, .587, .114], dtype=np.float64)
        # Keep float outputs, including model overshoot; clamp only preview PNGs.
        signal = mapping.inverse(gray)
        bicubic_signal = mapping.inverse(bicubic)
        report.update(checkpoint={'path': str(args.weights.resolve()), 'sha256': file_sha256(args.weights)},
                      protocol=protocol.to_dict(), model_spec=spec,
                      training_data_fingerprint=checkpoint['data_fingerprint'],
                      output_shape=list(gray.shape), tile_size=tile, overlap=overlap,
                      tiling_matches_checkpoint=(tile == checkpoint['execution']['tile_size'] and
                                                  overlap == checkpoint['execution']['overlap']),
                      prediction_outside_mapping_range=int(np.count_nonzero((gray < 0) | (gray > 1))),
                      output_units='original numerical scale via inverse input mapping; uncalibrated',
                      rgb_to_signal='0.299 R + 0.587 G + 0.114 B (full-range grayscale, not studio-range Y)')
        h, w = raw.shape
        scale = protocol.scale
        coordinates = {'units': 'input pixel index; physical pitch unknown',
                       'x_lr': np.arange(w).tolist(), 'y_lr': np.arange(h).tolist(),
                       'x_hr': ((np.arange(w * scale) + .5) / scale - .5).tolist(),
                       'y_hr': ((np.arange(h * scale) + .5) / scale - .5).tolist()}
    args.output.mkdir(parents=True, exist_ok=True)
    save_matrix(args.output, 'input_signal', raw)
    save_matrix(args.output, 'input_normalized', normalized)
    gray_preview(normalized).save(args.output / 'input_preview.png')
    if args.opju:
        preview = project_preview(args.opju)
        if preview:
            (args.output / 'origin_project_preview.png').write_bytes(preview)
    if not args.inspect_only:
        save_matrix(args.output, 'restored_normalized', gray)
        save_matrix(args.output, 'restored_signal', signal)
        save_matrix(args.output, 'bicubic_signal', bicubic_signal)
        np.save(args.output / 'restored_rgb.npy', restored)
        gray_preview(gray).save(args.output / 'restored_preview.png')
        gray_preview(bicubic).save(args.output / 'bicubic_preview.png')
        write_json(args.output / 'coordinates.json', coordinates)
    write_json(args.output / 'measurement.json', report)
    print(json.dumps({'input_shape': list(raw.shape), 'min': float(raw.min()), 'max': float(raw.max()),
                      'output_shape': report.get('output_shape'), 'mode': report['mode']}, ensure_ascii=False))
    print(f'Results: {args.output / "measurement.json"}')


if __name__ == '__main__':
    main()
