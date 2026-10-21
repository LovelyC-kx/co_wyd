"""Optional evaluation of saved 8-bit PNGs using the shared Y-channel metrics.

For benchmark results use test.py, which scores unquantized float predictions.
This utility scores saved/quantized files and does not download LPIPS weights.
"""
import argparse
from pathlib import Path
from dataset_load import PairedImageDataset
from metrics import y_metrics, json_scores
from thz_protocol import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated_images_path', '--generated-dir', dest='generated', required=True)
    parser.add_argument('--target_path', '--target-dir', dest='target', required=True)
    parser.add_argument('--crop_border', '--crop-border', dest='border', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('results/saved_png_metrics.json'))
    args = parser.parse_args(argv)
    dataset = PairedImageDataset(args.generated, scale=1, target_dir=args.target)
    rows = []
    for sample in dataset:
        rows.append(dict(name=sample['name'], **y_metrics(sample['lr'], sample['hr'], args.border)))
    means = {key: sum(row[key] for row in rows) / len(rows) for key in ('psnr_y', 'ssim_y')}
    write_json(args.output, {'metrics_on': 'saved 8-bit images, Y channel',
                            'border': args.border, 'mean': json_scores(means),
                            'images': [dict(name=row['name'], **json_scores(
                                {k: v for k, v in row.items() if k != 'name'})) for row in rows]})
    print(json_scores(means))


if __name__ == '__main__':
    main()
