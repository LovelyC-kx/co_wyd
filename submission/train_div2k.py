"""Train submission backbones with exactly the root DIV2K data/optimizer protocol."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--backbone', choices=('snn', 'ann', 'plain'), default='snn')
    parser.add_argument('--dim', type=int, default=24)
    parser.add_argument('--use-refinement', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    reserved = ('--model-factory', '--model-kwargs', '--model-input', '--spatial-multiple')
    if any(token.split('=')[0] in reserved for token in remaining):
        raise ValueError('This entry point sets its model interface; use root train.py for custom factories')
    from train import main as shared_train
    if '--help' in remaining or '-h' in remaining:
        parser.print_help()
    if not any(token.split('=')[0] == '--output' for token in remaining):
        def option(name):
            for index, token in enumerate(remaining):
                if token.startswith(name + '='):
                    return token.split('=', 1)[1]
                if token == name and index + 1 < len(remaining):
                    return remaining[index + 1]
            return None
        from thz_protocol import Protocol
        configuration = option('--config') or ROOT / 'configs/thznet_div2k.json'
        scale = option('--scale') or Protocol.load(configuration).scale
        remaining += ['--output', f'runs/submission_{args.backbone}_x{scale}']
    shared_train(remaining + [
        '--model-factory', 'submission.models_div2k:build_model',
        '--model-kwargs', json.dumps({'backbone': args.backbone, 'dim': args.dim,
                                     'use_refinement': args.use_refinement}),
        '--model-input', 'bicubic', '--spatial-multiple', '8'])


if __name__ == '__main__':
    main()
