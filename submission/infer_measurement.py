"""Use the shared raw-measurement inference entry point with any submission checkpoint."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from infer_measurement import main as shared_infer
    shared_infer(argv)


if __name__ == '__main__':
    main()
