"""Evaluate the submission checkpoint with the same root DIV2K/Y-metric pipeline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from test import main as shared_test
    shared_test(argv)


if __name__ == '__main__':
    main()
