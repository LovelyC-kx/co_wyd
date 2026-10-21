import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Small CPU regression checks should not spawn many BLAS threads per convolution.
torch.set_num_threads(1)
