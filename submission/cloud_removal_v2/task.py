"""
v2 task shim.

The v1 `CloudRemovalSNNTask` already provides:
  * VLIFNet construction with configurable dim/T/backend,
  * Charbonnier + λ·(1-SSIM) loss,
  * reset_net before + after every forward,
  * persistent AdamW with warmup + cosine LR,
  * get_weights / apply_global_weights symmetry.

All of these are identical between v1 and v2-A — the only thing v2 does
differently at the TASK level is receive data from an
`AugmentedPairedCloudDataset`, which is orthogonal to the task internals
(the task just consumes whatever DataLoader yields).

We therefore re-export the v1 classes from here so v2 code can import
them without reaching across packages, but we do NOT introduce a new
task class.
"""

from cloud_removal_v1.task import (
    CloudRemovalSNNTask,
    CharbonnierLoss,
    SSIMLoss,
    CloudLoss,
)

__all__ = [
    "CloudRemovalSNNTask",
    "CharbonnierLoss",
    "SSIMLoss",
    "CloudLoss",
]
