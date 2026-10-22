"""
v1 cloud-removal configuration.

A standalone factory `build_v1_args(**overrides)` that returns a fully
populated `argparse.Namespace`.  Avoids the legacy `config.py` habit of
running argparse at import time.

Locked v1 hyperparameters
-------------------------
* CUHK-CR1 (RGB only, NIR is v2)
* 50/5/1 Walker Star = 5 planes × 10 satellites
* VLIFNet dim=24, en=[2,2,4,4], de=[2,2,2,2], T=4, torch backend
* AdamW(1e-3) + 3-epoch warmup + cosine → 1e-7
* Charbonnier + 0.1·(1-SSIM)
* num_epoch=10 smoke, intra_plane_iters=2, local_iters=2
* bn_local=False for v1 FedAvg baseline (v2 flips to FedBN)
* eval_mode='center_patch' at 64² (fits V100 16 GB; matches training dist)
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional


V1_DEFAULTS: Dict[str, Any] = {
    # ---- Run identity ----
    "run_name":           "v1_smoke",
    "seed":               1234,
    "device":             "cuda:0",
    "output_dir":         "./Outputs",
    "tensorboard_dir":    "./Outputs/tb",

    # ---- Dataset ----
    "data_root":          "./data/CUHK-CR1",
    "dataset_name":       "CUHK-CR1",
    "train_split":        "train",
    "test_split":         "test",
    "patch_size":         64,
    "train_batch_size":   4,
    "test_batch_size":    1,
    # With 50 satellites each owning a DataLoader, num_workers > 0 would
    # spawn 50 × num_workers worker processes per epoch for ~10-sample
    # client partitions — not worth it.  Keep at 0 for v1.
    "num_workers":        0,

    # ---- Constellation ----
    "num_planes":         5,
    "sats_per_plane":     10,
    "partition_mode":     "iid",
    "partition_seed":     0,

    # ---- Model ----
    "vlif_dim":           24,
    "en_blocks":          [2, 2, 4, 4],
    "de_blocks":          [2, 2, 2, 2],
    "T":                  4,
    "vlif_backend":       "torch",
    "use_refinement":     False,

    # ---- Optimisation ----
    "lr":                 1e-3,
    "min_lr":             1e-7,
    "wd":                 0.0,
    "warmup_epochs":      3,
    "clip_grad":          1.0,
    "use_amp":            False,

    # ---- Schedule ----
    "num_epoch":          10,
    "intra_plane_iters":  2,
    "local_iters":        2,

    # ---- Loss ----
    "ssim_weight":        0.1,
    "charbonnier_eps":    1e-3,

    # ---- Aggregation ----
    "bn_local":           False,

    # ---- Evaluation ----
    "eval_every":         1,
    "eval_mode":          "center_patch",
    "eval_patch_size":    64,
    "sliding_window":     64,
    "sliding_stride":     32,
}


def build_v1_args(**overrides: Any) -> argparse.Namespace:
    cfg = dict(V1_DEFAULTS)
    cfg.update(overrides)
    ns = argparse.Namespace(**cfg)
    _validate(ns)
    return ns


def _validate(ns: argparse.Namespace) -> None:
    assert ns.num_planes > 0 and ns.sats_per_plane > 0
    assert len(ns.en_blocks) == 4 and len(ns.de_blocks) == 4
    # Hard assertion: upstream VLIFNet hard-codes T=4 in several sub-blocks.
    assert ns.T == 4, (
        f"T must be 4 in v1 (upstream VLIFNet sub-blocks hard-code T=4). "
        f"Got T={ns.T}.")
    assert ns.train_batch_size >= 1
    assert ns.patch_size >= 16
    assert ns.eval_patch_size >= 16
    assert ns.vlif_backend in ("torch", "cupy")
    assert ns.partition_mode in ("iid", "dirichlet_cluster")
    assert ns.eval_mode in ("center_patch", "fullimage", "sliding")
    assert 0.0 < ns.lr and ns.min_lr >= 0
    assert 0 <= ns.ssim_weight <= 1
    assert ns.num_epoch >= 1


def parse_v1_cli(argv: Optional[list] = None) -> argparse.Namespace:
    """CLI wrapper over build_v1_args."""
    p = argparse.ArgumentParser(description="v1 cloud-removal smoke run")
    p.add_argument("--data_root",        type=str,   default=V1_DEFAULTS["data_root"])
    p.add_argument("--dataset_name",     type=str,   default=V1_DEFAULTS["dataset_name"])
    p.add_argument("--output_dir",       type=str,   default=V1_DEFAULTS["output_dir"])
    p.add_argument("--tensorboard_dir",  type=str,   default=V1_DEFAULTS["tensorboard_dir"])
    p.add_argument("--run_name",         type=str,   default=V1_DEFAULTS["run_name"])
    p.add_argument("--num_epoch",        type=int,   default=V1_DEFAULTS["num_epoch"])
    p.add_argument("--num_planes",       type=int,   default=V1_DEFAULTS["num_planes"])
    p.add_argument("--sats_per_plane",   type=int,   default=V1_DEFAULTS["sats_per_plane"])
    p.add_argument("--patch_size",       type=int,   default=V1_DEFAULTS["patch_size"])
    p.add_argument("--train_batch_size", type=int,   default=V1_DEFAULTS["train_batch_size"])
    p.add_argument("--num_workers",      type=int,   default=V1_DEFAULTS["num_workers"])
    p.add_argument("--vlif_dim",         type=int,   default=V1_DEFAULTS["vlif_dim"])
    p.add_argument("--vlif_backend",     type=str,   default=V1_DEFAULTS["vlif_backend"],
                   choices=["torch", "cupy"])
    p.add_argument("--lr",               type=float, default=V1_DEFAULTS["lr"])
    p.add_argument("--bn_local",         action="store_true",
                   help="Enable FedBN-style BN-local aggregation (v2 toggle).")
    p.add_argument("--eval_mode",        type=str,   default=V1_DEFAULTS["eval_mode"],
                   choices=["center_patch", "fullimage", "sliding"])
    p.add_argument("--eval_patch_size",  type=int,   default=V1_DEFAULTS["eval_patch_size"])
    p.add_argument("--sliding_window",   type=int,   default=V1_DEFAULTS["sliding_window"])
    p.add_argument("--sliding_stride",   type=int,   default=V1_DEFAULTS["sliding_stride"])
    p.add_argument("--seed",             type=int,   default=V1_DEFAULTS["seed"])
    p.add_argument("--device",           type=str,   default=V1_DEFAULTS["device"])
    return build_v1_args(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    ns = build_v1_args()
    print("v1 default config:")
    for k, v in sorted(vars(ns).items()):
        print(f"  {k:22s} = {v}")
