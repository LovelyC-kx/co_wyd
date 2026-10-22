"""
v1 smoke run entry point.

Reproduces the spirit of FLSNN Fig 5 (RelaySum vs Gossip vs All-Reduce)
but for cloud-removal regression on CUHK-CR1 with VLIFNet.

Usage
-----
    python -m cloud_removal_v1.run_smoke                            # defaults
    python -m cloud_removal_v1.run_smoke --data_root /abs/path/CUHK-CR1
    python -m cloud_removal_v1.run_smoke --eval_mode sliding        # slow but memory-safe
    python -m cloud_removal_v1.run_smoke --num_epoch 3 --num_planes 3 --sats_per_plane 3   # fast dryrun

Outputs
-------
    Outputs/v1_smoke_<run>_<Scheme>.npz
    Outputs/v1_smoke_<run>_summary.json
    Outputs/tb/<run>/
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Running with `python -m cloud_removal_v1.run_smoke` resolves the package
# relative imports.  Running with `python cloud_removal_v1/run_smoke.py`
# would hit the "no known parent package" error, so we redirect:
if __package__ in (None, ""):
    # Add the parent of cloud_removal_v1/ to sys.path and re-run as a module.
    _this = Path(__file__).resolve()
    _parent = _this.parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v1.run_smoke import main    # noqa: E402
    if __name__ == "__main__":
        main()
    sys.exit(0)

from .constants import GOSSIP, RELAYSUM, ALLREDUCE, SCHEMES
from .config import parse_v1_cli
from .dataset import (
    PairedCloudDataset,
    derived_train_test_split,
    build_plane_satellite_partitions,
    seed_worker,
)
from .constellation import CloudRemovalConstellation
from .evaluation import evaluate_per_plane, average_eval_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    print(f"[v1] {msg}", flush=True)


def _cupy_available() -> bool:
    try:
        import cupy   # noqa: F401
        return True
    except Exception:
        return False


def _negotiate_backend(requested: str) -> str:
    if requested == "cupy" and not _cupy_available():
        _log("WARN: cupy not importable → falling back to backend='torch'")
        return "torch"
    return requested


def _load_datasets(args) -> Tuple[object, object]:
    """Return (train_subset, test_subset).

    * Explicit-split path: two PairedCloudDataset instances sharing the
      same root but different patch_size: train crops 64², test keeps
      full-resolution (for sliding / centre-crop eval).
    * Flat-layout path: derived_train_test_split() returns two Subsets
      over TWO base datasets (one with crop, one without) so train/test
      semantics are consistent with the explicit-split case.
    """
    try:
        train = PairedCloudDataset(args.data_root, split=args.train_split,
                                   patch_size=args.patch_size)
        test  = PairedCloudDataset(args.data_root, split=args.test_split,
                                   patch_size=None)
        _log(f"Explicit split: |train|={len(train)}  |test|={len(test)}")
        return train, test
    except FileNotFoundError:
        train_sub, test_sub = derived_train_test_split(
            args.data_root, args.patch_size,
            test_ratio=0.2, seed=args.partition_seed)
        _log(f"Flat layout (derived 8:2): |train|={len(train_sub)}  |test|={len(test_sub)}")
        return train_sub, test_sub


# ---------------------------------------------------------------------------
# One scheme
# ---------------------------------------------------------------------------

def _run_scheme(scheme: str,
                args,
                test_loader: DataLoader,
                client_datasets,
                device: torch.device,
                writer,
                ) -> Dict:
    _log(f"========== scheme = {scheme} ==========")
    constellation = CloudRemovalConstellation(
        num_planes=args.num_planes,
        sats_per_plane=args.sats_per_plane,
        client_datasets=client_datasets,
        args=args,
        device=str(device),
        logger=_log,
    )

    history = {
        "epochs":         [],
        "train_loss":     [],
        "eval_psnr":      [],
        "eval_ssim":      [],
        "comm_bytes":     [],
        "per_plane_psnr": [],
        "per_plane_ssim": [],
        "wall_seconds":   [],
    }

    for ep in range(1, args.num_epoch + 1):
        t0 = time.time()
        train_loss = constellation.train_one_round(scheme)

        do_eval = (ep % args.eval_every == 0) or (ep == args.num_epoch)
        if do_eval:
            per_plane = evaluate_per_plane(
                constellation, test_loader,
                mode=args.eval_mode,
                patch_size=args.eval_patch_size,
                window=args.sliding_window,
                stride=args.sliding_stride,
                device=device,
            )
            mean = average_eval_results(per_plane)
            psnr, ssim = mean.mean_psnr, mean.mean_ssim
            pp_psnr = [r.mean_psnr for r in per_plane]
            pp_ssim = [r.mean_ssim for r in per_plane]
        else:
            psnr, ssim, pp_psnr, pp_ssim = np.nan, np.nan, [], []

        dt = time.time() - t0
        cb = int(constellation.round_bytes[-1])

        history["epochs"].append(ep)
        history["train_loss"].append(float(train_loss))
        history["eval_psnr"].append(float(psnr))
        history["eval_ssim"].append(float(ssim))
        history["comm_bytes"].append(cb)
        history["per_plane_psnr"].append(pp_psnr)
        history["per_plane_ssim"].append(pp_ssim)
        history["wall_seconds"].append(dt)

        _log(f"[{scheme}] ep {ep:02d}/{args.num_epoch}  "
             f"loss={train_loss:.4f}  "
             f"PSNR={psnr:.3f}dB  SSIM={ssim:.4f}  "
             f"comm={cb/1e6:.1f}MB  time={dt:.1f}s")

        if writer is not None:
            writer.add_scalar(f"{scheme}/train_loss", train_loss, ep)
            if not np.isnan(psnr):
                writer.add_scalar(f"{scheme}/eval_psnr", psnr, ep)
                writer.add_scalar(f"{scheme}/eval_ssim", ssim, ep)
            writer.add_scalar(f"{scheme}/comm_bytes", cb, ep)

    del constellation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_v1_cli(argv)
    _set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")
    if device.type == "cuda":
        _log(f"CUDA device name: {torch.cuda.get_device_name(device)}")

    args.vlif_backend = _negotiate_backend(args.vlif_backend)
    _log(f"VLIFNet backend: {args.vlif_backend}")

    # ---- Data ----------------------------------------------------------
    train_sub, test_sub = _load_datasets(args)

    total_clients = args.num_planes * args.sats_per_plane
    assert len(train_sub) >= total_clients, (
        f"|train|={len(train_sub)} too small for "
        f"{args.num_planes}×{args.sats_per_plane}={total_clients} clients.")

    client_datasets = build_plane_satellite_partitions(
        train_sub,
        num_planes=args.num_planes,
        sats_per_plane=args.sats_per_plane,
        mode=args.partition_mode,
        seed=args.partition_seed,
    )
    per_client_sizes = [[len(ds) for ds in row] for row in client_datasets]
    _log(f"per-client sizes: {per_client_sizes}")

    test_loader_kwargs = dict(
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        test_loader_kwargs["worker_init_fn"] = seed_worker
    test_loader = DataLoader(test_sub, **test_loader_kwargs)

    # ---- Tensorboard ---------------------------------------------------
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.run_name))
    except Exception as e:
        _log(f"tensorboard disabled ({e})")

    # ---- Run schemes ---------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Dict] = {}
    for scheme in SCHEMES:
        out_npz = os.path.join(args.output_dir,
                               f"v1_smoke_{args.run_name}_{scheme}.npz")
        history = _run_scheme(scheme, args, test_loader, client_datasets,
                              device, writer)
        np.savez(out_npz,
                 epochs=np.array(history["epochs"]),
                 train_loss=np.array(history["train_loss"]),
                 eval_psnr=np.array(history["eval_psnr"]),
                 eval_ssim=np.array(history["eval_ssim"]),
                 comm_bytes=np.array(history["comm_bytes"]),
                 wall_seconds=np.array(history["wall_seconds"]),
                 per_plane_psnr=np.array(history["per_plane_psnr"], dtype=object),
                 per_plane_ssim=np.array(history["per_plane_ssim"], dtype=object))
        _log(f"wrote {out_npz}")
        all_results[scheme] = history

    if writer is not None:
        writer.close()

    # ---- Summary JSON -------------------------------------------------
    summary = {
        "config": {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, list, tuple))},
        "final": {
            s: {
                "PSNR": float(all_results[s]["eval_psnr"][-1]),
                "SSIM": float(all_results[s]["eval_ssim"][-1]),
                "train_loss": float(all_results[s]["train_loss"][-1]),
                "total_comm_bytes":  int(sum(all_results[s]["comm_bytes"])),
                "total_wall_seconds": float(sum(all_results[s]["wall_seconds"])),
            }
            for s in all_results
        },
    }
    summary_path = os.path.join(args.output_dir,
                                f"v1_smoke_{args.run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"wrote {summary_path}")
    _log("done.")


if __name__ == "__main__":
    main()
