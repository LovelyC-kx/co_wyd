"""
Qualitative cloud-removal grid.

Builds a matplotlib grid for a fixed set of test images showing:
    (cloudy input) | (6 model restores) | (clear ground truth)

The 6 columns are the 6 combinations of bn_mode × aggregation scheme.
Each cell is annotated with per-image PSNR (dB).

Requires the final checkpoints saved by run_smoke.py:
    Outputs_v2/ckpts/<run_name>_<bn_mode>_<scheme>_plane<p>.pt

We use plane 0 for every cell: for FedAvg all planes are identical;
for FedBN the planes diverge and plane-0 is the representative we show
(noted in the paper).

Usage
-----
    python -m cloud_removal_v2.visualize --run_name v2a --n_samples 6
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.visualize import main
    if __name__ == "__main__":
        main()
    sys.exit(0)

from .config import parse_v2a_cli
from .dataset import MultiSourceCloudDataset

from cloud_removal_v1.constants import GOSSIP, RELAYSUM, ALLREDUCE, SCHEMES, SCHEME_LABEL
from cloud_removal_v1.models import build_vlifnet
from cloud_removal_v1.evaluation import _torch_psnr
from spikingjelly.activation_based import functional


_BN_MODES = ("fedavg", "fedbn")


def _load_model(args, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Build a fresh VLIFNet and load a state_dict."""
    model = build_vlifnet(
        dim=args.vlif_dim,
        en_num_blocks=tuple(args.en_blocks),
        de_num_blocks=tuple(args.de_blocks),
        T=args.T,
        use_refinement=False,
        inp_channels=3, out_channels=3,
        backend=args.vlif_backend,
        bn_variant=getattr(args, "bn_variant", "tdbn"),
        backbone=getattr(args, "backbone", "snn"),
    ).to(device)
    # B-LOAD-1: torch ≥2.6 flips `weights_only` default to True, which would
    # silently fail on our state_dict pickles (which contain SpikingJelly
    # memory scalars and are not on the new safe-pickle allowlist).  The
    # checkpoints we wrote come from our own training run, so `weights_only=
    # False` is the correct + safe default here.
    sd = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    functional.reset_net(model)
    return model


def _center_crop_np(t: torch.Tensor, p: int) -> torch.Tensor:
    _, H, W = t.shape
    top = (H - p) // 2
    left = (W - p) // 2
    return t[:, top:top + p, left:left + p]


def _to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """[C, H, W] float in [0, 1] → [H, W, C] uint8."""
    t = t.clamp(0, 1).detach().cpu().numpy()
    t = np.transpose(t, (1, 2, 0))
    return (t * 255.0 + 0.5).astype(np.uint8)


@torch.no_grad()
def _infer_one(model: torch.nn.Module, cloudy: torch.Tensor,
               device: torch.device, patch_size: int) -> torch.Tensor:
    x = cloudy.unsqueeze(0).to(device)
    x = _center_crop_np(x[0], patch_size).unsqueeze(0)
    functional.reset_net(model)
    pred = model(x)[0]
    functional.reset_net(model)
    return pred.clamp(0, 1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   type=str, default="v2a")
    p.add_argument("--output_dir", type=str, default="./Outputs_v2")
    p.add_argument("--ckpt_dir",   type=str, default="./Outputs_v2/ckpts")
    p.add_argument("--viz_out",    type=str, default=None,
                   help="Output PDF path (defaults to "
                        "Outputs_v2/v2a_<run>_qualitative.pdf).")
    p.add_argument("--n_samples",  type=int, default=6)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--patch_size", type=int, default=256,
                   help="Centre crop side used for inference AND the PDF tile "
                        "display.  Default 256 (up from training's 64) because "
                        "matplotlib upsamples 64x64 tiles very blurrily in the "
                        "output PDF.  VLIFNet is fully convolutional so larger "
                        "crops work without re-training.  Pass --patch_size 64 "
                        "to reproduce the old behavior (and to match the "
                        "eval-mode PSNR numbers reported in summary.json "
                        "exactly).  Memory cost at 256: ~4x the 64-crop budget.")
    # Data root overrides — visualize is usually run with the same
    # --data_root / --source_root_{1,2} as the training invocation.  If
    # neither is passed, falls back to V2A_DEFAULTS["sources"] (works
    # only if ./data/CUHK-CR{1,2} exists).
    p.add_argument("--data_root",     type=str, default=None,
                   help="Parent directory containing CUHK-CR1/ and CUHK-CR2/ "
                        "sub-folders.  Overrides the default source list.")
    p.add_argument("--source_root_1", type=str, default=None)
    p.add_argument("--source_root_2", type=str, default=None)
    p.add_argument("--plane_idx",     type=int, default=0,
                   help="Which plane's checkpoint to use per cell.")
    # Re-use config defaults for model build
    base = parse_v2a_cli([])
    p.add_argument("--bn_variant", type=str,
                   default=getattr(base, "bn_variant", "tdbn"),
                   choices=["tdbn", "bn2d"],
                   help="Must match the value used at training time; mismatch "
                        "causes strict state_dict load to fail.")
    p.add_argument("--backbone", type=str,
                   default=getattr(base, "backbone", "snn"),
                   choices=["snn", "ann"],
                   help="Must match the value used at training time.")
    p.set_defaults(
        vlif_dim=base.vlif_dim,
        en_blocks=base.en_blocks,
        de_blocks=base.de_blocks,
        T=base.T,
        vlif_backend=base.vlif_backend,
        sources=base.sources,
        train_split=base.train_split,
        test_split=base.test_split,
        device=base.device,
    )
    args = p.parse_args(argv)

    # Resolve source list with CLI overrides (same priority as run_smoke):
    # 1. --source_root_{1,2}
    # 2. --data_root → <data_root>/CUHK-CR{1,2}
    # 3. fall back to V2A_DEFAULTS["sources"]
    if args.source_root_1 or args.source_root_2:
        srcs = []
        if args.source_root_1:
            srcs.append({"root": args.source_root_1, "label": 0, "name": "src0"})
        if args.source_root_2:
            srcs.append({"root": args.source_root_2, "label": 1, "name": "src1"})
        args.sources = srcs
    elif args.data_root:
        srcs = []
        for i, sub in enumerate(("CUHK-CR1", "CUHK-CR2")):
            root = os.path.join(args.data_root, sub)
            if os.path.isdir(root):
                srcs.append({"root": root, "label": i, "name": sub})
        if not srcs:
            srcs = [{"root": args.data_root, "label": 0, "name": "only"}]
        args.sources = srcs

    # Build test dataset (no augment, full resolution)
    test = MultiSourceCloudDataset(
        args.sources, split=args.test_split, patch_size=None,
        with_labels=False, strict=True)
    print(test.describe())

    rng = np.random.RandomState(args.sample_seed)
    sample_ids = rng.choice(len(test), size=args.n_samples, replace=False)
    print(f"chosen test sample ids: {sample_ids.tolist()}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load 6 models
    models: Dict = {}
    ck_root = Path(args.ckpt_dir)
    for bn in _BN_MODES:
        for scheme in SCHEMES:
            tag = f"{args.run_name}_{bn}_{scheme}"
            ck = ck_root / f"{tag}_plane{args.plane_idx}.pt"
            if not ck.exists():
                # Fall back to plane0 (older runs from before the bugfix
                # may only have saved plane0 for FedAvg cells).
                fallback = ck_root / f"{tag}_plane0.pt"
                if fallback.exists() and fallback != ck:
                    print(f"NOTE: {ck.name} not found; falling back to "
                          f"{fallback.name}")
                    ck = fallback
                else:
                    print(f"WARN: missing {ck}; skipping ({bn}, {scheme})")
                    continue
            models[(bn, scheme)] = _load_model(args, ck, device)
            print(f"loaded {ck.name}")

    if not models:
        print("No models loaded — aborting.", file=sys.stderr)
        sys.exit(1)

    # Prepare figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "STIXGeneral"
    n_cols = 1 + len(models) + 1           # cloudy + models + clear
    n_rows = args.n_samples
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.0 * n_cols, 2.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    # Column headers
    col_titles = ["Cloudy"]
    model_keys = list(models.keys())  # (bn, scheme) pairs in insertion order
    for bn, scheme in model_keys:
        col_titles.append(f"{SCHEME_LABEL[scheme]}\n{bn.upper()}")
    col_titles.append("Clear GT")

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for r, si in enumerate(sample_ids):
        cloudy, clear = test[int(si)]    # full-res [3, H, W]
        # All viz at `patch_size` centre crop, matching the training-eval scale
        cloudy_c = _center_crop_np(cloudy, args.patch_size)
        clear_c  = _center_crop_np(clear,  args.patch_size)

        # Col 0: cloudy
        ax = axes[r, 0]
        ax.imshow(_to_uint8_hwc(cloudy_c))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            pass  # title already set

        # Cols 1..N-2: models
        for c_off, (bn, scheme) in enumerate(model_keys):
            pred = _infer_one(models[(bn, scheme)], cloudy, device, args.patch_size)
            psnr = _torch_psnr(pred, clear_c.to(device)).item()
            ax = axes[r, 1 + c_off]
            ax.imshow(_to_uint8_hwc(pred))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{psnr:.2f} dB", fontsize=8)

        # Last col: clear GT
        ax = axes[r, -1]
        ax.imshow(_to_uint8_hwc(clear_c))
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out = args.viz_out or os.path.join(
        args.output_dir, f"v2a_{args.run_name}_qualitative.pdf")
    fig.savefig(out, format="pdf", dpi=600)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
