"""
Per-plane PSNR / SSIM spread for v2 Path-A.

Each cell (bn_mode × scheme) has 5 planes; at every eval round we store
the per-plane PSNR & SSIM in the npz (``per_plane_psnr``,
``per_plane_ssim``).  This script extracts the final-epoch values and
plots:

  (top)    per-cell box-plot of per-plane PSNR at final epoch
  (bottom) per-cell box-plot of per-plane SSIM at final epoch

Why it matters for the paper:

* Under FedBN, planes maintain local BN and should therefore diverge;
  the spread directly quantifies 'how much divergence does FedBN
  introduce'.
* Under FedAvg with Gossip or RelaySum, planes also diverge (each
  plane averages only with chain neighbours — see constellation.py
  intra-plane step).  The spread is typically smaller than FedBN's
  but non-zero.
* Under FedAvg with AllReduce (bn_local=False), all planes converge
  to the same weights on each round — spread should be ~0.

If the observed FedBN spread is noticeably wider than FedAvg spread,
it's direct evidence that FedBN IS doing what it claims (per-plane
specialisation), independent of whether it improves mean PSNR.

Also saves a text table with per-cell (min, max, median, mean, std)
PSNR and SSIM across planes for the paper Section 6.

Usage
-----
    python -m cloud_removal_v2.plot_per_plane --run_name v2a
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.plot_per_plane import main
    if __name__ == "__main__":
        main()
        sys.exit(0)

from cloud_removal_v1.constants import (
    color_list, SCHEMES, SCHEME_LABEL,
)

_BN_MODES = ("fedavg", "fedbn")
_BN_LABEL = {"fedavg": "FedAvg", "fedbn": "FedBN"}


def _load_cell_final(output_dir: Path, run: str, bn: str, scheme: str
                     ) -> Optional[Tuple[List[float], List[float], int]]:
    """Return (per_plane_psnr_final, per_plane_ssim_final,
    final_epoch) for the LAST eval round of the given cell.  Returns
    None if the npz is missing or has no valid eval row."""
    path = output_dir / f"v2a_{run}_{bn}_{scheme}.npz"
    if not path.exists():
        print(f"WARN: missing {path}", file=sys.stderr)
        return None
    data = np.load(path, allow_pickle=True)
    epochs = np.asarray(data["epochs"])
    pp_psnr_all = data["per_plane_psnr"]  # object array of 1-D lists
    pp_ssim_all = data["per_plane_ssim"]
    # Find the last row that is not empty (eval only runs every
    # ``eval_every`` rounds, so most rows are ``[]``).
    for i in range(len(pp_psnr_all) - 1, -1, -1):
        row_p = pp_psnr_all[i]
        row_s = pp_ssim_all[i]
        if row_p is not None and len(row_p) > 0:
            return (list(map(float, row_p)),
                    list(map(float, row_s)),
                    int(epochs[i]))
    print(f"WARN: {path.name} has no valid per-plane eval row",
          file=sys.stderr)
    return None


def _cell_label(bn: str, scheme: str) -> str:
    return f"{_BN_LABEL[bn]}\n{SCHEME_LABEL[scheme]}"


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   type=str, default="v2a")
    p.add_argument("--output_dir", type=str, default="./Outputs_v2")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    cells = []                       # list of (bn, scheme, psnr_list, ssim_list, ep)
    for si, scheme in enumerate(SCHEMES):
        for bn in _BN_MODES:
            loaded = _load_cell_final(out_dir, args.run_name, bn, scheme)
            if loaded is None:
                continue
            pp, ss, ep = loaded
            cells.append((bn, scheme, pp, ss, ep))

    if not cells:
        print("No cells loaded — run run_smoke first.", file=sys.stderr)
        sys.exit(1)

    # ---- Plot ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "STIXGeneral"

    labels = [_cell_label(bn, sc) for (bn, sc, _, _, _) in cells]
    psnr_cols = [c[2] for c in cells]
    ssim_cols = [c[3] for c in cells]
    final_ep = cells[0][4]   # should be identical for all cells after a full run

    # Colour per scheme (match plot_results.py palette).
    box_colors = []
    for (bn, scheme, *_rest) in cells:
        si = SCHEMES.index(scheme)
        box_colors.append(color_list[si % len(color_list)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9.0, 1.3 * len(cells)),
                                                  9.0),
                                   sharex=True)

    # --- PSNR panel
    bp1 = ax1.boxplot(psnr_cols, labels=labels, patch_artist=True,
                      widths=0.55, showmeans=True,
                      meanprops=dict(marker="D", markerfacecolor="black",
                                     markersize=5))
    for patch, color in zip(bp1["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    # Scatter individual plane values for transparency.
    for i, col in enumerate(psnr_cols):
        xs = np.full(len(col), i + 1) + np.linspace(-0.15, 0.15, len(col))
        ax1.scatter(xs, col, color="black", s=18, zorder=3,
                    edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Per-plane PSNR (dB)", fontsize=16)
    ax1.set_title(f"Final-epoch (ep={final_ep}) per-plane spread "
                  f"across 6 (BN × scheme) cells",
                  fontsize=12)
    ax1.grid(True, alpha=0.4, axis="y")

    # --- SSIM panel
    bp2 = ax2.boxplot(ssim_cols, labels=labels, patch_artist=True,
                      widths=0.55, showmeans=True,
                      meanprops=dict(marker="D", markerfacecolor="black",
                                     markersize=5))
    for patch, color in zip(bp2["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for i, col in enumerate(ssim_cols):
        xs = np.full(len(col), i + 1) + np.linspace(-0.15, 0.15, len(col))
        ax2.scatter(xs, col, color="black", s=18, zorder=3,
                    edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Per-plane SSIM", fontsize=16)
    ax2.grid(True, alpha=0.4, axis="y")
    plt.setp(ax2.get_xticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    out_pdf = out_dir / f"v2a_{args.run_name}_per_plane.pdf"
    fig.savefig(out_pdf, format="pdf", dpi=1200)
    plt.close(fig)
    print(f"wrote {out_pdf}")

    # ---- Text table ---------------------------------------------------
    out_txt = out_dir / f"v2a_{args.run_name}_per_plane.txt"
    with open(out_txt, "w") as f:
        f.write("Per-plane final-epoch statistics (bn_mode, scheme, "
                "epoch, PSNR min/max/median/mean/std, "
                "SSIM min/max/median/mean/std):\n")
        for (bn, scheme, pp, ss, ep) in cells:
            ppn = np.asarray(pp, dtype=float)
            ssn = np.asarray(ss, dtype=float)
            f.write(
                f"  {_BN_LABEL[bn]:>6s}  {SCHEME_LABEL[scheme]:26s}  "
                f"ep={ep:3d}  "
                f"PSNR min={ppn.min():.3f} max={ppn.max():.3f} "
                f"med={np.median(ppn):.3f} "
                f"mean={ppn.mean():.3f} std={ppn.std():.4f}  "
                f"SSIM min={ssn.min():.4f} max={ssn.max():.4f} "
                f"med={np.median(ssn):.4f} "
                f"mean={ssn.mean():.4f} std={ssn.std():.5f}\n"
            )
    print(f"wrote {out_txt}")

    # Terse stdout summary: std-of-planes is the key quantity.
    print()
    print(f"{'cell':40s}  PSNR-std  SSIM-std")
    for (bn, scheme, pp, ss, ep) in cells:
        psnr_std = float(np.std(pp))
        ssim_std = float(np.std(ss))
        print(f"  {_BN_LABEL[bn]:6s}  {SCHEME_LABEL[scheme]:26s}  "
              f"{psnr_std:.4f}   {ssim_std:.5f}")


if __name__ == "__main__":
    main()
