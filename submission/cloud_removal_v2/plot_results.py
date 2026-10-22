"""
v2 Path-A plot generator.

Reads Outputs_v2/v2a_<run>_<bn>_<scheme>.npz produced by run_smoke.py
and produces three comparison figures where each figure overlays six
curves = (RelaySum, Gossip, All-Reduce) × (FedAvg, FedBN).

  Outputs_v2/v2a_<run>_train_loss.pdf
  Outputs_v2/v2a_<run>_test_psnr.pdf
  Outputs_v2/v2a_<run>_test_ssim.pdf

Styling:
  * colour encodes aggregation scheme (matches v1 palette);
  * line style encodes BN mode: solid = FedAvg, dashed = FedBN;
  * marker shape encodes BN mode too (redundant for b/w print-out);
  * legend lists 6 curves.

Usage
-----
    python -m cloud_removal_v2.plot_results --run_name v2a
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.plot_results import main
    if __name__ == "__main__":
        main()
    sys.exit(0)

from cloud_removal_v1.constants import (
    color_list, marker_list, SCHEMES, SCHEME_LABEL,
)

_BN_MODES = ("fedavg", "fedbn")
_BN_LABEL = {"fedavg": "FedAvg", "fedbn": "FedBN"}
_BN_STYLE = {"fedavg": "-", "fedbn": "--"}
_BN_MARKER_OFFSET = {"fedavg": 0, "fedbn": 3}   # pick a different marker pool


def _load_cell(output_dir: Path, run: str, bn: str, scheme: str
               ) -> Optional[Dict]:
    path = output_dir / f"v2a_{run}_{bn}_{scheme}.npz"
    if not path.exists():
        print(f"WARN: missing {path}", file=sys.stderr)
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "epochs":     data["epochs"],
        "train_loss": data["train_loss"],
        "eval_psnr":  data["eval_psnr"],
        "eval_ssim":  data["eval_ssim"],
        "comm_bytes": data["comm_bytes"],
    }


def _plot_metric(series: Dict, metric: str, ylabel: str,
                 output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    fig = plt.figure(figsize=(10, 8))
    lines, legends = [], []
    for si, scheme in enumerate(SCHEMES):
        for bi, bn in enumerate(_BN_MODES):
            s = series.get((bn, scheme))
            if s is None:
                continue
            xs = s["epochs"]
            ys = s[metric]
            if np.issubdtype(ys.dtype, np.floating):
                mask = ~np.isnan(ys)
                xs, ys = xs[mask], ys[mask]
            if len(xs) == 0:
                continue
            color = color_list[si % len(color_list)]
            ls = _BN_STYLE[bn]
            marker = marker_list[(si + _BN_MARKER_OFFSET[bn]) % len(marker_list)]
            line, = plt.plot(
                xs, ys,
                color=color,
                linestyle=ls,
                marker=marker,
                markerfacecolor="none",
                ms=7,
                markeredgewidth=2.0,
                linewidth=2.2,
                markevery=max(1, len(xs) // 8),
            )
            lines.append(line)
            legends.append(f"{SCHEME_LABEL[scheme]} · {_BN_LABEL[bn]}")

    plt.legend(lines, legends, fontsize=18, loc="best", ncol=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Inter-Plane Communication Rounds", fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    fig.savefig(output_path, format="pdf", dpi=1200)
    plt.close(fig)
    print(f"wrote {output_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   type=str, default="v2a")
    p.add_argument("--output_dir", type=str, default="./Outputs_v2")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    series: Dict = {}
    for bn in _BN_MODES:
        for scheme in SCHEMES:
            series[(bn, scheme)] = _load_cell(out_dir, args.run_name, bn, scheme)
    if not any(v is not None for v in series.values()):
        print("No .npz results found — run run_smoke first.", file=sys.stderr)
        sys.exit(1)

    _plot_metric(series, "train_loss", "Training Loss",
                 out_dir / f"v2a_{args.run_name}_train_loss.pdf")
    _plot_metric(series, "eval_psnr",  "Test PSNR (dB)",
                 out_dir / f"v2a_{args.run_name}_test_psnr.pdf")
    _plot_metric(series, "eval_ssim",  "Test SSIM",
                 out_dir / f"v2a_{args.run_name}_test_ssim.pdf")


if __name__ == "__main__":
    main()
