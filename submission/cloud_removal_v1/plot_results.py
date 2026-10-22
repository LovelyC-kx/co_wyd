"""
Plot v1 smoke-run results.

Reads Outputs/v1_smoke_<run>_*.npz produced by run_smoke.py and writes
three PDF comparison figures:

    Outputs/v1_<run>_train_loss.pdf
    Outputs/v1_<run>_test_psnr.pdf
    Outputs/v1_<run>_test_ssim.pdf

Matplotlib style matches the original FLSNN paper's Fig 5.

Usage
-----
    python -m cloud_removal_v1.plot_results --run_name v1_smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

if __package__ in (None, ""):
    _this = Path(__file__).resolve()
    _parent = _this.parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v1.plot_results import main
    if __name__ == "__main__":
        main()
    sys.exit(0)

from .constants import color_list, marker_list, SCHEMES, SCHEME_LABEL


def _load_scheme(output_dir: Path, run_name: str, scheme: str) -> Optional[Dict]:
    path = output_dir / f"v1_smoke_{run_name}_{scheme}.npz"
    if not path.exists():
        print(f"WARN: missing {path}", file=sys.stderr)
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "epochs":      data["epochs"],
        "train_loss":  data["train_loss"],
        "eval_psnr":   data["eval_psnr"],
        "eval_ssim":   data["eval_ssim"],
        "comm_bytes":  data["comm_bytes"],
    }


def _plot_metric(series: Dict[str, Dict], metric: str, ylabel: str,
                 output_path: Path, percent: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    fig = plt.figure(figsize=(10, 8))
    lines, legends = [], []
    for idx, scheme in enumerate(SCHEMES):
        s = series.get(scheme)
        if s is None:
            continue
        xs, ys = s["epochs"], s[metric]
        if np.issubdtype(ys.dtype, np.floating):
            mask = ~np.isnan(ys)
            xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            continue
        line, = plt.plot(xs, ys,
                         color=color_list[idx % len(color_list)],
                         linestyle="-",
                         marker=marker_list[idx % len(marker_list)],
                         markerfacecolor="none", ms=7,
                         markeredgewidth=2.5, linewidth=2.5,
                         markevery=max(1, len(xs) // 10))
        lines.append(line)
        legends.append(SCHEME_LABEL[scheme])

    plt.legend(lines, legends, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Global Epoch", fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    if percent:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    fig.savefig(output_path, format="pdf", dpi=1200)
    plt.close(fig)
    print(f"wrote {output_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   type=str, default="v1_smoke")
    p.add_argument("--output_dir", type=str, default="./Outputs")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    series = {s: _load_scheme(out_dir, args.run_name, s) for s in SCHEMES}
    if not any(v is not None for v in series.values()):
        print("No .npz results found — run run_smoke.py first.", file=sys.stderr)
        sys.exit(1)

    _plot_metric(series, "train_loss", "Training Loss",
                 out_dir / f"v1_{args.run_name}_train_loss.pdf")
    _plot_metric(series, "eval_psnr",  "Test PSNR (dB)",
                 out_dir / f"v1_{args.run_name}_test_psnr.pdf")
    _plot_metric(series, "eval_ssim",  "Test SSIM",
                 out_dir / f"v1_{args.run_name}_test_ssim.pdf")


if __name__ == "__main__":
    main()
