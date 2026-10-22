"""
Communication-efficiency plot for v2 Path-A.

X-axis: cumulative inter-plane communication bytes (GB).
Y-axis: test PSNR (dB).

Six curves, one per (BN mode, aggregation scheme) cell.  The styling
matches ``plot_results.py`` for consistency across paper figures:
colour = scheme, line style = BN mode (solid FedAvg / dashed FedBN).

The plot makes the "AllReduce is Pareto-optimal" claim directly
visible: AllReduce curves advance along the x-axis slower than
RelaySum / Gossip (lower bytes / round) yet reach equal or higher PSNR.

Reads the same ``Outputs_v2/v2a_<run>_<bn>_<scheme>.npz`` files as
``plot_results.py``; writes to ``Outputs_v2/v2a_<run>_comm_efficiency.pdf``.

Communication-accounting caveat (inherited from run_smoke.py): our
``comm_bytes`` per round counts per-plane EGRESS only (sum of
out-degree * state_bytes).  RelaySum / Gossip ≈ 8 * state_bytes on a
chain of 5 planes; AllReduce ≈ 5 * state_bytes.  A fully symmetric
upload+download AllReduce implementation would be 10x state_bytes and
cost MORE than gossip on chain-5 —— see
docs/v2_remaining_issues.md §3.1.  The paper Section 6 must disclose
this convention explicitly.

Usage
-----
    python -m cloud_removal_v2.plot_comm_efficiency --run_name v2a
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow both ``python -m cloud_removal_v2.plot_comm_efficiency`` and
# ``python cloud_removal_v2/plot_comm_efficiency.py``.
if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.plot_comm_efficiency import main
    if __name__ == "__main__":
        main()
        sys.exit(0)

from cloud_removal_v1.constants import (
    color_list, marker_list, SCHEMES, SCHEME_LABEL,
)

_BN_MODES = ("fedavg", "fedbn")
_BN_LABEL = {"fedavg": "FedAvg", "fedbn": "FedBN"}
_BN_STYLE = {"fedavg": "-", "fedbn": "--"}
_BN_MARKER_OFFSET = {"fedavg": 0, "fedbn": 3}


def _load_cell(output_dir: Path, run: str, bn: str, scheme: str
               ) -> Optional[Dict[str, np.ndarray]]:
    path = output_dir / f"v2a_{run}_{bn}_{scheme}.npz"
    if not path.exists():
        print(f"WARN: missing {path}", file=sys.stderr)
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "epochs":     np.asarray(data["epochs"]),
        "eval_psnr":  np.asarray(data["eval_psnr"], dtype=float),
        "comm_bytes": np.asarray(data["comm_bytes"], dtype=np.int64),
    }


def _cum_bytes_at_eval_rounds(s: Dict[str, np.ndarray]
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cum_bytes_GB, psnr_at_eval_rounds).

    ``comm_bytes`` is per-round (length T); ``eval_psnr`` has the same
    length but most entries are NaN (eval runs only every
    ``eval_every`` rounds).  We want, for each round where a valid
    PSNR exists, the cumulative bytes up to and including that round.
    """
    cb = s["comm_bytes"]
    psnr = s["eval_psnr"]
    assert cb.shape == psnr.shape, \
        f"comm_bytes / eval_psnr length mismatch: {cb.shape} vs {psnr.shape}"
    cum = np.cumsum(cb)
    valid = ~np.isnan(psnr)
    return cum[valid] / 1e9, psnr[valid]


def _plot(series: Dict[Tuple[str, str], Optional[Dict]],
          output_path: Path,
          target_psnr: Optional[float] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    fig = plt.figure(figsize=(10, 8))
    lines, legends = [], []
    # Track reach-times for the optional target PSNR
    reach_rows: List[Tuple[str, str, float]] = []

    for si, scheme in enumerate(SCHEMES):
        for bn in _BN_MODES:
            s = series.get((bn, scheme))
            if s is None:
                continue
            xs, ys = _cum_bytes_at_eval_rounds(s)
            if xs.size == 0:
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

            if target_psnr is not None and ys.max() >= target_psnr:
                idx = int(np.argmax(ys >= target_psnr))
                reach_rows.append(
                    (SCHEME_LABEL[scheme], _BN_LABEL[bn], xs[idx]))

    if target_psnr is not None:
        plt.axhline(target_psnr, color="grey", linestyle=":",
                    linewidth=1.2, zorder=0)
        plt.text(plt.xlim()[1] * 0.98, target_psnr,
                 f"  target = {target_psnr:.2f} dB",
                 fontsize=12, va="bottom", ha="right", color="grey")

    plt.legend(lines, legends, fontsize=16, loc="lower right", ncol=1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Cumulative Inter-Plane Communication (GB)", fontsize=22)
    plt.ylabel("Test PSNR (dB)", fontsize=22)
    plt.tight_layout()
    plt.grid(True, alpha=0.5)
    fig.savefig(output_path, format="pdf", dpi=1200)
    plt.close(fig)
    print(f"wrote {output_path}")

    if reach_rows:
        print()
        print("Bytes to reach target PSNR:")
        for scheme, bn, gb in sorted(reach_rows, key=lambda r: r[2]):
            print(f"  {scheme:30s} {bn:7s}  {gb:.3f} GB")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   type=str, default="v2a")
    p.add_argument("--output_dir", type=str, default="./Outputs_v2")
    p.add_argument("--target_psnr", type=float, default=None,
                   help="If set, draw a horizontal guide-line at this "
                        "PSNR and print per-cell 'bytes-to-reach' values "
                        "to stdout.  Typical choice: ~21.0 dB for v2-A.")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    series: Dict[Tuple[str, str], Optional[Dict]] = {}
    for bn in _BN_MODES:
        for scheme in SCHEMES:
            series[(bn, scheme)] = _load_cell(out_dir, args.run_name, bn, scheme)
    if not any(v is not None for v in series.values()):
        print("No .npz results found — run run_smoke first.", file=sys.stderr)
        sys.exit(1)

    _plot(series, out_dir / f"v2a_{args.run_name}_comm_efficiency.pdf",
          target_psnr=args.target_psnr)


if __name__ == "__main__":
    main()
