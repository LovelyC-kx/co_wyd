"""
Dirichlet-over-source partition heat-map for v2 Path-A.

Reproduces the EXACT partition run_smoke.py used (same Dirichlet alpha,
same partition_seed, same min_per_client) and visualises the per-client
composition of CR1 (thin) / CR2 (thick) samples as a heat-map.

The goal is to make the "severe non-IID under α=0.1" claim
reviewer-proof: the heat-map at α=0.1 must show most clients as almost
pure-CR1 or almost pure-CR2, with only a few mixed clients, per the
Hsu et al. 2019 Dirichlet-partition behaviour.

Determinism: this script requires that
 (a) the CUHK-CR1 and CUHK-CR2 datasets are accessible at the path
     supplied via ``--data_root`` (or via the config default), and
 (b) the partition seed + alpha + num_planes + sats_per_plane +
     min_samples_per_client match the settings you want to visualise
     — the defaults match V2A_DEFAULTS.

Outputs
-------
``Outputs_v2/v2a_<run>_partition.pdf``
    A two-column 50-row heat-map whose rows are satellites (ordered
    plane-major), whose columns are {CR1 thin, CR2 thick}, and whose
    cells show the per-client sample count in that (satellite, source)
    bucket.  Colour intensity encodes count.

``Outputs_v2/v2a_<run>_partition_summary.txt``
    Text summary of per-client sizes and per-plane totals.

Usage
-----
    python -m cloud_removal_v2.plot_partition_heatmap \\
        --data_root /root/autodl-tmp/C-CUHK \\
        --run_name v2a
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.plot_partition_heatmap import main
    if __name__ == "__main__":
        main()
        sys.exit(0)

from .config import V2A_DEFAULTS
from .dataset import (
    MultiSourceCloudDataset,
    dirichlet_source_partition,
)


def _resolve_sources(args) -> list:
    """Same source-resolution logic as run_smoke / visualize."""
    if args.source_root_1 or args.source_root_2:
        srcs = []
        if args.source_root_1:
            srcs.append({"root": args.source_root_1, "label": 0, "name": "src0"})
        if args.source_root_2:
            srcs.append({"root": args.source_root_2, "label": 1, "name": "src1"})
        return srcs
    if args.data_root:
        srcs = []
        for i, sub in enumerate(("CUHK-CR1", "CUHK-CR2")):
            root = os.path.join(args.data_root, sub)
            if os.path.isdir(root):
                srcs.append({"root": root, "label": i, "name": sub})
        if srcs:
            return srcs
        return [{"root": args.data_root, "label": 0, "name": "only"}]
    return V2A_DEFAULTS["sources"]


def _plane_sat_labels(num_planes: int, sats_per_plane: int) -> List[str]:
    out = []
    for p in range(num_planes):
        for s in range(sats_per_plane):
            out.append(f"P{p}S{s}")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   type=str, default="v2a")
    p.add_argument("--output_dir", type=str, default="./Outputs_v2")
    p.add_argument("--data_root",     type=str, default=None)
    p.add_argument("--source_root_1", type=str, default=None)
    p.add_argument("--source_root_2", type=str, default=None)
    p.add_argument("--num_planes",    type=int,
                   default=V2A_DEFAULTS["num_planes"])
    p.add_argument("--sats_per_plane", type=int,
                   default=V2A_DEFAULTS["sats_per_plane"])
    p.add_argument("--partition_alpha", type=float,
                   default=V2A_DEFAULTS["partition_alpha"])
    p.add_argument("--partition_seed",  type=int,
                   default=V2A_DEFAULTS["partition_seed"])
    p.add_argument("--min_per_client",  type=int,
                   default=V2A_DEFAULTS["min_samples_per_client"])
    p.add_argument("--patch_size",  type=int,
                   default=V2A_DEFAULTS["patch_size"],
                   help="Passed through to MultiSourceCloudDataset; "
                        "irrelevant for the partition shape, but required "
                        "for dataset construction.")
    p.add_argument("--train_split", type=str,
                   default=V2A_DEFAULTS["train_split"])
    args = p.parse_args(argv)

    sources = _resolve_sources(args)
    print(f"partition sources: {[s['name'] for s in sources]}")

    # Re-build the training dataset to obtain per-sample source labels.
    # Use with_labels=False (we fetch labels via source_labels() method;
    # strict=True so that a mis-configured --data_root fails loudly.
    train = MultiSourceCloudDataset(
        sources,
        split=args.train_split,
        patch_size=args.patch_size,
        with_labels=False,
        strict=True,
    )
    print(train.describe())

    labels = train.source_labels()
    total_clients = args.num_planes * args.sats_per_plane
    flat_indices = dirichlet_source_partition(
        source_labels=labels,
        num_clients=total_clients,
        alpha=args.partition_alpha,
        seed=args.partition_seed,
        min_per_client=args.min_per_client,
    )

    # Build the (clients x sources) count matrix.
    n_sources = int(labels.max()) + 1
    counts = np.zeros((total_clients, n_sources), dtype=np.int64)
    for c, idx_list in enumerate(flat_indices):
        if not idx_list:
            continue
        arr = labels[np.asarray(idx_list, dtype=np.int64)]
        for s in range(n_sources):
            counts[c, s] = int((arr == s).sum())
    per_client_sizes = counts.sum(axis=1)

    # ---- Plot ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "STIXGeneral"

    client_labels = _plane_sat_labels(args.num_planes, args.sats_per_plane)
    source_names = [s.get("name", f"src{s['label']}") for s in sources]

    fig, ax = plt.subplots(figsize=(3.0 + 0.25 * n_sources,
                                    2.0 + 0.18 * total_clients))
    im = ax.imshow(counts, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Sample count", fontsize=12)

    # Annotate each cell with its count.
    vmax = counts.max()
    for c in range(total_clients):
        for s in range(n_sources):
            val = counts[c, s]
            ax.text(s, c, str(val), ha="center", va="center",
                    fontsize=7,
                    color=("white" if val > vmax * 0.6 else "black"))

    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(source_names, fontsize=10)
    ax.set_yticks(range(total_clients))
    ax.set_yticklabels(client_labels, fontsize=6)

    # Separators between planes
    for p_line in range(1, args.num_planes):
        ax.axhline(p_line * args.sats_per_plane - 0.5,
                   color="black", linewidth=0.6)

    ax.set_xlabel("Source (cloud type)", fontsize=12)
    ax.set_ylabel("Client (plane · satellite)", fontsize=12)
    ax.set_title(
        f"Dirichlet(α={args.partition_alpha}) over source — "
        f"seed={args.partition_seed}, "
        f"N={args.num_planes}×{args.sats_per_plane}, "
        f"min_per_client={args.min_per_client}",
        fontsize=11,
    )
    plt.tight_layout()

    out_pdf = Path(args.output_dir) / f"v2a_{args.run_name}_partition.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=600)
    plt.close(fig)
    print(f"wrote {out_pdf}")

    # ---- Text summary --------------------------------------------------
    out_txt = Path(args.output_dir) / f"v2a_{args.run_name}_partition_summary.txt"
    with open(out_txt, "w") as f:
        f.write(
            f"Dirichlet-over-source partition summary\n"
            f"  alpha            = {args.partition_alpha}\n"
            f"  seed             = {args.partition_seed}\n"
            f"  num_planes       = {args.num_planes}\n"
            f"  sats_per_plane   = {args.sats_per_plane}\n"
            f"  min_per_client   = {args.min_per_client}\n"
            f"  total_clients    = {total_clients}\n"
            f"  total_samples    = {int(per_client_sizes.sum())}\n"
            f"  per-source totals = {dict(zip(source_names, counts.sum(axis=0).tolist()))}\n"
            f"\n"
            f"Per-client size statistics:\n"
            f"  min    = {int(per_client_sizes.min())}\n"
            f"  max    = {int(per_client_sizes.max())}\n"
            f"  median = {int(np.median(per_client_sizes))}\n"
            f"  mean   = {per_client_sizes.mean():.2f}\n"
            f"  std    = {per_client_sizes.std():.2f}\n"
            f"\n"
            f"Per-client table (thin=CR1, thick=CR2, size, "
            f"dominant source fraction):\n"
        )
        for c in range(total_clients):
            sz = int(per_client_sizes[c])
            if sz == 0:
                dom = 0.0
            else:
                dom = float(counts[c].max()) / sz
            f.write(
                f"  {client_labels[c]:>7s}  "
                f"{source_names[0]}={int(counts[c,0]):4d}  "
                f"{source_names[1]}={int(counts[c,1]):4d}  "
                f"size={sz:4d}  dominant_frac={dom:.2f}\n"
            )
    print(f"wrote {out_txt}")

    # ---- Terse stdout summary -----------------------------------------
    frac_single_source = float(
        (counts.min(axis=1) == 0).sum()) / total_clients
    print(f"\nsummary: {int(per_client_sizes.sum())} samples, "
          f"{total_clients} clients, "
          f"{frac_single_source*100:.0f}% are pure-single-source "
          f"(α={args.partition_alpha})")


if __name__ == "__main__":
    main()
