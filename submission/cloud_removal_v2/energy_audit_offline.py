"""Offline (dataset-free) energy re-audit for the ORBITALIF paper numbers.

Round-2 audit deliverable.  Everything here is computable WITHOUT the
CUHK-CR dataset or a GPU:

  * Parameter counts (exact).
  * Per-layer MAC counts at the paper's inference shape (1 x 3 x 64 x 64,
    T = 4) — MACs depend only on tensor shapes, so a CPU forward pass on
    random input measures them exactly.  Uses the round-2-patched
    EnergyMeter, which now also hooks nn.Conv3d (the original tool
    silently skipped Conv3d, biasing both ANN and SNN totals low).
  * Energy figures derived from those MACs under the paper's constants
    (4.6 pJ/MAC [Horowitz 2014, 45 nm]; 77 fJ/SOP neuromorphic).

The ONE quantity that genuinely needs the dataset is the per-layer input
non-zero rate r_l.  We do NOT fabricate it: the SNN bounds below re-use
the MAC-weighted average rate r = 0.826 that the paper reports as
measured on 32 CUHK-CR1 test patches (and which is consistent with its
published 0.287 mJ / 3.7 G SOP figures).  Rates measured on random input
are printed for reference only and are NOT used in any energy figure.

Run:  python -m cloud_removal_v2.energy_audit_offline [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))

import torch

from cloud_removal_v1.models import build_vlifnet, build_plain_unet
from cloud_removal_v2.energy_estimation import EnergyMeter, _reset_snn_safely

ANN_PJ_PER_MAC = 4.6      # Horowitz ISSCC 2014, 45 nm fixed-point MAC
AC_PJ_PER_SOP = 0.077     # 77 fJ/SOP, neuromorphic constant used by the paper
R_BAR_CR1 = 0.826         # MAC-weighted mean input non-zero rate, as
                          # measured by the paper on CUHK-CR1 test patches
                          # (NOT re-measured here; requires the dataset)


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _measure_macs(model, n_passes: int = 1):
    """Forward random 1x3x64x64 input(s) through `model` under a patched
    EnergyMeter and return the per-image MAC breakdown (shape-exact)."""
    meter = EnergyMeter(model)
    model.eval()
    for _ in range(n_passes):
        x = torch.rand(1, 3, 64, 64)
        _reset_snn_safely(model)
        model(x)
        _reset_snn_safely(model)
    meter.remove()
    per_kind = {}
    for s in meter.mac_stats.values():
        per_kind.setdefault(s.kind, 0.0)
        per_kind[s.kind] += s.mac_total / n_passes
    rows = meter.per_layer_table(n_passes)
    return per_kind, rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=str, default=None, help="optional JSON output path")
    args = p.parse_args(argv)

    report = {"note": "MACs are shape-exact (measured via hooks on random "
                      "input); SNN activity uses the paper's measured "
                      f"r_bar={R_BAR_CR1} (dataset required to re-measure)."}

    # ---- ORBITALIF / ORBITANN (identical layer set => identical MACs) ----
    vlif = build_vlifnet(dim=24, en_num_blocks=(2, 2, 4, 4),
                         de_num_blocks=(2, 2, 2, 2), T=4,
                         backend="torch", bn_variant="tdbn", backbone="snn")
    report["params_vlifnet"] = _count_params(vlif)
    kinds, rows = _measure_macs(vlif)
    conv2d = kinds.get("Conv2d", 0.0)
    linear = kinds.get("Linear", 0.0)
    conv3d = kinds.get("Conv3d", 0.0)
    eventful = conv2d + linear            # ops that CAN be event-driven
    total = eventful + conv3d
    report["macs_vlifnet"] = {
        "conv2d": conv2d, "linear": linear, "conv3d": conv3d,
        "eventful_conv2d_linear": eventful, "total": total,
    }

    # ---- PlainUNet (ORBITUNET baseline) ----
    plain = build_plain_unet(dim=24, en_blocks=(2, 2, 4), de_blocks=(2, 2, 2))
    report["params_plain_unet"] = _count_params(plain)
    kinds_p, _ = _measure_macs(plain)
    plain_total = sum(kinds_p.values())
    report["macs_plain_unet"] = {**kinds_p, "total": plain_total}

    # ---- Energy figures (mJ) ----------------------------------------------
    mj = 1e-9  # pJ -> mJ
    e_ann = total * ANN_PJ_PER_MAC * mj                       # ORBITANN, T=4
    e_plain = plain_total * ANN_PJ_PER_MAC * mj               # ORBITUNET
    sops = eventful * R_BAR_CR1                               # event-driven SOPs
    e_snn_legacy = sops * AC_PJ_PER_SOP * mj                  # paper semantics
    e_snn_corrected = (sops * AC_PJ_PER_SOP + conv3d * ANN_PJ_PER_MAC) * mj
    e_snn_upper = (sops * ANN_PJ_PER_MAC + conv3d * ANN_PJ_PER_MAC) * mj
    report["energy_mj"] = {
        "ann_orbitann_T4": e_ann,
        "plain_unet": e_plain,
        "snn_lower_paper_semantics": e_snn_legacy,
        "snn_lower_corrected_conv3d_dense": e_snn_corrected,
        "snn_upper_conservative": e_snn_upper,
    }
    report["ratios"] = {
        "ann_over_snn_paper_semantics": e_ann / e_snn_legacy,
        "ann_over_snn_corrected": e_ann / e_snn_corrected,
        "ann_over_snn_upper": e_ann / e_snn_upper,
        "plain_over_snn_paper_semantics": e_plain / e_snn_legacy,
        "plain_over_snn_corrected": e_plain / e_snn_corrected,
        "ann_T1_equivalent_over_snn_corrected":
            (e_ann / 4.0) / e_snn_corrected,   # T=1 ANN would run 1/4 the MACs
    }

    # ---- Pretty print ------------------------------------------------------
    print(f"params: VLIFNet={report['params_vlifnet']:,}  "
          f"PlainUNet={report['params_plain_unet']:,}")
    print(f"MACs (per 64x64 image): conv2d={conv2d/1e9:.3f}G  linear={linear/1e6:.3f}M  "
          f"conv3d={conv3d/1e6:.3f}M  total={total/1e9:.3f}G")
    print(f"PlainUNet MACs: {plain_total/1e9:.3f}G")
    print(f"SOPs @ r_bar={R_BAR_CR1}: {sops/1e9:.3f}G")
    print("--- energy (mJ) ---")
    for k, v in report["energy_mj"].items():
        print(f"  {k:38s}: {v:8.4f}")
    print("--- ratios ---")
    for k, v in report["ratios"].items():
        print(f"  {k:38s}: {v:8.2f}x")

    top3d = sorted((r for r in rows if r["kind"] == "Conv3d"),
                   key=lambda r: r["mac_per_image"], reverse=True)[:6]
    print("--- Conv3d layers previously missed by the meter ---")
    for r in top3d:
        print(f"  {r['name']:55s} {r['mac_per_image']/1e6:9.3f} MMAC")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
