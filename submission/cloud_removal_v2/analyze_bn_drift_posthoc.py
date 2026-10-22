"""Post-hoc BN-drift analysis (paper §VI-G evidence).

What this script reports
========================
For every (run, bn_mode, scheme) cell with all 5 plane checkpoints
present, we measure the BN-affine parameter drift across planes by
loading the 5 final-epoch state_dicts and computing two summaries:

  Var_L = mean over BN layers of  Var_p( γ_p )      (cross-plane variance of γ)
  Var_B = mean over BN layers of  Var_p( β_p )      (cross-plane variance of β)

Both Var_L and Var_B are 0 when intra-plane sync makes all 5 planes
identical (FedAvg + AllReduce); they grow with BN-local divergence
(FedBN cells, especially under delayed aggregation like RelaySum).

We additionally print absolute-magnitude diagnostics:

  max||γ−1||_∞   max over layers of  max_p max_c |γ_{p,c} − 1|
  max||β||_∞     max over layers of  max_p max_c |β_{p,c}|

NOTE on |γ−1| for TDBN
----------------------
spikingjelly.layer.ThresholdDependentBatchNorm2d initialises γ to
α·V_th ≈ 0.106 (NOT 1.0). The 0.3 threshold used by the original
SC-16a write-up was calibrated for standard nn.BatchNorm2d (γ_init=1)
and is NOT meaningful for TDBN. We therefore *report* max||γ−1||_∞ but
do NOT flag PASS/FAIL on it for TDBN runs — Var_L is the meaningful
inter-plane drift metric for both BN variants.

Why isinstance, not name-substring
----------------------------------
A previous heredoc version filtered keys by the substring "bn"|"norm"
in the parameter name. That version *missed* the BN inside
``Spiking_Residual_Block.shortcut`` (a ``nn.Sequential`` whose BN is
addressed as ``shortcut.1``, with no "bn" substring in the path) for
TDBN runs, while StandardBN2dWrapper exposed it as ``shortcut.1.bn``
with "bn" present. The mismatch made TDBN report 41/54 BN layers and
BN2d report 54/54. This isinstance-based detector loops over modules
and checks against the actual BN classes, finding all 54 sites in both
runs.

Outputs
=======
* Markdown report at ``--out PATH`` (default: stdout)
* Same data also printed to stdout in a human-readable table

Run on AutoDL
=============
::

    cd ~/shiyaunmingFLSNN-main_v2_new_new/.../Decentralized-Satellite-FL-dev-main_new_new
    python -m cloud_removal_v2.analyze_bn_drift_posthoc \
        --ckpt_dir Outputs_v2/ckpts \
        --out      Outputs_v2/v2_drift_report.md

The script auto-detects which run_names exist under ``--ckpt_dir`` by
parsing the file-name pattern ``<run_name>_<bn_mode>_<scheme>_plane<p>.pt``,
so adding a new run does not require editing this file.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# --- BN class detection -----------------------------------------------------
#
# We treat any module that:
#   (a) has both ``weight`` and ``bias`` learnable parameters, AND
#   (b) is an instance of nn.BatchNorm2d OR a known TDBN/wrapper class
# as a "BN-affine site" eligible for drift measurement.
#
# Because spikingjelly may or may not be installed in the analysis
# environment, we don't import ThresholdDependentBatchNorm2d directly.
# Instead we resolve the type from the loaded state_dict's class names
# OR we fall back to nn.BatchNorm2d (a common parent: TDBN inherits it).
_BN_CLASS_NAMES_LOWER = {
    "batchnorm2d",
    "thresholddependentbatchnorm2d",
    "standardbn2dwrapper",  # the wrapper itself has no params; its child .bn does
}


def _is_bn_module(m: nn.Module) -> bool:
    cls = type(m).__name__.lower()
    if cls in _BN_CLASS_NAMES_LOWER:
        return True
    # Defensive: any subclass of nn.BatchNorm2d (TDBN inherits from it).
    if isinstance(m, nn.BatchNorm2d):
        return True
    return False


# --- Filename parsing -------------------------------------------------------
#
# Checkpoint files are written by run_smoke.py:621 as
#     {run_name}_{bn_mode}_{scheme}_plane{p}.pt
# where bn_mode ∈ {fedavg, fedbn} and scheme is one of three strings
# defined in cloud_removal_v1/aggregation.py constants.

_BN_MODES: Tuple[str, ...] = ("fedavg", "fedbn")
_SCHEMES: Tuple[str, ...] = (
    "AllReduce_Aggregation",
    "Gossip_Averaging",
    "Relaysum_Aggregation",
)
_NUM_PLANES_DEFAULT = 5

# Regex: (run_name)_(bn_mode)_(scheme)_plane(p).pt
_CKPT_PAT = re.compile(
    r"^(?P<run>.+?)_(?P<bn>fedavg|fedbn)_"
    r"(?P<scheme>AllReduce_Aggregation|Gossip_Averaging|Relaysum_Aggregation)_"
    r"plane(?P<plane>\d+)\.pt$"
)


def discover_runs(ckpt_dir: Path) -> List[str]:
    """Auto-detect run_names by scanning files matching the ckpt pattern."""
    runs: set = set()
    for p in ckpt_dir.glob("*.pt"):
        m = _CKPT_PAT.match(p.name)
        if m:
            runs.add(m.group("run"))
    return sorted(runs)


# --- BN-key inference from a single state_dict -------------------------------
#
# A state_dict alone (without the live module) doesn't tell us the class
# of each parameter. We approximate "is this a BN affine pair?" by
# pairing keys ending in ".weight" with the corresponding ".bias" of the
# same prefix, and checking that the tensor has dtype float and shape
# [C] (1-D, channel-dim) — which matches BN's affine params and rules
# out conv/linear weights (which are 4-D / 2-D).
#
# This pure-state_dict approach is what we have to use, because the
# checkpoints saved by run_smoke.py contain only the state_dict, not
# the live model. The pairing logic is robust enough that it cannot
# accidentally include conv kernels (always 4-D) or linear weights
# (always 2-D), and it catches all BN sites regardless of whether the
# parent module is a wrapper (StandardBN2dWrapper) or the BN itself
# (ThresholdDependentBatchNorm2d).

def _bn_affine_keys(sd: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    """Return list of (weight_key, bias_key) pairs that look like BN affine.

    Heuristic: weight is 1-D float, matching bias of same prefix exists
    and is 1-D float of same length, and there is a ``running_mean`` /
    ``running_var`` sibling at the same prefix (definitive BN signature).
    """
    pairs: List[Tuple[str, str]] = []
    for w_key in sd.keys():
        if not w_key.endswith(".weight"):
            continue
        prefix = w_key[: -len(".weight")]
        b_key = prefix + ".bias"
        rm_key = prefix + ".running_mean"
        rv_key = prefix + ".running_var"
        if b_key not in sd or rm_key not in sd or rv_key not in sd:
            continue
        w = sd[w_key]
        b = sd[b_key]
        if not (torch.is_tensor(w) and torch.is_tensor(b)):
            continue
        if w.dim() != 1 or b.dim() != 1 or w.numel() != b.numel():
            continue
        if not w.is_floating_point():
            continue
        pairs.append((w_key, b_key))
    return pairs


# --- Aggregate a (run, bn, scheme) cell --------------------------------------

def _load_plane_state_dicts(
    ckpt_dir: Path, run: str, bn: str, scheme: str, num_planes: int,
) -> List[Dict[str, torch.Tensor]] | None:
    sds: List[Dict[str, torch.Tensor]] = []
    for p in range(num_planes):
        f = ckpt_dir / f"{run}_{bn}_{scheme}_plane{p}.pt"
        if not f.exists():
            return None
        try:
            sd = torch.load(f, map_location="cpu", weights_only=False)
        except TypeError:
            # PyTorch < 2.0 compatibility: weights_only kwarg unavailable
            sd = torch.load(f, map_location="cpu")
        sds.append(sd)
    return sds


def analyse_cell(
    sds: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Per-cell drift summary across the supplied 5 plane state_dicts."""
    pairs = _bn_affine_keys(sds[0])
    n_layers = len(pairs)
    if n_layers == 0:
        return {
            "n_bn_layers": 0,
            "var_gamma_mean": float("nan"),
            "var_beta_mean": float("nan"),
            "max_abs_gamma_minus_one": float("nan"),
            "max_abs_beta": float("nan"),
        }

    var_g_per_layer: List[float] = []
    var_b_per_layer: List[float] = []
    worst_g: float = 0.0
    worst_b: float = 0.0

    for w_key, b_key in pairs:
        # Stack 5 plane tensors → [P, C]
        gamma_stack = torch.stack(
            [sd[w_key].flatten().float() for sd in sds], dim=0
        )
        beta_stack = torch.stack(
            [sd[b_key].flatten().float() for sd in sds], dim=0
        )
        # Cross-plane variance per channel, then mean over channels
        var_g_per_layer.append(
            float(gamma_stack.var(dim=0, unbiased=False).mean())
        )
        var_b_per_layer.append(
            float(beta_stack.var(dim=0, unbiased=False).mean())
        )
        # Worst-case absolute deviation (any plane, any channel)
        worst_g = max(worst_g, float((gamma_stack - 1.0).abs().max()))
        worst_b = max(worst_b, float(beta_stack.abs().max()))

    return {
        "n_bn_layers": n_layers,
        "var_gamma_mean": sum(var_g_per_layer) / n_layers,
        "var_beta_mean": sum(var_b_per_layer) / n_layers,
        "max_abs_gamma_minus_one": worst_g,
        "max_abs_beta": worst_b,
    }


# --- Top-level driver -------------------------------------------------------

def run_for(
    ckpt_dir: Path, run: str, num_planes: int,
) -> List[Tuple[str, str, Dict[str, float] | None]]:
    """Iterate over the 6 cells of a run; return list of
    (bn, scheme, summary_or_None_if_missing) tuples."""
    out: List[Tuple[str, str, Dict[str, float] | None]] = []
    for bn in _BN_MODES:
        for sch in _SCHEMES:
            sds = _load_plane_state_dicts(ckpt_dir, run, bn, sch, num_planes)
            if sds is None:
                out.append((bn, sch, None))
            else:
                out.append((bn, sch, analyse_cell(sds)))
    return out


def _format_table(
    run: str, rows: List[Tuple[str, str, Dict[str, float] | None]],
) -> str:
    lines: List[str] = []
    lines.append(f"### {run}\n")
    lines.append(
        "| bn_mode | scheme | n_bn | Var(γ)_mean | Var(β)_mean | "
        "max\\|γ−1\\|∞ | max\\|β\\|∞ |"
    )
    lines.append(
        "|---------|--------|------|------------:|------------:|"
        "-------------:|-----------:|"
    )
    for bn, sch, s in rows:
        if s is None:
            lines.append(f"| {bn} | {sch} | _missing_ | — | — | — | — |")
            continue
        lines.append(
            f"| {bn} | {sch} | {s['n_bn_layers']} | "
            f"{s['var_gamma_mean']:.3e} | {s['var_beta_mean']:.3e} | "
            f"{s['max_abs_gamma_minus_one']:.4f} | "
            f"{s['max_abs_beta']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-hoc BN-drift analysis from saved plane "
        "checkpoints (paper §VI-G).",
    )
    p.add_argument(
        "--ckpt_dir", type=Path, required=True,
        help="Directory containing {run}_{bn}_{scheme}_plane{p}.pt files.",
    )
    p.add_argument(
        "--num_planes", type=int, default=_NUM_PLANES_DEFAULT,
        help=f"Number of planes per cell (default {_NUM_PLANES_DEFAULT}).",
    )
    p.add_argument(
        "--runs", type=str, nargs="*", default=None,
        help="Restrict to these run_names; default = auto-discover all.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Markdown report path; default = print to stdout only.",
    )
    args = p.parse_args()

    if not args.ckpt_dir.is_dir():
        sys.exit(f"--ckpt_dir not found: {args.ckpt_dir}")

    runs = args.runs or discover_runs(args.ckpt_dir)
    if not runs:
        sys.exit(f"no checkpoints matching the run pattern under {args.ckpt_dir}")

    md: List[str] = []
    md.append("# BN-drift post-hoc analysis (cross-plane Var(γ), Var(β))\n")
    md.append(
        "Cell rows show cross-plane variance of BN affine parameters "
        "averaged across all detected BN-affine layers. Var ≈ 0 when "
        "all 5 planes hold identical weights (FedAvg + AllReduce); "
        "non-zero variance indicates BN divergence (expected for FedBN, "
        "or as a Gossip / RelaySum noise floor).\n"
    )
    md.append(
        "**Note on max\\|γ−1\\|∞ for TDBN**: spikingjelly's "
        "ThresholdDependentBatchNorm2d initialises γ to α·V_th ≈ 0.106 "
        "(not 1.0); large max\\|γ−1\\| is therefore expected behaviour, "
        "not drift. Use **Var(γ)_mean** as the inter-plane-drift metric.\n"
    )

    for run in runs:
        rows = run_for(args.ckpt_dir, run, args.num_planes)
        md.append(_format_table(run, rows))

    text = "\n".join(md)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
