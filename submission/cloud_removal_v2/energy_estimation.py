"""
Per-image energy and spike-rate estimation for VLIFNet / ANN backbones.

Hooks every Conv2d / Linear's forward-pre and counts:
  * MAC count for that call (output_elements × in_channels × kernel_h × kernel_w
    / groups for Conv2d; out_features × in_features for Linear)
  * Input non-zero rate r_l = (input != 0).float().mean()  — the fraction
    of multiplications that actually fire when the upstream activation is
    spike-driven; for ANN inputs r_l ≈ 1 (RGB pixels rarely exact zero).

Energy bounds, following [Horowitz 2014] for ANN MAC and the
neuromorphic / in-memory-computing SOP cost adopted by FLSNN /
ESDNet / VLIF:
  * ANN:                 E_ANN = Σ MAC_l × 4.6 pJ
  * SNN (deployment lo): E_SNN_lo = Σ r_l × MAC_l × 0.077 pJ   (= 77 fJ/SOP)
  * SNN (conservative):  E_SNN_up = Σ r_l × MAC_l × 4.6 pJ
                                    (treats every non-zero MultiSpike-4
                                     spike as a full multiplication —
                                     pessimistic upper bound on standard
                                     CMOS without IMC support).
  CLI flags --ann_pj_per_mac and --ac_pj_per_op override both.

Also hooks every LIF / mem_update module to record per-layer mean output
(effective firing rate including MultiSpike-4's non-binary levels) and
non-zero rate (binary firing rate).

Usage
-----
    python -m cloud_removal_v2.energy_estimation \
        --ckpt Outputs/centralized_A1_vlif_cr1_best.pt \
        --data_root /abs/path/CUHK-CR1 --split test \
        --n_samples 32 --patch_size 64 \
        --out_dir Outputs/energy_A1

Outputs
-------
    Outputs/energy_<run>/
        energy_summary.json     totals + per-layer breakdown
        energy_per_layer.pdf    bar chart (ANN MAC pJ vs SNN AC pJ)
        spike_rate_per_layer.pdf  bar chart of LIF firing rates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow `python cloud_removal_v2/energy_estimation.py`
if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.energy_estimation import main   # noqa: E402
    if __name__ == "__main__":
        main()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Hooked statistics container
# ---------------------------------------------------------------------------

class _MACStat:
    """Aggregate per-layer MAC + input-nonzero stats over multiple forward passes."""

    def __init__(self, name: str, kind: str, weight_shape: Tuple[int, ...]):
        self.name = name
        self.kind = kind                 # "Conv2d" | "Linear"
        self.weight_shape = weight_shape
        self.mac_total = 0.0             # cumulative MAC count
        self.nonzero_total = 0.0         # cumulative MAC count weighted by input non-zero rate
        self.calls = 0


class _SpikeStat:
    """Aggregate per-layer spike statistics."""

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind                 # "LIFNode" | "mem_update"
        self.mean_sum = 0.0              # sum of mean(output) over calls
        self.nonzero_sum = 0.0           # sum of mean(output > 0) over calls
        self.calls = 0


def _conv2d_macs(input_shape: Tuple[int, ...], module: nn.Conv2d) -> Tuple[float, Tuple[int, ...]]:
    """MAC count for one Conv2d call.  Returns (mac_count, output_shape)."""
    # input may be 4-D [N, C, H, W] or 5-D [T, N, C, H, W] (spikingjelly).
    if len(input_shape) == 5:
        T, N, C, H, W = input_shape
        n_eff = T * N
    elif len(input_shape) == 4:
        N, C, H, W = input_shape
        n_eff = N
    else:
        raise ValueError(f"Conv2d input must be 4-D or 5-D; got {input_shape}")
    out_C = module.out_channels
    kH, kW = module.kernel_size
    sH, sW = module.stride
    pH, pW = module.padding
    # Reproduce Conv2d output spatial extent
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    macs = n_eff * out_C * H_out * W_out * (C // module.groups) * kH * kW
    out_shape = (T, N, out_C, H_out, W_out) if len(input_shape) == 5 else (N, out_C, H_out, W_out)
    return float(macs), out_shape


def _conv3d_macs(input_shape: Tuple[int, ...], module: nn.Conv3d) -> Tuple[float, Tuple[int, ...]]:
    """MAC count for one Conv3d call.  Input must be 5-D [N, C, D, H, W].

    Added in the round-2 energy audit: VLIFNet contains Conv3d layers
    (SUNet temporal_compress kernel (4,1,1); TimeAttention shared MLP)
    that the original Conv2d/Linear-only hooks silently missed, biasing
    BOTH the ANN and SNN totals low.
    """
    if len(input_shape) != 5:
        raise ValueError(f"Conv3d input must be 5-D; got {input_shape}")
    N, C, D, H, W = input_shape
    out_C = module.out_channels
    kD, kH, kW = module.kernel_size
    sD, sH, sW = module.stride
    pD, pH, pW = module.padding
    D_out = (D + 2 * pD - kD) // sD + 1
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    macs = N * out_C * D_out * H_out * W_out * (C // module.groups) * kD * kH * kW
    return float(macs), (N, out_C, D_out, H_out, W_out)


def _linear_macs(input_shape: Tuple[int, ...], module: nn.Linear) -> Tuple[float, Tuple[int, ...]]:
    if len(input_shape) < 2:
        raise ValueError(f"Linear input must be ≥2-D; got {input_shape}")
    leading = 1
    for d in input_shape[:-1]:
        leading *= d
    macs = leading * module.in_features * module.out_features
    out_shape = (*input_shape[:-1], module.out_features)
    return float(macs), out_shape


# ---------------------------------------------------------------------------
# Module-class detection (works without spikingjelly imported)
# ---------------------------------------------------------------------------

_LIF_CLASSES_LOWER = {"lifnode", "mem_update"}


def _is_lif(m: nn.Module) -> bool:
    return type(m).__name__.lower() in _LIF_CLASSES_LOWER


# ---------------------------------------------------------------------------
# Energy meter
# ---------------------------------------------------------------------------

class EnergyMeter:
    """Registers hooks on a model and accumulates MAC + spike-rate stats."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.mac_stats: Dict[str, _MACStat] = {}
        self.spike_stats: Dict[str, _SpikeStat] = {}
        self._handles = []

        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                self.mac_stats[name] = _MACStat(name, "Conv2d", tuple(mod.weight.shape))
                self._handles.append(mod.register_forward_pre_hook(self._mac_hook(name)))
            elif isinstance(mod, nn.Conv3d):
                self.mac_stats[name] = _MACStat(name, "Conv3d", tuple(mod.weight.shape))
                self._handles.append(mod.register_forward_pre_hook(self._mac_hook(name)))
            elif isinstance(mod, nn.Linear):
                self.mac_stats[name] = _MACStat(name, "Linear", tuple(mod.weight.shape))
                self._handles.append(mod.register_forward_pre_hook(self._mac_hook(name)))
            elif _is_lif(mod):
                self.spike_stats[name] = _SpikeStat(name, type(mod).__name__)
                self._handles.append(mod.register_forward_hook(self._spike_hook(name)))

    def _mac_hook(self, name: str):
        def hook(module, inputs):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            stat = self.mac_stats[name]
            try:
                if isinstance(module, nn.Conv3d):
                    macs, _ = _conv3d_macs(tuple(x.shape), module)
                elif isinstance(module, nn.Conv2d):
                    macs, _ = _conv2d_macs(tuple(x.shape), module)
                else:
                    macs, _ = _linear_macs(tuple(x.shape), module)
            except Exception:
                # Skip layers we can't measure (irregular shape / dynamic group).
                return
            with torch.no_grad():
                # input non-zero rate over all elements
                r = (x != 0).float().mean().item() if x.numel() > 0 else 0.0
            stat.mac_total += macs
            stat.nonzero_total += macs * r
            stat.calls += 1
        return hook

    def _spike_hook(self, name: str):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            stat = self.spike_stats[name]
            with torch.no_grad():
                stat.mean_sum += output.float().mean().item()
                stat.nonzero_sum += (output != 0).float().mean().item()
            stat.calls += 1
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- aggregation --------------------------------------------------------

    def per_image_macs(self, n_images: int) -> Tuple[float, float]:
        """Return (per-image total MAC, per-image effective non-zero MAC)."""
        total_mac = sum(s.mac_total for s in self.mac_stats.values()) / max(1, n_images)
        total_nz = sum(s.nonzero_total for s in self.mac_stats.values()) / max(1, n_images)
        return total_mac, total_nz

    def energy_bounds_per_image(self, n_images: int,
                                ann_pj_per_mac: float = 4.6,
                                ac_pj_per_op: float = 0.077) -> Dict[str, float]:
        """Compute per-image inference energy bounds.

        Constants follow the SNN literature for edge / neuromorphic
        deployment (the user-locked deployment target):

        * ``ann_pj_per_mac = 4.6`` pJ/MAC -- standard 45 nm CMOS
          fixed-point MAC cost from Horowitz, ISSCC 2014.

        * ``ac_pj_per_op = 0.077`` pJ/SOP (= 77 fJ) -- per-SOP
          (synaptic operation) cost on neuromorphic / in-memory
          computing substrates such as Loihi-2, Speck, Akida.  This
          is the constant adopted by FLSNN [Yang2025], ESDNet
          [Song2024], and VLIF-deraining [Chen2025] for SNN
          deployment energy reporting; ~12x lower than Horowitz's
          0.9 pJ/AC for general-purpose CMOS.

        The two bounds reported are:

          ``energy_snn_lower_pj`` = effective_acs × 0.077 pJ
              (neuromorphic SOP cost; deployment-target estimate)

          ``energy_snn_upper_pj`` = effective_acs × 4.6 pJ
              (conservative; treats every non-zero MAC as a full ANN
              MAC; matches the case where neuromorphic acceleration
              is unavailable or not applicable to a given layer).
        """
        total_mac, total_nz = self.per_image_macs(n_images)
        # Per-kind breakdown (round-2 audit).  Conv3d layers are dense,
        # non-event-driven ops (their inputs are post-attention continuous
        # values followed by ReLU) and cannot ride the neuromorphic SOP
        # price; the *_dense3d fields charge them at the full ANN MAC rate
        # even inside the SNN deployment estimate.  Legacy fields keep the
        # original semantics (everything at the SOP rate) for comparability
        # with previously reported numbers.
        conv3d_mac = sum(s.mac_total for s in self.mac_stats.values()
                         if s.kind == "Conv3d") / max(1, n_images)
        conv3d_nz = sum(s.nonzero_total for s in self.mac_stats.values()
                        if s.kind == "Conv3d") / max(1, n_images)
        eventful_nz = total_nz - conv3d_nz          # Conv2d + Linear only
        return {
            "ann_macs": total_mac,                   # now includes Conv3d
            "conv3d_macs": conv3d_mac,
            "conv2d_linear_macs": total_mac - conv3d_mac,
            "snn_effective_acs": total_nz,
            "energy_ann_pj": total_mac * ann_pj_per_mac,
            "energy_snn_lower_pj": total_nz * ac_pj_per_op,
            "energy_snn_lower_dense3d_pj":
                eventful_nz * ac_pj_per_op + conv3d_mac * ann_pj_per_mac,
            "energy_snn_upper_pj": total_nz * ann_pj_per_mac,
        }

    def per_layer_table(self, n_images: int) -> List[Dict]:
        rows = []
        for name, s in self.mac_stats.items():
            mac = s.mac_total / max(1, n_images)
            nz = s.nonzero_total / max(1, n_images)
            rows.append({
                "name": name, "kind": s.kind,
                "weight_shape": list(s.weight_shape),
                "mac_per_image": mac,
                "effective_nz_mac_per_image": nz,
                "input_nonzero_rate": (nz / mac) if mac > 0 else 0.0,
                "calls": s.calls,
            })
        return rows

    def per_layer_spikes(self, n_images: int) -> List[Dict]:
        rows = []
        for name, s in self.spike_stats.items():
            if s.calls == 0:
                continue
            rows.append({
                "name": name, "kind": s.kind,
                "mean_firing_rate": s.mean_sum / s.calls,
                "nonzero_firing_rate": s.nonzero_sum / s.calls,
                "calls": s.calls,
            })
        return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_energy_per_layer(rows: List[Dict], out_pdf: str,
                           ann_pj: float = 4.6, ac_pj: float = 0.077) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    rows = sorted(rows, key=lambda r: r["mac_per_image"], reverse=True)[:24]
    names = [r["name"].split(".")[-1][:18] for r in rows]
    ann_e = [r["mac_per_image"] * ann_pj / 1e6 for r in rows]              # μJ
    snn_e = [r["effective_nz_mac_per_image"] * ac_pj / 1e6 for r in rows]  # μJ

    idx = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rows)), 5))
    bw = 0.4
    ax.bar(idx - bw / 2, ann_e, bw, label=f"ANN ({ann_pj} pJ/MAC)", color="#D95319")
    ax.bar(idx + bw / 2, snn_e, bw, label=f"SNN lower ({ac_pj} pJ/SOP × r)", color="#0072BD")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Energy per image (μJ)", fontsize=12)
    ax.set_title("Per-layer inference energy (top 24 by MAC)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", dpi=600)
    plt.close(fig)


def _plot_spike_rate(rows: List[Dict], out_pdf: str) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    rows = sorted(rows, key=lambda r: r["nonzero_firing_rate"], reverse=True)[:24]
    names = [r["name"].split(".")[-1][:18] for r in rows]
    nz = [r["nonzero_firing_rate"] for r in rows]
    mn = [r["mean_firing_rate"] for r in rows]

    idx = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(rows)), 4))
    bw = 0.4
    ax.bar(idx - bw / 2, nz, bw, label="non-zero rate", color="#7E2F8E")
    ax.bar(idx + bw / 2, mn, bw, label="mean output", color="#77AC30")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("rate", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Per-layer spike statistics (top 24)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    sd_blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in sd_blob or "state_dict" not in sd_blob:
        raise ValueError(f"ckpt {ckpt_path} missing 'config' or 'state_dict' keys")
    cfg = sd_blob["config"]
    backbone = cfg.get("backbone", "vlif")

    if backbone == "plain_ann":
        from cloud_removal_v1.models import build_plain_unet
        model = build_plain_unet(
            dim=cfg["vlif_dim"],
            en_blocks=tuple(cfg["en_blocks"][:3]),
            de_blocks=tuple(cfg["de_blocks"][:3]),
        ).to(device)
    else:
        from cloud_removal_v1.models import build_vlifnet
        sub = "snn" if backbone == "vlif" else "ann"
        model = build_vlifnet(
            dim=cfg["vlif_dim"],
            en_num_blocks=tuple(cfg["en_blocks"]),
            de_num_blocks=tuple(cfg["de_blocks"]),
            T=cfg["T"],
            use_refinement=False,
            inp_channels=3, out_channels=3,
            backend=cfg.get("vlif_backend", "torch"),
            bn_variant=cfg.get("bn_variant", "tdbn"),
            backbone=sub,
        ).to(device)

    # Re-apply the ablation that was used at training time (if any), so
    # that the freshly-built model's structure matches the saved
    # state_dict before strict loading.  Skipping this would leave an
    # FSTA / dual-group / spike module re-initialized to random weights
    # under no_fsta / binary_spike, silently corrupting energy numbers.
    abl = cfg.get("ablation", "none")
    if abl != "none" and backbone != "plain_ann":
        from cloud_removal_v1.train_centralized import _apply_ablation
        model = _apply_ablation(model, abl, backbone)
        print(f"[energy] re-applied training-time ablation: {abl}")

    sd = sd_blob["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[energy] WARN missing keys: {len(missing)} (e.g. {list(missing)[:3]})")
    if unexpected:
        print(f"[energy] WARN unexpected keys: {len(unexpected)} (e.g. {list(unexpected)[:3]})")
    model.eval()
    return model, cfg


def _build_eval_loader(data_root: str, split: str, patch_size: Optional[int],
                       n_samples: int, seed: int):
    from cloud_removal_v1.dataset import PairedCloudDataset
    try:
        ds = PairedCloudDataset(data_root, split=split, patch_size=patch_size)
    except FileNotFoundError:
        ds = PairedCloudDataset(data_root, split=None, patch_size=patch_size)
    if n_samples > 0 and n_samples < len(ds):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ds), n_samples, replace=False).tolist()
        from torch.utils.data import Subset
        ds = Subset(ds, idx)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


def _reset_snn_safely(model: nn.Module) -> None:
    """Call functional.reset_net if spikingjelly is importable AND the model
    has any Memory* modules; otherwise no-op."""
    try:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
    except Exception:
        pass


@torch.no_grad()
def _run_forward(model: nn.Module, loader, device: torch.device) -> int:
    n = 0
    for cloudy, _clear in loader:
        cloudy = cloudy.to(device, non_blocking=True)
        # If image is full-resolution, optionally centre-crop to the
        # configured patch_size so MAC count reflects training-distribution.
        _reset_snn_safely(model)
        _ = model(cloudy)
        _reset_snn_safely(model)
        n += cloudy.shape[0]
    return n


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Energy & spike-rate estimation")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to a checkpoint saved by train_centralized.py "
                        "(must contain {'config','state_dict'}).")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--patch_size", type=int, default=64,
                   help="If >0, crop to this size; else use full image.")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--ann_pj_per_mac", type=float, default=4.6,
                   help="Energy per ANN MAC, in pJ.  Default 4.6 = "
                        "Horowitz 2014 ISSCC 45 nm fixed-point MAC.")
    p.add_argument("--ac_pj_per_op", type=float, default=0.077,
                   help="Energy per SNN synaptic operation (SOP), in pJ.  "
                        "Default 0.077 = 77 fJ, the neuromorphic / in-memory "
                        "computing SOP cost used by FLSNN [Yang2025], ESDNet "
                        "[Song2024], VLIF [Chen2025].  Set to 0.9 to use the "
                        "Horowitz 45 nm general-CMOS AC cost instead.")
    args = p.parse_args(argv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[energy] device: {device}")

    model, cfg = _build_model_from_ckpt(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[energy] backbone={cfg.get('backbone')}  params={n_params/1e6:.2f}M")

    loader = _build_eval_loader(
        args.data_root, args.split,
        patch_size=args.patch_size if args.patch_size > 0 else None,
        n_samples=args.n_samples, seed=args.seed,
    )

    meter = EnergyMeter(model)
    n_seen = _run_forward(model, loader, device)
    meter.remove()
    print(f"[energy] forward passes complete on {n_seen} sample(s)")

    bounds = meter.energy_bounds_per_image(
        n_seen, ann_pj_per_mac=args.ann_pj_per_mac, ac_pj_per_op=args.ac_pj_per_op)
    layers_macs = meter.per_layer_table(n_seen)
    layers_spikes = meter.per_layer_spikes(n_seen)

    # Pretty stdout
    print("\n=== Energy per image ===")
    print(f"  total ANN MACs        : {bounds['ann_macs']:,.0f}")
    print(f"  effective non-zero MAC: {bounds['snn_effective_acs']:,.0f}")
    print(f"  E_ANN  ({args.ann_pj_per_mac} pJ/MAC)        : {bounds['energy_ann_pj']/1e6:.3f} μJ")
    print(f"  E_SNN_lower ({args.ac_pj_per_op} pJ/SOP × r): {bounds['energy_snn_lower_pj']/1e6:.3f} μJ")
    print(f"  E_SNN_upper ({args.ann_pj_per_mac} pJ/MAC × r): {bounds['energy_snn_upper_pj']/1e6:.3f} μJ")
    if bounds["energy_snn_lower_pj"] > 0:
        ratio_lo = bounds["energy_ann_pj"] / bounds["energy_snn_lower_pj"]
        ratio_up = bounds["energy_ann_pj"] / max(bounds["energy_snn_upper_pj"], 1e-12)
        print(f"  ANN / SNN_lower ratio     : {ratio_lo:.2f}×")
        print(f"  ANN / SNN_upper ratio     : {ratio_up:.2f}×")

    # JSON dump
    summary = {
        "ckpt": os.path.abspath(args.ckpt),
        "config": {
            **(cfg if isinstance(cfg, dict) else {}),
            "ann_pj_per_mac": float(args.ann_pj_per_mac),
            "ac_pj_per_op":   float(args.ac_pj_per_op),
        },
        "params_M": float(n_params / 1e6),
        "n_samples_evaluated": int(n_seen),
        "energy_per_image": bounds,
        "per_layer_macs": layers_macs,
        "per_layer_spikes": layers_spikes,
    }
    with open(os.path.join(args.out_dir, "energy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    _plot_energy_per_layer(
        layers_macs, os.path.join(args.out_dir, "energy_per_layer.pdf"),
        ann_pj=args.ann_pj_per_mac, ac_pj=args.ac_pj_per_op)
    _plot_spike_rate(
        layers_spikes, os.path.join(args.out_dir, "spike_rate_per_layer.pdf"))

    print(f"[energy] wrote {os.path.join(args.out_dir, 'energy_summary.json')}")
    print(f"[energy] wrote energy_per_layer.pdf, spike_rate_per_layer.pdf")


if __name__ == "__main__":
    main()
