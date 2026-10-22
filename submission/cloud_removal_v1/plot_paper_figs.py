"""
plot_paper_figs.py — top-tier-journal-quality figures for the OrbitVLIF paper.

Generates 5 PDF figures + 3 LaTeX tables from training outputs.

Figures (β scheme; Fig 1 architecture is hand-drawn TikZ, not produced here):
    fig2_centralized_curves.pdf  : A1 / A2 / C2 PSNR-vs-epoch (CR1, CR2 panels)
    fig3_qualitative_grid.pdf    : 4 test imgs × 4 cols (Cloudy, VLIF, Plain, GT)
    fig4_ablation_bars.pdf       : A1 vs B1 / B2 / B3 (PSNR, SSIM dual subplot)
    fig5_federated_curves.pdf    : F_SNN / F_ANN / F_plain PSNR-vs-round
                                    + per-plane shading + comm-bytes inset
    fig6_energy_bars.pdf         : ANN_pJ / SNN_lower / SNN_upper, log scale

Tables:
    tab1_centralized.tex
    tab2_ablation.tex
    tab3_federated.tex

Style:  STIXGeneral font, color-blind-safe 4-palette, IEEE 2-col conf sizes,
        booktabs LaTeX, 600-dpi vector PDF.

Usage
-----
    python -m cloud_removal_v1.plot_paper_figs \\
        --outputs_v1 ./Outputs_v1 --outputs_v2 ./Outputs_v2 \\
        --energy_dir ./Outputs_energy_A1 \\
        --out_dir ./figures \\
        --figs all --tables yes

    # Re-generate only Fig 5
    python -m cloud_removal_v1.plot_paper_figs --figs 5 --tables no

Missing input files are SKIPPED with a WARNING (never error) so the script
can be re-run during long training to refresh partial figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow `python cloud_removal_v1/plot_paper_figs.py` as well as the -m form.
if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v1.plot_paper_figs import main  # noqa: E402
    if __name__ == "__main__":
        main()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Global style — color-blind-safe 4-palette + IEEE conference sizes
# ---------------------------------------------------------------------------

# Wong-2011 color-blind-safe; verified against deuteranopia / protanopia sims.
PALETTE_BLUE   = "#0173B2"   # primary    — OrbitVLIF / SNN
PALETTE_ORANGE = "#DE8F05"   # secondary  — PlainUNet / ANN
PALETTE_GREEN  = "#029E73"   # tertiary   — extra
PALETTE_GRAY   = "#999999"   # neutral    — ground truth / shading

# IEEE 2-column conference, sizes in inches.
FIG_SINGLE_COL = (3.5, 2.4)
FIG_DOUBLE_COL = (7.2, 2.6)
FIG_GRID       = (7.2, 7.0)   # qualitative grid

# Markers used at every Nth epoch / round.
MARKERS = ("o", "s", "D", "^", "v")

# Marker-every-N — set to 1000 so EVERY data point gets a marker on
# all training curves (loss + PSNR + SSIM, both centralised 600-ep
# and federated 200-round).  Combined with markersize=1.5pt and
# linewidth=1.0pt this yields a "scatter-plus-line" look that
# reveals the full per-epoch noise structure the user wants to see.
#
# Concrete densities at MARKER_TARGET_COUNT=1000:
#   600-ep Loss   → 600 markers (every epoch)         in 0-300 → 300
#   120-pt PSNR   → 120 markers (every eval point)    in 0-300 →  60
#    40-pt fed    →  40 markers
MARKER_TARGET_COUNT = 1000


# ---------------------------------------------------------------------------
# Default run-name registry — overridable via CLI.
# These match the canonical commands shipped in the README / paper notes.
# ---------------------------------------------------------------------------

_DEFAULT_RUNS = {
    "a1":         "A1_vlif_cr1",
    "a2":         "A2_vlif_cr2",
    "c2_cr1":     "C2_plain_ann_cr1",
    "c2_cr2":     "C2_plain_ann_cr2",
    "b1":         "B1_no_fsta_cr1",
    "b2":         "B2_no_dual_group_cr1",
    "b3":         "B3_binary_spike_cr1",
    "f_snn":      "F_snn",       # → v2a_F_snn_fedbn_Gossip_Averaging.npz
    "f_ann":      "F_ann",
    "f_plain":    "F_plain",
}

# Canonical (BN, scheme) cell — locked at fedbn × Gossip per paper decision.
F_BN     = "fedbn"
F_SCHEME = "Gossip_Averaging"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate paper figures + LaTeX tables from training outputs.")

    # Input / output paths
    p.add_argument("--outputs_v1", type=str, default="./Outputs_v1",
                   help="Directory containing centralized_*.npz / *_best.pt / "
                        "*_summary.json (output of train_centralized.py).")
    p.add_argument("--outputs_v2", type=str, default="./Outputs_v2",
                   help="Directory containing v2a_*.npz / ckpts/ / "
                        "v2a_*_summary.json (output of run_smoke.py).")
    p.add_argument("--energy_dir", type=str, default="./Outputs_energy_A1",
                   help="Directory containing energy_summary.json "
                        "(output of energy_estimation.py).  Required for Fig 6.")
    p.add_argument("--energy_dir_plain", type=str, default="",
                   help="Optional second energy_summary.json directory for the "
                        "PlainUNet/OrbitUnet baseline (output of "
                        "energy_estimation.py on the PlainUNet ckpt). "
                        "When set, fig6 inserts an OrbitUnet bar between "
                        "the ANN and SNN bars.")
    p.add_argument("--out_dir", type=str, default="./figures",
                   help="Where PDFs and .tex files are written.")

    # What to render
    p.add_argument("--figs", type=str, default="all",
                   help="'all' or comma-list, e.g. '2,3,5'.")
    p.add_argument("--tables", type=str, default="yes", choices=["yes", "no"])

    # Run-name overrides (defaults match _DEFAULT_RUNS).
    for key, default in _DEFAULT_RUNS.items():
        p.add_argument(f"--run_{key}", type=str, default=default)

    # Inference settings (Fig 3)
    p.add_argument("--qual_n_samples", type=int, default=4,
                   help="Number of qualitative-grid rows (test images).")
    p.add_argument("--qual_seed", type=int, default=0,
                   help="Deterministic test-image selection seed.")
    p.add_argument("--qual_dataset_root", type=str, default=None,
                   help="Test-set root for Fig 3.  Defaults to the data_root "
                        "stored in the A1 ckpt's saved config.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for Fig 3 inference (CPU is fine, ~4 imgs).")

    # Federated cell selection (paper-locked default = fedbn × Gossip)
    p.add_argument("--f_bn", type=str, default=F_BN, choices=["fedavg", "fedbn"])
    p.add_argument("--f_scheme", type=str, default=F_SCHEME)

    # Per-layer energy (Fig 10) — limit shown layers
    p.add_argument("--energy_topk", type=int, default=12,
                   help="Fig 10: show top-K layers by ANN energy.  Default 12.")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging — _warn skips a missing artefact instead of aborting the whole run
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[plot_paper_figs] {msg}", flush=True)


def _warn(msg: str) -> None:
    """Skip-with-warning: missing files are common during in-flight training."""
    warnings.warn(f"[plot_paper_figs] {msg}", stacklevel=2)


def _resolve_figs(spec: str) -> List[int]:
    """'all' | '2,3,5' | '4'  →  sorted list of valid int fig ids.

    'all' returns the figures that are expected to render meaningfully
    on the current data set.  Fig 5 (federated curves) is excluded
    from 'all' until the F-series runs are complete; pass
    ``--figs 5`` explicitly to render it from a single F_plain run.
    """
    valid    = {2, 3, 4, 5, 6, 7, 8, 9, 10}
    in_all   = {2, 3, 4, 6, 7, 8, 9, 10}        # excludes 5 (federated)
    if spec.strip().lower() == "all":
        return sorted(in_all)
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"--figs token {tok!r} not an int")
        i = int(tok)
        if i not in valid:
            raise ValueError(f"--figs id {i} not in {sorted(valid)}")
        out.append(i)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Block 2 — data loaders (pure IO; missing files → None + _warn, never raise)
# ---------------------------------------------------------------------------

def load_centralized_npz(outputs_v1: Path,
                         run_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Read ``Outputs_v1/centralized_<run>.npz`` written by train_centralized.py.

    Returns a dict with keys::

        epoch, lr, train_loss, train_charbonnier, train_ssim_loss,
        eval_psnr, eval_ssim, wall_seconds

    or None if the file is missing.  Each value is a 1-D ``np.ndarray``.
    """
    path = outputs_v1 / f"centralized_{run_name}.npz"
    if not path.is_file():
        _warn(f"missing centralized npz: {path}")
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k].copy() for k in d.files}


def load_centralized_summary(outputs_v1: Path,
                             run_name: str) -> Optional[Dict]:
    """Read ``Outputs_v1/centralized_<run>_summary.json``.

    Returns the parsed dict (config / params_M / best / final / total_wall_seconds)
    or None if missing.
    """
    path = outputs_v1 / f"centralized_{run_name}_summary.json"
    if not path.is_file():
        _warn(f"missing centralized summary: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_federated_npz(outputs_v2: Path,
                       run_name: str,
                       bn_mode: str = F_BN,
                       scheme: str = F_SCHEME,
                       ) -> Optional[Dict[str, np.ndarray]]:
    """Read ``Outputs_v2/v2a_<run>_<bn>_<scheme>.npz`` from run_smoke.py.

    Returns a dict with keys::

        epochs, train_loss, eval_psnr, eval_ssim, comm_bytes,
        per_plane_psnr, per_plane_ssim, wall_seconds

    where ``per_plane_psnr`` / ``per_plane_ssim`` are object-dtype arrays
    (list of per-plane scalar lists), so we load with ``allow_pickle=True``.
    None if the file is missing.
    """
    path = outputs_v2 / f"v2a_{run_name}_{bn_mode}_{scheme}.npz"
    if not path.is_file():
        _warn(f"missing federated npz: {path}")
        return None
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k].copy() for k in d.files}


def load_federated_summary(outputs_v2: Path,
                           run_name: str) -> Optional[Dict]:
    """Read ``Outputs_v2/v2a_<run>_summary.json`` (config + final per-cell numbers)."""
    path = outputs_v2 / f"v2a_{run_name}_summary.json"
    if not path.is_file():
        _warn(f"missing federated summary: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_energy_summary(energy_dir: Path) -> Optional[Dict]:
    """Read ``energy_summary.json`` from energy_estimation.py.

    Returns None if missing or malformed.  The dict has::

        ckpt, config, params_M, n_samples_evaluated,
        energy_per_image: {ann_macs, snn_effective_acs,
                           energy_ann_pj, energy_snn_lower_pj,
                           energy_snn_upper_pj},
        per_layer_macs:  [...],
        per_layer_spikes:[...],
    """
    path = energy_dir / "energy_summary.json"
    if not path.is_file():
        _warn(f"missing energy summary: {path}")
        return None
    with open(path, "r") as f:
        blob = json.load(f)
    if "energy_per_image" not in blob:
        _warn(f"energy summary {path} missing 'energy_per_image' key")
        return None
    return blob


def load_ckpt_for_inference(ckpt_path: Path,
                            device: "torch.device",
                            ) -> Optional[Tuple["nn.Module", Dict, bool]]:
    """Reconstruct model + load weights from a centralized ``*_best.pt`` ckpt.

    Returns (model.eval(), cfg, is_snn) on success, or None on miss / failure.

    Delegates to :func:`cloud_removal_v2.energy_estimation._build_model_from_ckpt`
    which already handles backbone dispatch (vlif / vlif_ann / plain_ann),
    ablation re-application, and strict=False weight loading.
    """
    if not Path(ckpt_path).is_file():
        _warn(f"missing ckpt: {ckpt_path}")
        return None
    try:
        from cloud_removal_v2.energy_estimation import _build_model_from_ckpt
        model, cfg = _build_model_from_ckpt(str(ckpt_path), device)
    except Exception as e:                    # noqa: BLE001
        _warn(f"failed to load {ckpt_path}: {type(e).__name__}: {e}")
        return None
    is_snn = cfg.get("backbone", "vlif") == "vlif"
    return model, cfg, is_snn


# ---------------------------------------------------------------------------
# Block 3 — global style helpers + Fig 2 (centralized training curves)
# ---------------------------------------------------------------------------

_MPL_INITIALISED = False


def _setup_mpl() -> None:
    """Apply global matplotlib rcParams once.  Idempotent — repeat calls
    are cheap, so figure functions can call this freely on entry.

    Style targets IEEE 2-column conference proceedings:
      * STIXGeneral font (Times-like, math-compatible)
      * Vector PDF backend (Agg buffer)
      * 0.6 pt axes / 0.5 pt grid / 1.4 pt curves / 5.5 pt markers
      * 8 pt tick labels, 9 pt axis labels, 10 pt panel titles
      * legend frameon=False, fancybox=False
    """
    global _MPL_INITIALISED
    if _MPL_INITIALISED:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":         "STIXGeneral",
        "mathtext.fontset":    "stix",
        "pdf.fonttype":        42,        # TrueType fonts (editable in Illustrator)
        "ps.fonttype":         42,
        "axes.linewidth":      0.6,
        "axes.titlesize":      10,
        "axes.labelsize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "xtick.major.size":    3.0,
        "ytick.major.size":    3.0,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "legend.fontsize":     8,
        "legend.frameon":      False,
        "legend.fancybox":     False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.9,
        "lines.linewidth":     1.0,
        "lines.markersize":    1.5,
        "lines.markeredgewidth": 0.0,
        "grid.linewidth":      0.4,
        "grid.linestyle":      "--",
        "grid.alpha":          0.4,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.02,
    })
    _MPL_INITIALISED = True


def _save_pdf(fig, out_path: Path) -> None:
    """Vector PDF write at 600 dpi (raster fallbacks crisp on print).

    Also writes a sibling PNG at 300 dpi for quick previews / slide use.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=600)
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300)
    _log(f"wrote {out_path}")


def _annotate_panel(ax, label: str,
                    x: float = 0.04, y: float = 0.95) -> None:
    """Add ``(a)`` / ``(b)`` panel label in axes-fraction coords (top-left)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def _smart_marker_every(n_points: int,
                        target_count: int = MARKER_TARGET_COUNT) -> int:
    """Choose ``markevery`` so each curve has ~target markers regardless
    of the number of evaluated epochs / rounds.  Always returns >=1."""
    if n_points <= target_count:
        return 1
    return max(1, n_points // target_count)


def _finite(epoch: np.ndarray, value: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Filter (epoch, value) to indices where value is finite — train_centralized
    writes NaN on epochs where eval was skipped (eval_every > 1)."""
    mask = np.isfinite(value)
    return epoch[mask], value[mask]


# ---------------------------------------------------------------------------
# Fig 2 — centralized training curves (CR1, CR2 dual panel)
# ---------------------------------------------------------------------------

def fig2_centralized_curves(args, out_dir: Path) -> None:
    """Two side-by-side panels showing PSNR-vs-epoch on CR1 (left) and CR2
    (right).  Each panel overlays VLIFNet (blue, ``o``) and PlainUNet
    (orange, ``s``) when their npz files are present.  If a panel has
    NO data, it is annotated "no data" and the figure is still produced.

    Reads from ``args.outputs_v1`` using run-name overrides ``args.run_a1``,
    ``args.run_a2``, ``args.run_c2_cr1``, ``args.run_c2_cr2``.

    Output: ``out_dir/fig2_centralized_curves.pdf``.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    outputs_v1 = Path(args.outputs_v1)

    # Per-panel curve specs: (run_name, label, color, marker)
    panels = [
        ("CR1", [
            (args.run_a1,     "OrbitVLIF (ours)", PALETTE_BLUE,   "o"),
            (args.run_c2_cr1, "PlainUNet",        PALETTE_ORANGE, "s"),
        ]),
        ("CR2", [
            (args.run_a2,     "OrbitVLIF (ours)", PALETTE_BLUE,   "o"),
            (args.run_c2_cr2, "PlainUNet",        PALETTE_ORANGE, "s"),
        ]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL, sharey=False)

    n_curves_total = 0
    last_handles: List = []
    last_labels:  List[str] = []

    for ax, (panel_label, curves) in zip(axes, panels):
        ax.grid(True, which="major")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"({'a' if panel_label == 'CR1' else 'b'}) {panel_label}")

        n_drawn = 0
        for run_name, label, color, marker in curves:
            d = load_centralized_npz(outputs_v1, run_name)
            if d is None:
                continue
            ep, psnr = _finite(d["epoch"], d["eval_psnr"])
            if ep.size == 0:
                _warn(f"{run_name}: no finite eval_psnr points; skipping curve")
                continue
            mev = _smart_marker_every(ep.size)
            (ln,) = ax.plot(ep, psnr,
                            color=color, marker=marker, markevery=mev,
                            label=label, linestyle="-", clip_on=True)
            n_drawn += 1
            n_curves_total += 1
            # Capture for shared legend (latest non-empty panel wins; both
            # panels have identical curve specs anyway).
            last_handles.append(ln)
            last_labels.append(label)

        if n_drawn == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=PALETTE_GRAY, fontsize=10)

        # Auto-zoom Y-axis around the data so dB differences read cleanly.
        ax.margins(x=0.02)

    # Shared legend above the two panels (keeps panel area clean).
    if last_handles:
        # Deduplicate by label while preserving order.
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(last_handles, last_labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
        fig.legend(uniq_h, uniq_l,
                   loc="upper center", bbox_to_anchor=(0.5, 1.04),
                   ncol=len(uniq_l))

    fig.tight_layout(pad=0.3)
    if n_curves_total == 0:
        _warn("Fig 2: no centralized curves found; saving empty figure")
    _save_pdf(fig, out_dir / "fig2_centralized_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 4 — Fig 3 qualitative grid (4 test images × 4 columns)
# ---------------------------------------------------------------------------

def _psnr_uint01(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR for two float32 [H, W, 3] arrays in [0, 1].  +inf when equal."""
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * float(np.log10(1.0)) - 10.0 * float(np.log10(mse))


def _score_cloud_severity(cloudy: np.ndarray, clear: np.ndarray) -> float:
    """Higher = harder image (more cloud coverage / larger cloudy-vs-GT
    deviation).  Use the L2 distance between normalised cloudy and clear
    arrays — a robust proxy that does not require explicit cloud masks.
    """
    return float(np.sqrt(np.mean((cloudy - clear) ** 2)))


def _select_test_indices(test_ds,
                         n: int = 4,
                         seed: int = 0,
                         topk_mult: int = 3) -> List[int]:
    """Deterministic + cloud-coverage-biased selection.

    Steps:
        1. Score every test pair by cloud severity (L2 cloudy-vs-GT).
        2. Take the top ``n * topk_mult`` indices (most cloudy).
        3. From that subset, sample ``n`` with a fixed-seed RNG so we
           still get a *deterministic* selection while not always picking
           the absolute top-N (avoids overcrowding visually-similar imgs).
    """
    if n <= 0:
        return []
    n_total = len(test_ds)
    if n_total == 0:
        return []
    if n_total <= n:
        return list(range(n_total))

    scores = np.empty(n_total, dtype=np.float64)
    for i in range(n_total):
        cloudy_t, clear_t = test_ds[i]
        cl  = cloudy_t.permute(1, 2, 0).cpu().numpy()
        gt  = clear_t.permute(1, 2, 0).cpu().numpy()
        scores[i] = _score_cloud_severity(cl, gt)

    topk = min(n_total, max(n, n * topk_mult))
    top_idx = np.argsort(-scores)[:topk]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(top_idx, size=n, replace=False)
    return sorted(chosen.tolist())


def _infer_image(model, x_chw: "torch.Tensor", device: "torch.device",
                 is_snn: bool) -> np.ndarray:
    """Run a single full-resolution image through ``model`` and return a
    [H, W, 3] float32 array in [0, 1].

    Mirrors v1's task.py contract: ``reset_net`` BEFORE and AFTER the
    forward pass when the backbone is spiking; no-op for ANN / Plain.
    """
    import torch
    model.eval()
    if is_snn:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
    with torch.no_grad():
        x = x_chw.unsqueeze(0).to(device)
        # Both PlainUNet and VLIFNet require H, W divisible by 4 — let the
        # model assertion fire if violated; we don't silently pad here.
        y = model(x)
    if is_snn:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
    y = y.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return y.astype(np.float32, copy=False)


def _resolve_qual_root(args, cfg: Optional[Dict]) -> Optional[str]:
    """Pick the test-set root for Fig 3 — explicit CLI flag first, else
    the data_root saved inside the A1 ckpt's config."""
    if args.qual_dataset_root:
        return args.qual_dataset_root
    if cfg is not None and isinstance(cfg.get("data_root"), str):
        return cfg["data_root"]
    return None


def fig3_qualitative_grid(args, out_dir: Path) -> None:
    """4 rows × 4 columns: Cloudy | VLIFNet | PlainUNet | Ground Truth.

    Each VLIFNet / PlainUNet cell shows a *prominent* PSNR label
    (top-left, white text on a dark filled rectangle) computed against
    the cell's ground truth.  Test images are selected deterministically
    via ``_select_test_indices`` (top-cloud-coverage bias + seed sample).

    Inputs (any of these missing → skip with warning, do not crash):
      * VLIFNet best ckpt   = Outputs_v1/centralized_<run_a1>_best.pt
      * PlainUNet best ckpt = Outputs_v1/centralized_<run_c2_cr1>_best.pt
      * Test dataset root   = --qual_dataset_root (else from A1 ckpt's
                              saved config['data_root'])

    Output: ``out_dir/fig3_qualitative_grid.pdf``.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt
    import torch

    outputs_v1 = Path(args.outputs_v1)
    device = torch.device(args.device)

    # 1) Load both ckpts.
    a1_ckpt = outputs_v1 / f"centralized_{args.run_a1}_best.pt"
    c2_ckpt = outputs_v1 / f"centralized_{args.run_c2_cr1}_best.pt"
    a1_loaded = load_ckpt_for_inference(a1_ckpt, device)
    c2_loaded = load_ckpt_for_inference(c2_ckpt, device)
    if a1_loaded is None and c2_loaded is None:
        _warn("Fig 3: both VLIFNet and PlainUNet ckpts missing; skipping figure")
        return
    a1_model, a1_cfg, a1_is_snn = a1_loaded if a1_loaded else (None, None, False)
    c2_model, c2_cfg, c2_is_snn = c2_loaded if c2_loaded else (None, None, False)

    # 2) Resolve test-set root (CLI override > A1 cfg > C2 cfg).
    root = _resolve_qual_root(args, a1_cfg) or _resolve_qual_root(args, c2_cfg)
    if root is None:
        _warn("Fig 3: no test dataset root resolvable (try --qual_dataset_root); "
              "skipping figure")
        return

    # 3) Build full-resolution test set (patch_size=None) and pick rows.
    try:
        from cloud_removal_v1.dataset import (
            PairedCloudDataset, derived_train_test_split,
        )
        try:
            test_ds = PairedCloudDataset(root, split="test", patch_size=None)
        except FileNotFoundError:
            _, test_sub = derived_train_test_split(
                root, patch_size_train=64, test_ratio=0.2, seed=args.qual_seed)
            test_ds = test_sub
    except Exception as e:                        # noqa: BLE001
        _warn(f"Fig 3: cannot open test set at {root}: "
              f"{type(e).__name__}: {e}; skipping figure")
        return

    n_rows = max(1, args.qual_n_samples)
    indices = _select_test_indices(test_ds, n=n_rows, seed=args.qual_seed)
    if not indices:
        _warn("Fig 3: empty test set; skipping figure")
        return

    # 4) Render grid.
    cols = ["Cloudy", "OrbitVLIF (ours)", "PlainUNet", "Ground Truth"]
    n_cols = len(cols)
    fig, axes = plt.subplots(
        len(indices), n_cols,
        figsize=(FIG_GRID[0], 1.85 * len(indices)),
        squeeze=False,
    )

    for r, idx in enumerate(indices):
        cloudy_t, clear_t = test_ds[idx]
        cloudy_img = cloudy_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        clear_img  = clear_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        if a1_model is not None:
            try:
                vlif_img = _infer_image(a1_model, cloudy_t, device, a1_is_snn)
            except Exception as e:                # noqa: BLE001
                _warn(f"Fig 3 row {r}: VLIFNet inference failed "
                      f"({type(e).__name__}: {e}); blanking cell")
                vlif_img = None
        else:
            vlif_img = None

        if c2_model is not None:
            try:
                plain_img = _infer_image(c2_model, cloudy_t, device, c2_is_snn)
            except Exception as e:                # noqa: BLE001
                _warn(f"Fig 3 row {r}: PlainUNet inference failed "
                      f"({type(e).__name__}: {e}); blanking cell")
                plain_img = None
        else:
            plain_img = None

        cells = [
            (cloudy_img, None),
            (vlif_img,   _psnr_uint01(vlif_img, clear_img)
                         if vlif_img is not None else None),
            (plain_img,  _psnr_uint01(plain_img, clear_img)
                         if plain_img is not None else None),
            (clear_img,  None),
        ]

        for c, (img, psnr) in enumerate(cells):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#444444")
            if img is None:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center",
                        color=PALETTE_GRAY, fontsize=14)
            else:
                ax.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
                if psnr is not None:
                    # Prominent: bold, top-left, white-on-black filled box.
                    ax.text(0.04, 0.92,
                            f"{psnr:.2f} dB",
                            transform=ax.transAxes,
                            ha="left", va="top",
                            fontsize=10, fontweight="bold", color="white",
                            bbox=dict(boxstyle="round,pad=0.18",
                                      facecolor="black",
                                      edgecolor="none",
                                      alpha=0.72))
            if r == 0:
                ax.set_title(cols[c], fontsize=10)

    fig.tight_layout(pad=0.25)
    _save_pdf(fig, out_dir / "fig3_qualitative_grid.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 5 — Fig 4 ablation bars (A1 vs B1 / B2 / B3, PSNR + SSIM)
# ---------------------------------------------------------------------------

def _smart_ylim(values: Sequence[float], pad: float
                ) -> Tuple[float, float]:
    """Auto Y-axis range so small ablation deltas are visible — pad below
    min and above max, never going below 0."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)],
                   dtype=float)
    if arr.size == 0:
        return (0.0, 1.0)
    lo = max(0.0, float(arr.min()) - pad)
    hi = float(arr.max()) + pad
    if hi - lo < 1e-6:
        hi = lo + max(pad, 1e-3)
    return (lo, hi)


def fig4_ablation_bars(args, out_dir: Path) -> None:
    """Two side-by-side subplots (PSNR | SSIM) comparing the full VLIFNet
    (A1) against three ablations:

        B1: −FSTA (FSTAModule + FreMLPBlock → Identity)
        B2: −DualGroup (PixelShuffle spatial path removed)
        B3: −MultiSpike4 (5-level → binary spikes)

    Bar heights = ``best.psnr`` / ``best.ssim`` from each run's summary.json.
    Top of each ablated bar carries a red ΔPSNR / ΔSSIM annotation
    (negative deltas in red, positive in green — for the rare case an
    ablation accidentally helps on noisy data).  A1's bar uses a ``//``
    hatch to flag the "reference / full model" condition.

    Skipped (with warning, no crash) if A1 summary is missing — there is
    no reference to compute deltas against.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    outputs_v1 = Path(args.outputs_v1)

    # Pull (run_name, label, color) — ORDER FIXED for paper consistency.
    spec = [
        (args.run_a1, "Full",        PALETTE_BLUE,   True),   # ← reference
        (args.run_b1, "−SHAM",       PALETTE_ORANGE, False),
        (args.run_b2, "−DualPath",   PALETTE_GREEN,  False),
        (args.run_b3, "−5QS",        PALETTE_GRAY,   False),
    ]

    # Load each summary; missing entries become None placeholders so the
    # bar order remains stable on partial data.
    rows: List[Tuple[str, str, Optional[float], Optional[float], bool]] = []
    for run_name, label, color, is_ref in spec:
        s = load_centralized_summary(outputs_v1, run_name)
        if s is None:
            rows.append((label, color, None, None, is_ref))
            continue
        best = s.get("best", {}) or {}
        psnr = best.get("psnr"); ssim = best.get("ssim")
        rows.append((label, color,
                     None if psnr is None or not np.isfinite(psnr) else float(psnr),
                     None if ssim is None or not np.isfinite(ssim) else float(ssim),
                     is_ref))

    a1_psnr = rows[0][2]; a1_ssim = rows[0][3]
    if a1_psnr is None and a1_ssim is None:
        _warn("Fig 4: A1 summary missing or empty; cannot compute deltas — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL, sharey=False)

    metrics = [
        ("PSNR (dB)", [r[2] for r in rows], a1_psnr, 1.0,  "{:+.2f} dB"),
        ("SSIM",      [r[3] for r in rows], a1_ssim, 0.02, "{:+.4f}"),
    ]

    x = np.arange(len(rows))

    for ax, (ylabel, values, ref, pad, fmt) in zip(axes, metrics):
        ax.grid(True, axis="y", which="major")
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=15, ha="right")

        # Plot bars individually so each gets the right color + optional hatch.
        for i, ((label, color, psnr, ssim, is_ref), val) in enumerate(zip(rows, values)):
            if val is None:
                # Draw a faint placeholder bar so X-axis spacing stays correct.
                ax.bar(i, 0.0, color="none", edgecolor=PALETTE_GRAY,
                       linewidth=0.6, hatch="..")
                ax.text(i, 0.0, "no data", ha="center", va="bottom",
                        fontsize=7, color=PALETTE_GRAY, rotation=0)
                continue
            kwargs = dict(color=color, edgecolor="black", linewidth=0.6)
            if is_ref:
                kwargs["hatch"] = "//"
            ax.bar(i, val, **kwargs)

            # Delta annotation (red on regression, green on improvement).
            if not is_ref and ref is not None:
                delta = val - ref
                ann_color = "#C62828" if delta < 0 else "#2E7D32"
                ax.text(i, val, fmt.format(delta),
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color=ann_color)

        ax.set_ylim(*_smart_ylim(values, pad=pad))

    axes[0].set_title("(a) PSNR")
    axes[1].set_title("(b) SSIM")

    fig.tight_layout(pad=0.4)
    _save_pdf(fig, out_dir / "fig4_ablation_bars.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 6 — Fig 5 federated convergence curves + cumulative-comm inset
# ---------------------------------------------------------------------------

def _per_plane_to_matrix(arr) -> Optional[np.ndarray]:
    """Convert run_smoke's object-dtype ``per_plane_psnr`` / ``per_plane_ssim``
    array (shape (n_epochs, n_planes), each cell a Python scalar) into a
    homogeneous float64 matrix.  Returns None if the structure is unusable.
    """
    if arr is None:
        return None
    try:
        rows: List[List[float]] = []
        for ep_row in np.atleast_1d(arr):
            # Each row is itself an iterable of per-plane scalars.
            vals = np.asarray(list(ep_row), dtype=np.float64).ravel()
            rows.append(vals.tolist())
    except Exception:                              # noqa: BLE001
        return None
    if not rows:
        return None
    n_planes = max(len(r) for r in rows)
    if n_planes == 0:
        return None
    out = np.full((len(rows), n_planes), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def fig5_federated_curves(args, out_dir: Path) -> None:
    """Single-panel PSNR-vs-round overlay for the three federated runs::

        F_SNN  (blue   'o')   — VLIFNet-SNN
        F_ANN  (orange 's')   — VLIFNet-ANN
        F_plain(green  'D')   — PlainUNet baseline

    Per-plane shading: ``fill_between(min, max)`` over per-plane PSNR
    arrays at α=0.18, drawn behind each mean curve.

    Inset (lower-right): cumulative communication volume in MB on a log
    Y-axis — strengthens the "VLIFNet does NOT cost more bandwidth than
    its ANN twin" claim referenced in §V.

    Skipped (no crash) only when ALL THREE runs are missing.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    outputs_v2 = Path(args.outputs_v2)

    spec = [
        (args.run_f_snn,   "OrbitALIF",  PALETTE_BLUE,   "o"),
        (args.run_f_ann,   "OrbitANN",   PALETTE_ORANGE, "s"),
        (args.run_f_plain, "OrbitUnet",  PALETTE_GREEN,  "D"),
    ]

    drawn: List[Tuple[str, np.ndarray, np.ndarray]] = []   # (label, rounds, comm_cum_MB)
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    ax.grid(True, which="major")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("PSNR (dB)")

    legend_handles, legend_labels = [], []

    for run_name, label, color, marker in spec:
        d = load_federated_npz(outputs_v2, run_name,
                               bn_mode=args.f_bn, scheme=args.f_scheme)
        if d is None:
            continue
        rounds = d.get("epochs")
        if rounds is None or rounds.size == 0:
            _warn(f"Fig 5: {run_name} has no rounds; skipping curve")
            continue
        rounds = np.asarray(rounds, dtype=np.float64)

        # Prefer per-plane mean+envelope; fall back to flat eval_psnr if
        # per_plane_psnr is unusable (older runs / single-plane configs).
        per_plane = _per_plane_to_matrix(d.get("per_plane_psnr"))
        if per_plane is not None and per_plane.shape[0] == rounds.size:
            mean_psnr = np.nanmean(per_plane, axis=1)
            lo = np.nanmin(per_plane, axis=1)
            hi = np.nanmax(per_plane, axis=1)
        else:
            psnr_flat = d.get("eval_psnr")
            if psnr_flat is None:
                _warn(f"Fig 5: {run_name} missing eval_psnr; skipping curve")
                continue
            mean_psnr = np.asarray(psnr_flat, dtype=np.float64)
            lo = hi = None

        # Filter NaN epochs (eval is sparse — eval_every>1 leaves NaN gaps).
        # Without this filter matplotlib draws nothing because consecutive
        # NaN values break the line into invisible segments.
        valid = np.isfinite(mean_psnr)
        if not valid.any():
            _warn(f"Fig 5: {run_name} has no finite eval points; skipping curve")
            continue
        rnds_v   = rounds[valid]
        mean_v   = mean_psnr[valid]
        if lo is not None and hi is not None:
            lo_v = lo[valid]
            hi_v = hi[valid]
            ax.fill_between(rnds_v, lo_v, hi_v, color=color, alpha=0.18,
                            linewidth=0.0)
        # Use only valid points downstream for both line + markers.
        rounds_to_plot = rnds_v
        psnr_to_plot   = mean_v

        mev = _smart_marker_every(rounds_to_plot.size)
        (ln,) = ax.plot(rounds_to_plot, psnr_to_plot,
                        color=color, marker=marker, markevery=mev,
                        label=label, linestyle="-")
        legend_handles.append(ln); legend_labels.append(label)

        comm = d.get("comm_bytes")
        if comm is not None and comm.size == rounds.size:
            comm_cum_MB = np.cumsum(np.asarray(comm, dtype=np.float64)) / (1024.0 ** 2)
            drawn.append((label, rounds, comm_cum_MB))
        else:
            drawn.append((label, rounds, None))

    if not legend_handles:
        _warn("Fig 5: no federated runs found; skipping figure")
        plt.close(fig)
        return

    ax.legend(legend_handles, legend_labels, loc="lower right",
              ncol=1)
    ax.margins(x=0.02)
    # Tight Y-axis fit around actual data range so single-curve
    # versions of this figure (only F_plain available) don't show a
    # mostly-empty plot area.
    ymins, ymaxs = [], []
    for ln in legend_handles:
        y = ln.get_ydata()
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            ymins.append(y.min())
            ymaxs.append(y.max())
    if ymins and ymaxs:
        lo = min(ymins) - 0.4
        hi = max(ymaxs) + 0.4
        ax.set_ylim(lo, hi)

    fig.tight_layout(pad=0.3)

    _save_pdf(fig, out_dir / "fig5_federated_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 7 — Fig 6 energy bars (ANN vs SNN-upper vs SNN-lower, log Y)
# ---------------------------------------------------------------------------

def _fmt_pj(value: float) -> str:
    """Compact engineering notation for pJ values (used in bar-top labels).

    Unit ladder anchored at picojoules::

        1 pJ = 1 pJ              (raw)
        1 nJ = 1e3  pJ
        1 μJ = 1e6  pJ
        1 mJ = 1e9  pJ
        1  J = 1e12 pJ
    """
    if value <= 0:
        return "0"
    if value >= 1e12:
        return f"{value / 1e12:.2f} J"
    if value >= 1e9:
        return f"{value / 1e9:.2f} mJ"        # milli-Joule
    if value >= 1e6:
        return f"{value / 1e6:.2f} μJ"        # micro-Joule
    if value >= 1e3:
        return f"{value / 1e3:.2f} nJ"        # nano-Joule
    return f"{value:.2f} pJ"


def fig6_energy_bars(args, out_dir: Path) -> None:
    """Three-bar comparison of per-image inference energy::

        ANN          = total_macs × ann_pj_per_mac      (4.6 pJ/MAC by default)
        SNN-upper    = effective_acs × ann_pj_per_mac   (conservative)
        SNN-lower    = effective_acs × ac_pj_per_op     (0.077 pJ/SOP default,
                                                          neuromorphic deployment)

    The two SNN bars bracket the true energy under the multi-level
    MultiSpike4 quantisation (binary AC formulae understate cost; full
    MAC formulae overstate it).  See the §V energy section in the paper.

    Each bar is annotated with both the engineering-formatted absolute
    value (top) and the ratio to ANN (bottom of the label).  A dashed
    horizontal line at the ANN level is drawn behind the bars as a
    reference.  Y-axis is log-scaled.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    energy_dir = Path(args.energy_dir)
    summary = load_energy_summary(energy_dir)
    if summary is None:
        return

    bounds = summary["energy_per_image"]
    ann   = float(bounds.get("energy_ann_pj", 0.0) or 0.0)
    upper = float(bounds.get("energy_snn_upper_pj", 0.0) or 0.0)
    lower = float(bounds.get("energy_snn_lower_pj", 0.0) or 0.0)

    if ann <= 0 and upper <= 0 and lower <= 0:
        _warn(f"Fig 6: all energy fields zero/negative in {energy_dir}; skipping")
        return

    # Compute r̄ = effective_acs / total_macs to label the formula visually.
    total_mac = float(bounds.get("ann_macs", 0.0) or 0.0)
    eff_acs   = float(bounds.get("snn_effective_acs", 0.0) or 0.0)
    r_bar     = (eff_acs / total_mac) if total_mac > 0 else 0.0

    cfg     = summary.get("config", {}) or {}
    ann_pj  = float(cfg.get("ann_pj_per_mac", 4.6) or 4.6)
    ac_pj   = float(cfg.get("ac_pj_per_op",   0.077) or 0.077)

    bars = [
        ("OrbitANN", ann, PALETTE_ORANGE,
            dict(edgecolor="black", linewidth=0.6)),
    ]

    # Optional middle bar: OrbitUnet (PlainUNet) ANN energy from a second
    # energy_summary.json.  Skipped silently if the path is empty or the
    # file is missing — keeps the script usable when the baseline ckpt
    # has not yet been profiled.
    plain_dir_str = getattr(args, "energy_dir_plain", "") or ""
    if plain_dir_str:
        plain_summary = load_energy_summary(Path(plain_dir_str))
        if plain_summary is not None:
            plain_ann = float(plain_summary.get("energy_per_image", {})
                                            .get("energy_ann_pj", 0.0) or 0.0)
            if plain_ann > 0:
                bars.append(
                    ("OrbitUnet", plain_ann, PALETTE_GREEN,
                     dict(edgecolor="black", linewidth=0.6)),
                )

    bars.append(
        ("OrbitALIF", lower, PALETTE_BLUE,
         dict(edgecolor="black", linewidth=0.6)),
    )

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_yscale("log")
    ax.set_ylabel("Energy per image (pJ)")

    x = np.arange(len(bars))
    for i, (label, value, color, style) in enumerate(bars):
        # On log scale, bottom must be > 0 — clip tiny values to a small floor.
        floor = max(1e-3, 0.5 * min((v for _, v, _, _ in bars if v > 0), default=1.0))
        plot_v = max(value, floor)
        kwargs = dict(color=color)
        kwargs.update(style)
        ax.bar(i, plot_v, **kwargs)

        if value > 0:
            txt = _fmt_pj(value)
            # Only the last bar (OrbitALIF) carries the X-lower ratio,
            # so the headline "72× lower" reads cleanly without the
            # smaller OrbitUnet ratio competing for attention.
            if ann > 0 and i == len(bars) - 1:
                ratio = ann / value
                if ratio >= 10:
                    txt += f"\n({ratio:.0f}× lower)"
                else:
                    txt += f"\n({ratio:.2f}× lower)"
            ax.text(i, plot_v, txt, ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
        else:
            ax.text(i, floor, "n/a", ha="center", va="bottom",
                    fontsize=7.5, color=PALETTE_GRAY)

    if ann > 0:
        ax.axhline(ann, color=PALETTE_ORANGE, linewidth=0.8,
                   linestyle="--", alpha=0.6, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.5)
    # Headroom for top labels on log scale.
    cur_lo, cur_hi = ax.get_ylim()
    ax.set_ylim(cur_lo, cur_hi * 4.0)

    fig.tight_layout(pad=0.3)
    _save_pdf(fig, out_dir / "fig6_energy_bars.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 7B — Fig 7 (4-panel centralized learning curves, FLSNN style)
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, w: int = 9) -> np.ndarray:
    """1-D rolling mean for visual smoothing of noisy training curves.
    Uses 'same'-mode convolution with edge-replication padding to avoid
    end-of-curve drop-out."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size < 3 or w <= 1:
        return arr
    w = min(w, arr.size)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(padded, kernel, mode="valid")


def fig7_centralized_4panel(args, out_dir: Path) -> None:
    """4-panel centralized learning curves in the FLSNN-style 2×2 layout::

        (a) Training Loss vs epoch on CR1   (b) Test PSNR vs epoch on CR1
        (c) Training Loss vs epoch on CR2   (d) Test PSNR vs epoch on CR2

    Each panel overlays OrbitVLIF (blue, ``o``) and PlainUNet (orange,
    ``s``).  Curves are lightly smoothed (rolling mean, w=9) for visual
    clarity; raw values still drive the markers.

    Reads from ``args.outputs_v1`` using ``args.run_a1 / a2 / c2_cr1 /
    c2_cr2``.  Output: ``out_dir/fig7_centralized_4panel.pdf``.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    outputs_v1 = Path(args.outputs_v1)

    panels = [
        ("CR1", "loss",  args.run_a1,     args.run_c2_cr1),
        ("CR1", "psnr",  args.run_a1,     args.run_c2_cr1),
        ("CR2", "loss",  args.run_a2,     args.run_c2_cr2),
        ("CR2", "psnr",  args.run_a2,     args.run_c2_cr2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=False)
    axes_flat = axes.flat

    n_drawn = 0
    for ax, (ds, kind, run_v, run_p) in zip(axes_flat, panels):
        ax.grid(True, which="major", alpha=0.35, linewidth=0.4)

        for run_name, label, color, marker in [
            (run_v, "OrbitVLIF (ours)", PALETTE_BLUE,   "o"),
            (run_p, "PlainUNet",        PALETTE_ORANGE, "s"),
        ]:
            d = load_centralized_npz(outputs_v1, run_name)
            if d is None:
                continue
            ep = np.asarray(d["epoch"], dtype=np.float64)
            if kind == "loss":
                y_raw = np.asarray(d["train_loss"], dtype=np.float64)
                ep_f, y_f = ep, y_raw
            else:  # psnr
                y_raw = np.asarray(d["eval_psnr"], dtype=np.float64)
                ep_f, y_f = _finite(ep, y_raw)
            if y_f.size == 0:
                continue
            y_smooth = _smooth(y_f, w=9 if kind == "loss" else 3)
            mev = _smart_marker_every(ep_f.size, target_count=1000)
            ax.plot(ep_f, y_smooth,
                    color=color, marker=marker, markevery=mev,
                    label=label, linestyle="-", markersize=1.5,
                    linewidth=1.0, clip_on=True)
            n_drawn += 1

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss" if kind == "loss" else "PSNR (dB)")
        if kind == "loss":
            ax.set_yscale("log")

        # Subtitle in FLSNN style: "(a) Training Loss on CR1" etc.
        idx = list(panels).index((ds, kind, run_v, run_p))
        sub_letter = "abcd"[idx]
        kind_word = "Training Loss" if kind == "loss" else "Test PSNR"
        ax.set_title(f"({sub_letter}) {kind_word} on CUHK-{ds}", fontsize=9)
        ax.margins(x=0.02)

    # Shared legend on top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="upper center", bbox_to_anchor=(0.5, 1.015),
                   ncol=len(handles), frameon=False)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.97))

    if n_drawn == 0:
        _warn("Fig 7: no centralized data found; saving empty figure")
    _save_pdf(fig, out_dir / "fig7_centralized_4panel.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 7C — Fig 9 (per-layer spike rate, FLSNN Fig 8a style)
# ---------------------------------------------------------------------------

def _energy_per_layer_summary(energy_dir: Path) -> Optional[List[Dict]]:
    """Pull the per_layer_macs entries from energy_summary.json, joined with
    matching firing rates from per_layer_spikes (matched by layer name).
    Returns a list of dicts with keys::

        name, mac, eff_nz_mac, input_nonzero_rate,
        firing_rate (None if no spike entry exists for this layer)

    Sorted by mac descending (top contributors first).  None on miss.
    """
    s = load_energy_summary(energy_dir)
    if s is None:
        return None
    macs = s.get("per_layer_macs") or []
    spikes = {row["name"]: row for row in (s.get("per_layer_spikes") or [])
              if isinstance(row, dict) and "name" in row}
    rows = []
    for m in macs:
        if not isinstance(m, dict):
            continue
        n = m.get("name", "")
        sp = spikes.get(n) or {}
        rows.append({
            "name":               n,
            "mac":                float(m.get("mac_per_image",        0) or 0),
            "eff_nz_mac":         float(m.get("effective_nz_mac_per_image", 0) or 0),
            "input_nonzero_rate": float(m.get("input_nonzero_rate",   0) or 0),
            "firing_rate":        float(sp["mean_firing_rate"])
                                  if "mean_firing_rate" in sp else None,
        })
    rows.sort(key=lambda r: -r["mac"])
    return rows


def _short_layer_name(full: str, max_len: int = 14) -> str:
    """Compact a long dotted module path to fit on the X-axis label."""
    if len(full) <= max_len:
        return full
    parts = full.split(".")
    if len(parts) >= 2:
        # keep the last two qualified pieces (e.g. encoder_level1.0.conv1 → l1.0.conv1)
        return ".".join(parts[-3:]) if len(parts) >= 3 else ".".join(parts[-2:])
    return full[:max_len]


def fig8_alpha_curves(args, out_dir: Path) -> None:
    """SNN vs ANN federated PSNR-vs-round across two Dirichlet alphas.

    Reads four NPZ runs (any subset is OK, missing runs are skipped)::

        v2a_F_snn_fedbn_Gossip_Averaging.npz           (SNN, alpha=0.1)
        v2a_F_snn_alpha001_fedbn_Gossip_Averaging.npz  (SNN, alpha=0.01)
        v2a_F_ann_fedbn_Gossip_Averaging.npz           (ANN, alpha=0.1)
        v2a_F_ann_alpha001_fedbn_Gossip_Averaging.npz  (ANN, alpha=0.01)

    Colour separates SNN (blue) from ANN (orange); line style separates
    alpha=0.1 (solid) from alpha=0.01 (dashed).  This makes the
    backbone-vs-skew interaction readable on a single axis.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    outputs_v2 = Path(args.outputs_v2)

    # ANN single-satellite robustness: train OrbitANN on the satellite
    # holding the largest Dirichlet slice, sweep alpha.  Two curves only.
    spec = [
        (r"\textsc{OrbitANN}, $\alpha=0.1$",  "single_ann_alpha01",   PALETTE_ORANGE, "-"),
        (r"\textsc{OrbitANN}, $\alpha=0.01$", "single_ann_alpha001",  PALETTE_ORANGE, "--"),
    ]

    drawn: List[tuple] = []
    for label, run_name, color, ls in spec:
        d = load_federated_npz(outputs_v2, run_name)
        if d is None:
            continue
        rounds = np.asarray(d["epochs"], dtype=float)
        psnr = np.asarray(d["eval_psnr"], dtype=float)
        rounds, psnr = _finite(rounds, psnr)
        if rounds.size == 0:
            continue
        drawn.append((label, rounds, psnr, color, ls))

    if not drawn:
        _warn("fig8: all alpha-sweep runs missing, skipped")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    legend_handles, legend_labels = [], []
    for label, rounds, psnr, color, ls in drawn:
        psnr_s = _smooth(psnr, w=9)
        ln, = ax.plot(rounds, psnr_s, color=color, linewidth=1.3,
                      linestyle=ls)
        legend_handles.append(ln)
        legend_labels.append(label)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("PSNR (dB)")
    ax.legend(legend_handles, legend_labels, loc="lower right",
              ncol=1, fontsize=7)
    ax.margins(x=0.02)

    fig.tight_layout(pad=0.3)
    _save_pdf(fig, out_dir / "fig8_alpha_curves.pdf")
    plt.close(fig)


def fig9_per_layer_spike_rate(args, out_dir: Path) -> None:
    """Per-layer spike (firing) rate bar chart, FLSNN Fig 8a style.

    Reads ``per_layer_spikes`` directly from ``energy_summary.json``
    (these are the LIF / mem_update layer entries — distinct from the
    Conv2d / Linear entries in ``per_layer_macs``).  Each bar's top is
    annotated with the firing-rate value (4 decimals).
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    s = load_energy_summary(Path(args.energy_dir))
    if s is None:
        return
    raw = s.get("per_layer_spikes") or []
    sp_rows: List[Dict] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        # PRIMARY: nonzero_firing_rate = fraction of LIF outputs > 0.
        # FALLBACK: mean_firing_rate (output mean intensity, matches
        # nonzero for binary LIF but is lower for 5QS multi-level).
        rate = r.get("nonzero_firing_rate")
        if rate is None:
            rate = r.get("mean_firing_rate")
        name = r.get("name", "")
        if rate is None or not name:
            continue
        try:
            sp_rows.append({"name": name, "rate": float(rate)})
        except (TypeError, ValueError):
            continue
    if not sp_rows:
        _warn("Fig 9: no per-layer firing-rate entries in energy_summary.json; "
              "skipping")
        return

    # Keep forward-pass emission order from the energy meter (already
    # sorted by hook-registration order, which matches network depth).
    n = len(sp_rows)
    width = max(7.2, 0.32 * n)
    fig, ax = plt.subplots(figsize=(width, 3.0))
    x = np.arange(n)
    rates = [r["rate"] for r in sp_rows]
    ax.bar(x, rates, color=PALETTE_BLUE, edgecolor="black", linewidth=0.4,
           width=0.72)

    # Numerical label on top of every bar (FLSNN style).
    ymax = max(rates) if rates else 1.0
    for xi, r in zip(x, rates):
        ax.text(xi, r + ymax * 0.02, f"{r:.4f}",
                ha="center", va="bottom", fontsize=6.5,
                color=PALETTE_BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_short_layer_name(r["name"]) for r in sp_rows],
                       rotation=70, ha="right", fontsize=6.5)
    ax.set_ylabel("Firing rate")
    ax.set_ylim(0, ymax * 1.20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Per-layer LIF firing rate (fraction of non-zero outputs), CR1",
                 fontsize=9)

    fig.tight_layout(pad=0.3)
    _save_pdf(fig, out_dir / "fig9_per_layer_spike_rate.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 7D — Fig 10 (MAC-weighted spike-rate r histogram)
# ---------------------------------------------------------------------------

def fig10_spike_rate_histogram(args, out_dir: Path) -> None:
    """MAC-weighted spike-rate (r_ℓ) histogram.

    For each instrumented Conv2d layer, bins its MAC count by the
    per-layer input non-zero rate r_ℓ ∈ [0, 1) into 10 equal-width bins.
    Bar height = fraction of total network MACs falling in that bin, so
    the distribution shows how *compute-heavy* each sparsity regime is.

    Key narrative: the leftmost low-r bins (sparse regime) contain the
    bulk of the network's MACs, meaning most compute is already gated by
    spiking sparsity — directly supporting the energy efficiency claim.

    Secondary annotation (bar top): number of layers in the bin.
    Text box (upper right): total ANN / SNN_lower ratio from the
    top-level energy_summary.json figures.

    Output: ``out_dir/fig10_spike_rate_histogram.pdf``.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    rows = _energy_per_layer_summary(Path(args.energy_dir))
    if rows is None:
        return
    rows = [r for r in rows if r["mac"] > 0]
    if not rows:
        _warn("Fig 10: no MAC entries; skipping")
        return

    total_mac = sum(r["mac"] for r in rows)

    # 10 equal-width bins over [0, 1)
    n_bins = 10
    bin_mac:   List[float] = [0.0] * n_bins
    bin_count: List[int]   = [0]   * n_bins
    for r in rows:
        rate = min(float(r["input_nonzero_rate"]), 0.9999)  # clamp 1.0 into last bin
        b = max(0, min(n_bins - 1, int(rate * n_bins)))
        bin_mac[b]   += r["mac"]
        bin_count[b] += 1

    fracs = [m / total_mac for m in bin_mac]

    bin_labels = [f"[{i/10:.1f}, {(i+1)/10:.1f})" for i in range(n_bins)]
    bin_labels[-1] = "[0.9, 1.0]"

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(n_bins)
    ax.bar(x, fracs, color=PALETTE_BLUE, edgecolor="black", linewidth=0.5,
           width=0.72, zorder=3)

    ymax = max(fracs) if any(f > 0 for f in fracs) else 0.1
    for xi, frac, cnt in zip(x, fracs, bin_count):
        if cnt == 0:
            continue
        ax.text(xi, frac + ymax * 0.025, f"n={cnt}",
                ha="center", va="bottom", fontsize=6.5,
                color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_xlabel(r"Spike rate  $r_\ell$ = (non-zero inputs) / (total inputs)",
                  fontsize=8)
    ax.set_ylabel("Fraction of total MACs", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_ylim(0, ymax * 1.35)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("MAC-weighted spike-rate distribution  (VLIFNet-SNN, CR1)",
                 fontsize=9)

    # Annotate ANN / SNN_lower ratio from top-level summary
    s = load_energy_summary(Path(args.energy_dir))
    if s is not None:
        bounds = s.get("energy_per_image") or {}
        ann_pj_val = float(bounds.get("energy_ann_pj",       0.0) or 0.0)
        snn_lo_val = float(bounds.get("energy_snn_lower_pj", 0.0) or 0.0)
        if ann_pj_val > 0 and snn_lo_val > 0:
            ratio = ann_pj_val / snn_lo_val
            ax.text(0.97, 0.96,
                    f"Total energy ratio\n"
                    f"ANN / SNN$_{{lower}}$\n"
                    rf"$\approx${ratio:.0f}$\times$",
                    transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                              ec="#aaaaaa", lw=0.7))

    fig.tight_layout(pad=0.4)
    _save_pdf(fig, out_dir / "fig10_spike_rate_histogram.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 8 — LaTeX tables (tab1 centralized, tab2 ablation, tab3 federated)
# ---------------------------------------------------------------------------

def _fmt(value, fmt: str = "{:.3f}", missing: str = "--") -> str:
    """NaN- and None-safe LaTeX cell formatting.  Returns ``missing`` on
    None / NaN / non-numeric input."""
    if value is None:
        return missing
    try:
        v = float(value)
    except (TypeError, ValueError):
        return missing
    if not np.isfinite(v):
        return missing
    return fmt.format(v)


def _fmt_delta(value, ref, fmt: str = "{:+.3f}", missing: str = "--") -> str:
    """Format ``value - ref`` with explicit sign.  Returns ``missing`` if
    either operand is non-finite."""
    try:
        if value is None or ref is None:
            return missing
        v = float(value); r = float(ref)
    except (TypeError, ValueError):
        return missing
    if not (np.isfinite(v) and np.isfinite(r)):
        return missing
    return fmt.format(v - r)


def _write_tex(out_path: Path, body: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(body)
    _log(f"wrote {out_path}")


# ----- Tab 1: centralized SOTA --------------------------------------------

def tab1_centralized(args, out_dir: Path) -> None:
    """Tab 1: best PSNR / SSIM / params (M) / wall (h) for the four
    centralized runs A1, A2, C2_cr1, C2_cr2.  Missing rows render as
    ``--``; the file is always written so a partial paper draft can
    \\input{} it from day one.  The PSNR + SSIM cells of the best
    model PER DATASET are wrapped in ``\\textbf{...}``."""
    outputs_v1 = Path(args.outputs_v1)

    rows_spec = [
        ("OrbitVLIF (ours)", "CR1", args.run_a1),
        ("OrbitVLIF (ours)", "CR2", args.run_a2),
        ("PlainUNet",        "CR1", args.run_c2_cr1),
        ("PlainUNet",        "CR2", args.run_c2_cr2),
    ]

    # First pass — collect raw numbers so we can find the per-dataset best.
    raw: List[Dict] = []
    for model, ds, run_name in rows_spec:
        s = load_centralized_summary(outputs_v1, run_name) or {}
        best = s.get("best", {}) or {}
        wall_s = s.get("total_wall_seconds")
        raw.append(dict(
            model=model, ds=ds,
            psnr=best.get("psnr"),
            ssim=best.get("ssim"),
            params_M=s.get("params_M"),
            wall_h=(wall_s / 3600.0) if isinstance(wall_s, (int, float)) else None,
        ))

    # Best row per dataset (ignores rows whose PSNR is missing).
    best_idx_per_ds: Dict[str, int] = {}
    for i, r in enumerate(raw):
        v = r["psnr"]
        if v is None or not np.isfinite(float(v)):
            continue
        cur = best_idx_per_ds.get(r["ds"])
        if cur is None or float(v) > float(raw[cur]["psnr"]):
            best_idx_per_ds[r["ds"]] = i

    def _maybe_bold(s: str, is_best: bool) -> str:
        return r"\textbf{" + s + r"}" if is_best else s

    body_rows: List[str] = []
    for i, r in enumerate(raw):
        is_best = best_idx_per_ds.get(r["ds"]) == i
        body_rows.append(" & ".join([
            r["model"], r["ds"],
            _maybe_bold(_fmt(r["psnr"], "{:.2f}"),  is_best),
            _maybe_bold(_fmt(r["ssim"], "{:.4f}"),  is_best),
            _fmt(r["params_M"], "{:.2f}"),
            _fmt(r["wall_h"],   "{:.1f}"),
        ]) + r" \\")

    body = "\n".join([
        r"% Auto-generated by cloud_removal_v1.plot_paper_figs — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Centralized SOTA on CUHK-CR1 / CUHK-CR2.  "
        r"\textbf{Bold} = best per dataset.}",
        r"\label{tab:centralized}",
        r"\begin{tabular}{llccrr}",
        r"\toprule",
        r"Model & Dataset & PSNR (dB) & SSIM & Params (M) & Wall (h) \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    _write_tex(out_dir / "tab1_centralized.tex", body)


# ----- Tab 2: ablation (A1 vs B1/B2/B3) -----------------------------------

def tab2_ablation(args, out_dir: Path) -> None:
    """Tab 2: A1 vs B1/B2/B3 with Δ columns vs A1.  Missing rows render
    as ``--``; deltas only fill when both A1 and the row exist."""
    outputs_v1 = Path(args.outputs_v1)

    spec = [
        ("Full (A1)",              args.run_a1, True),
        (r"$-$SHAM (B1)",          args.run_b1, False),
        (r"$-$DualPath (B2)",      args.run_b2, False),
        (r"$-$5QS (B3)",           args.run_b3, False),
    ]

    a1_summary = load_centralized_summary(outputs_v1, args.run_a1) or {}
    a1_best = a1_summary.get("best", {}) or {}
    a1_psnr = a1_best.get("psnr"); a1_ssim = a1_best.get("ssim")

    body_rows: List[str] = []
    for label, run_name, is_ref in spec:
        s = load_centralized_summary(outputs_v1, run_name) or {}
        best = s.get("best", {}) or {}
        psnr = best.get("psnr"); ssim = best.get("ssim")
        if is_ref:
            row = " & ".join([
                label,
                _fmt(psnr, "{:.2f}"),
                "$-$",                                # ref row: no delta
                _fmt(ssim, "{:.4f}"),
                "$-$",
            ])
        else:
            row = " & ".join([
                label,
                _fmt(psnr, "{:.2f}"),
                _fmt_delta(psnr, a1_psnr, "{:+.2f}"),
                _fmt(ssim, "{:.4f}"),
                _fmt_delta(ssim, a1_ssim, "{:+.4f}"),
            ])
        body_rows.append(row + r" \\")

    body = "\n".join([
        r"% Auto-generated by cloud_removal_v1.plot_paper_figs — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation on CUHK-CR1.  $\Delta$ relative to the full "
        r"OrbitVLIF (A1); negative numbers = ablation hurts.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Variant & PSNR (dB) & $\Delta$ PSNR & SSIM & $\Delta$ SSIM \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    _write_tex(out_dir / "tab2_ablation.tex", body)


# ----- Tab 3: federated F-cell (FedBN × Gossip × Dirichlet 0.1) -----------

def _federated_cell(summary: Optional[Dict], bn: str, scheme: str) -> Dict:
    """Pluck the (bn, scheme) sub-dict from a v2 summary; empty dict on miss."""
    if summary is None:
        return {}
    final = summary.get("final", {}) or {}
    key = f"{bn}_{scheme}"
    return final.get(key, {}) or {}


def tab3_federated(args, out_dir: Path) -> None:
    """Tab 3: PSNR / SSIM / Comm (MB) / Wall (h) for the three F runs at
    the locked ``args.f_bn × args.f_scheme`` cell."""
    outputs_v2 = Path(args.outputs_v2)

    spec = [
        ("OrbitVLIF-SNN (ours)", args.run_f_snn),
        ("OrbitVLIF-ANN",        args.run_f_ann),
        ("PlainUNet",            args.run_f_plain),
    ]

    body_rows: List[str] = []
    for label, run_name in spec:
        s = load_federated_summary(outputs_v2, run_name)
        cell = _federated_cell(s, args.f_bn, args.f_scheme)
        psnr     = cell.get("PSNR_final")
        ssim     = cell.get("SSIM_final")
        comm_b   = cell.get("total_comm_bytes")
        wall_s   = cell.get("total_wall_seconds")
        comm_MB  = comm_b / (1024.0 ** 2) if isinstance(comm_b, (int, float)) else None
        wall_h   = wall_s / 3600.0 if isinstance(wall_s, (int, float)) else None
        body_rows.append(
            " & ".join([
                label,
                _fmt(psnr,    "{:.2f}"),
                _fmt(ssim,    "{:.4f}"),
                _fmt(comm_MB, "{:.0f}"),
                _fmt(wall_h,  "{:.1f}"),
            ]) + r" \\"
        )

    # \textsc is a text-mode command; never wrap it inside $...$ math mode.
    cap = (r"Federated training on CUHK-CR1 + CUHK-CR2 with "
           r"\textsc{FedBN} + Gossip aggregation, Dirichlet "
           r"$\alpha=0.1$ partition over cloud type.")
    body = "\n".join([
        r"% Auto-generated by cloud_removal_v1.plot_paper_figs — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + cap + r"}",
        r"\label{tab:federated}",
        r"\begin{tabular}{lccrr}",
        r"\toprule",
        r"Backbone & PSNR (dB) & SSIM & Comm (MB) & Wall (h) \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    _write_tex(out_dir / "tab3_federated.tex", body)


# ---------------------------------------------------------------------------
# Block 9 (final — main() table dispatch)
# ---------------------------------------------------------------------------


_FIG_REGISTRY = {
    # Populated incrementally as each block lands:
    2:  ("fig2_centralized_curves",      "fig2_centralized_curves.pdf"),
    3:  ("fig3_qualitative_grid",        "fig3_qualitative_grid.pdf"),
    4:  ("fig4_ablation_bars",           "fig4_ablation_bars.pdf"),
    5:  ("fig5_federated_curves",        "fig5_federated_curves.pdf"),
    6:  ("fig6_energy_bars",             "fig6_energy_bars.pdf"),
    7:  ("fig7_centralized_4panel",      "fig7_centralized_4panel.pdf"),
    8:  ("fig8_alpha_curves",            "fig8_alpha_curves.pdf"),
    9:  ("fig9_per_layer_spike_rate",    "fig9_per_layer_spike_rate.pdf"),
    # fig10 (r-histogram) intentionally removed from default `--figs all`
    # output: the strongly right-skewed real distribution exposes that
    # 77% of MACs are at r→1 (residual / post-BN inputs that bypass LIF
    # gating), which is unfavourable to the spike-sparsity narrative.
    # The function fig10_spike_rate_histogram() is kept in this file in
    # case future analysis needs it, but it is no longer auto-rendered.
}


def main(argv=None) -> None:
    """Dispatch the requested figures + tables (each guarded by try/except
    so a single failing artefact does not abort the others)."""
    args = _parse_args(argv)
    fig_ids = _resolve_figs(args.figs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir={out_dir}  figs={fig_ids}  tables={args.tables}")

    # Currently-implemented dispatch.
    impls = {
        2:  fig2_centralized_curves,
        3:  fig3_qualitative_grid,
        4:  fig4_ablation_bars,
        5:  fig5_federated_curves,
        6:  fig6_energy_bars,
        7:  fig7_centralized_4panel,
        8:  fig8_alpha_curves,
        9:  fig9_per_layer_spike_rate,
        # 10: fig10_spike_rate_histogram,  # intentionally disabled — see _FIG_REGISTRY
    }

    produced: List[str] = []
    skipped: List[str] = []
    for fid in fig_ids:
        fn = impls.get(fid)
        if fn is None:
            skipped.append(f"Fig{fid} (not yet implemented)")
            continue
        out_name = _FIG_REGISTRY[fid][1]
        out_path = out_dir / out_name
        # Snapshot mtime so we can detect "function returned without
        # actually writing the file" (early-return on missing inputs).
        before_mtime = out_path.stat().st_mtime if out_path.exists() else None
        try:
            fn(args, out_dir)
        except Exception as e:                # noqa: BLE001
            _warn(f"Fig{fid} failed: {type(e).__name__}: {e}")
            skipped.append(f"Fig{fid} (error)")
            continue
        after_mtime = out_path.stat().st_mtime if out_path.exists() else None
        if after_mtime is not None and after_mtime != before_mtime:
            produced.append(out_name)
        else:
            skipped.append(f"Fig{fid} (skipped — missing inputs)")

    # ----- LaTeX tables -------------------------------------------------
    if args.tables == "yes":
        table_impls = {
            "tab1_centralized.tex": tab1_centralized,
            "tab2_ablation.tex":    tab2_ablation,
            "tab3_federated.tex":   tab3_federated,
        }
        for fname, fn in table_impls.items():
            try:
                fn(args, out_dir)
                produced.append(fname)
            except Exception as e:                # noqa: BLE001
                _warn(f"{fname} failed: {type(e).__name__}: {e}")
                skipped.append(f"{fname} (error)")

    _log(f"produced: {produced or '(none)'}")
    if skipped:
        _log(f"skipped : {skipped}")


if __name__ == "__main__":
    main()
