"""
v2 Path-A configuration.

Default run = **6 runs** sequentially:
    for bn_mode in (fedavg, fedbn):
        for scheme in (RelaySum, Gossip, All-Reduce):
            train 30 global epochs on CUHK-CR1 + CUHK-CR2
            with Dirichlet(alpha=0.1) non-IID partition over cloud type
            and synchronized geometric augmentation.

Overridable via `parse_v2a_cli()`; intended to be called from
`cloud_removal_v2.run_smoke`.

See also: cloud_removal_v2/docs/v2a_setup.md.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional


V2A_DEFAULTS: Dict[str, Any] = {
    # ---- Run identity ----
    "run_name":           "v2a",
    "seed":               1234,
    "deterministic":      False,    # opt-in; ~10-20% slower but bit-exact
    "device":             "cuda:0",
    "output_dir":         "./Outputs_v2",
    "tensorboard_dir":    "./Outputs_v2/tb",
    "ckpt_dir":           "./Outputs_v2/ckpts",

    # ---- Dataset roots ----
    # Each entry = (root, label, name).  label is the source-id used by
    # the Dirichlet partition (0 = thin, 1 = thick by convention).
    "sources": [
        {"root": "./data/CUHK-CR1", "label": 0, "name": "CUHK-CR1"},
        {"root": "./data/CUHK-CR2", "label": 1, "name": "CUHK-CR2"},
    ],
    "train_split":        "train",
    "test_split":         "test",

    # ---- Data pipeline ----
    "patch_size":         64,
    "train_batch_size":   4,
    "test_batch_size":    1,
    "num_workers":        0,      # 50 sats × workers ⇒ keep at 0
    # Augmentation probabilities (applied synchronously to cloudy/clear)
    "aug_hflip_p":        0.5,
    "aug_vflip_p":        0.5,
    "aug_rot90_p":        0.25,   # prob of +90°
    "aug_rot270_p":       0.25,   # prob of −90°
    # NO colour jitter — remote-sensing absolute colour has physical meaning
    "augment":            True,

    # ---- Constellation ----
    "num_planes":         5,
    "sats_per_plane":     10,
    "partition_mode":     "dirichlet_source",    # key change vs v1
    "partition_alpha":    0.1,                    # aggressive non-IID
    "partition_seed":     0,
    "min_samples_per_client": 5,   # enforce ≥5 per client after Dirichlet

    # ---- Model (same as v1) ----
    "vlif_dim":           24,
    "en_blocks":          [2, 2, 4, 4],
    "de_blocks":          [2, 2, 2, 2],
    "T":                  4,
    "vlif_backend":       "torch",
    "use_refinement":     False,
    # Phase-1 P1.2: BN variant for SC-16d ablation.
    # "tdbn" = threshold-dependent BN (default, FLSNN / Zheng 2021).
    # "bn2d" = standard nn.BatchNorm2d (Claim C16 binary ablation).
    "bn_variant":         "tdbn",
    # Phase-1 P1.3: backbone variant for FLSNN §VI-B ANN-vs-SNN comparison.
    # "snn" = LIFNode + MultiSpike4 (default, all results so far).
    # "ann" = ReLU replacing every spike activation (same architecture).
    "backbone":           "snn",

    # ---- Optimisation (same as v1) ----
    "lr":                 1e-3,
    "min_lr":             1e-7,
    "wd":                 0.0,
    "warmup_epochs":      3,
    "clip_grad":          1.0,
    "use_amp":            False,

    # ---- Training schedule ----
    "num_epoch":          30,       # bumped from 10 → 30
    "intra_plane_iters":  2,
    "local_iters":        2,

    # ---- Loss ----
    "ssim_weight":        0.1,
    "charbonnier_eps":    1e-3,

    # ---- Aggregation ----
    # v2-A sweeps this flag in the runner; the config value is the
    # *default* for one-off invocations.
    "bn_local":           False,

    # ---- Evaluation ----
    "eval_every":         5,        # every 5 global epochs → 6 evals / run
    "eval_mode":          "center_patch",
    "eval_patch_size":    64,
    "sliding_window":     64,
    "sliding_stride":     32,

    # ---- Visualization ----
    "viz_n_samples":      6,
    "viz_sample_seed":    42,

    # ---- Inline logging (Phase-1 P1.1: 70-ep diagnostic hooks) ----
    # BnDriftLogger: SC-16a/b/c support (||lambda-1||, ||beta||, per-plane
    # variance). Negligible compute (<10 ms / eval).
    "log_drift":          True,
    # CosineSimLogger: Seo24 Fig 13 proxy for cross-plane heterogeneity.
    # Records pairwise cosine(delta_plane_i, delta_plane_j) at eval_every
    # boundaries. ~200 ms / eval.
    "log_cosine_sim":     True,
}


def build_v2a_args(**overrides: Any) -> argparse.Namespace:
    cfg = dict(V2A_DEFAULTS)
    cfg.update(overrides)
    ns = argparse.Namespace(**cfg)
    _validate(ns)
    return ns


def _validate(ns: argparse.Namespace) -> None:
    assert ns.num_planes > 0 and ns.sats_per_plane > 0
    assert ns.T == 4, "upstream VLIFNet hard-codes T=4; see cloud_removal_v1"
    assert ns.train_batch_size >= 1
    assert ns.patch_size >= 16
    assert ns.eval_patch_size >= 16
    assert ns.vlif_backend in ("torch", "cupy")
    assert ns.bn_variant in ("tdbn", "bn2d"), \
        f"bn_variant must be 'tdbn' or 'bn2d', got {ns.bn_variant!r}"
    assert ns.backbone in ("snn", "ann", "plain"), \
        f"backbone must be 'snn', 'ann' or 'plain', got {ns.backbone!r}"
    if ns.backbone == "plain" and ns.bn_variant == "tdbn":
        raise ValueError(
            "backbone='plain' is incompatible with bn_variant='tdbn' "
            "(TDBN's gamma_init = alpha*V_th assumes spike thresholds; "
            "PlainUNet has none).  Use --bn_variant bn2d.")
    assert ns.partition_mode in ("iid", "dirichlet_source", "dirichlet_cluster")
    assert ns.eval_mode in ("center_patch", "fullimage", "sliding")
    assert 0.0 < ns.lr and ns.min_lr >= 0
    assert 0 <= ns.ssim_weight <= 1
    assert ns.num_epoch >= 1
    assert ns.partition_alpha > 0
    assert isinstance(ns.sources, list) and len(ns.sources) >= 1
    for s in ns.sources:
        assert "root" in s and "label" in s, f"malformed source entry: {s}"
    assert 0.0 <= ns.aug_hflip_p <= 1.0
    assert 0.0 <= ns.aug_vflip_p <= 1.0
    assert 0.0 <= ns.aug_rot90_p <= 1.0
    assert 0.0 <= ns.aug_rot270_p <= 1.0
    assert ns.aug_rot90_p + ns.aug_rot270_p <= 1.0
    # B-BN-1 INVARIANT: v1 task.py's DataLoader uses drop_last=True to shield
    # BatchNorm from a trailing single-sample batch (which drives running_var
    # → 0 and amplifies subsequent inputs ~316×).  That requires every client
    # to have at least one full batch — i.e. min_samples_per_client >=
    # train_batch_size; otherwise local_training returns NaN and pollutes
    # round_losses.  Fail loudly at config time, not 2 hours into a sweep.
    assert ns.min_samples_per_client >= ns.train_batch_size, (
        f"min_samples_per_client ({ns.min_samples_per_client}) must be >= "
        f"train_batch_size ({ns.train_batch_size}) so every client has at "
        f"least one full batch under drop_last=True.")


def parse_v2a_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI wrapper — exposes the knobs most likely to need overriding."""
    p = argparse.ArgumentParser(description="v2-A cloud-removal smoke")
    p.add_argument("--data_root", type=str, default=None,
                   help="When set, builds a 2-source config assuming "
                        "<data_root>/CUHK-CR1 and <data_root>/CUHK-CR2 "
                        "both exist.  Overrides the 'sources' default.")
    p.add_argument("--source_root_1", type=str, default=None)
    p.add_argument("--source_root_2", type=str, default=None)
    p.add_argument("--output_dir",   type=str, default=V2A_DEFAULTS["output_dir"])
    p.add_argument("--ckpt_dir",     type=str, default=V2A_DEFAULTS["ckpt_dir"])
    p.add_argument("--run_name",     type=str, default=V2A_DEFAULTS["run_name"])
    p.add_argument("--num_epoch",    type=int, default=V2A_DEFAULTS["num_epoch"])
    p.add_argument("--num_planes",   type=int, default=V2A_DEFAULTS["num_planes"])
    p.add_argument("--sats_per_plane", type=int, default=V2A_DEFAULTS["sats_per_plane"])
    p.add_argument("--patch_size",   type=int, default=V2A_DEFAULTS["patch_size"])
    p.add_argument("--train_batch_size", type=int, default=V2A_DEFAULTS["train_batch_size"])
    p.add_argument("--num_workers",  type=int, default=V2A_DEFAULTS["num_workers"])
    p.add_argument("--vlif_dim",     type=int, default=V2A_DEFAULTS["vlif_dim"])
    p.add_argument("--vlif_backend", type=str, default=V2A_DEFAULTS["vlif_backend"],
                   choices=["torch", "cupy"])
    p.add_argument("--bn_variant", type=str, default=V2A_DEFAULTS["bn_variant"],
                   choices=["tdbn", "bn2d"],
                   help="'tdbn' = threshold-dependent BN (default, FLSNN setup). "
                        "'bn2d' = standard nn.BatchNorm2d (SC-16d ablation).")
    p.add_argument("--backbone", type=str, default=V2A_DEFAULTS["backbone"],
                   choices=["snn", "ann", "plain"],
                   help="'snn' = LIFNode + MultiSpike4 (default). "
                        "'ann' = ReLU replaces every spike activation "
                        "(FLSNN §VI-B ANN-vs-SNN comparison). "
                        "'plain' = PlainUNet baseline (no spike, no FSTA, "
                        "no frequency module — F_plain experiment; "
                        "must be paired with --bn_variant bn2d).")
    p.add_argument("--lr",           type=float, default=V2A_DEFAULTS["lr"])
    p.add_argument("--partition_alpha", type=float, default=V2A_DEFAULTS["partition_alpha"])
    p.add_argument("--partition_seed",  type=int,   default=V2A_DEFAULTS["partition_seed"])
    p.add_argument("--no_augment",   action="store_true")
    p.add_argument("--eval_mode",    type=str, default=V2A_DEFAULTS["eval_mode"],
                   choices=["center_patch", "fullimage", "sliding"])
    p.add_argument("--eval_every",   type=int, default=V2A_DEFAULTS["eval_every"])
    p.add_argument("--seed",         type=int, default=V2A_DEFAULTS["seed"])
    p.add_argument("--device",       type=str, default=V2A_DEFAULTS["device"])
    p.add_argument("--deterministic", action="store_true",
                   help="Force cuDNN deterministic algorithms (slower ~10-20%, "
                        "bit-exact across runs).  Also enables "
                        "torch.use_deterministic_algorithms(warn_only=True).")
    # Sweep control: a single subset can be picked for re-running only one cell
    p.add_argument("--only_bn",      type=str, default=None,
                   choices=["fedavg", "fedbn"],
                   help="If set, only runs the given BN mode.")
    p.add_argument("--only_scheme",  type=str, default=None,
                   help="If set, only runs the given aggregation scheme tag.")
    # Phase-1 P1.1: inline logging toggles. Default ON for new 70-ep runs.
    p.add_argument("--no_log_drift",       action="store_true",
                   help="Disable per-epoch BnDriftLogger (SC-16a/b/c).")
    p.add_argument("--no_log_cosine_sim",  action="store_true",
                   help="Disable CosineSimLogger (cross-plane grad proxy).")
    parsed = p.parse_args(argv)

    overrides: Dict[str, Any] = {k: v for k, v in vars(parsed).items()
                                 if k not in ("data_root", "source_root_1",
                                              "source_root_2", "no_augment",
                                              "only_bn", "only_scheme",
                                              "no_log_drift",
                                              "no_log_cosine_sim")}
    if parsed.no_augment:
        overrides["augment"] = False
    if parsed.no_log_drift:
        overrides["log_drift"] = False
    if parsed.no_log_cosine_sim:
        overrides["log_cosine_sim"] = False

    # Resolve data roots
    if parsed.source_root_1 or parsed.source_root_2:
        srcs = []
        if parsed.source_root_1:
            srcs.append({"root": parsed.source_root_1, "label": 0, "name": "src0"})
        if parsed.source_root_2:
            srcs.append({"root": parsed.source_root_2, "label": 1, "name": "src1"})
        overrides["sources"] = srcs
    elif parsed.data_root:
        import os
        srcs = []
        for i, sub in enumerate(("CUHK-CR1", "CUHK-CR2")):
            root = os.path.join(parsed.data_root, sub)
            if os.path.isdir(root):
                srcs.append({"root": root, "label": i, "name": sub})
        if not srcs:
            # Fall back to the flat root itself
            srcs = [{"root": parsed.data_root, "label": 0, "name": "only"}]
        overrides["sources"] = srcs

    ns = build_v2a_args(**overrides)
    # Stash the sweep filters on the namespace (consumed by run_smoke)
    ns.only_bn = parsed.only_bn
    ns.only_scheme = parsed.only_scheme
    return ns


if __name__ == "__main__":
    ns = build_v2a_args()
    print("v2-A default config:")
    for k, v in sorted(vars(ns).items()):
        print(f"  {k:24s} = {v}")
