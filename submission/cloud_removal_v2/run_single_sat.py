"""Single-satellite training driver for the model-robustness check.

Picks the satellite with the largest Dirichlet slice at the requested
``--partition_alpha`` and trains the OrbitALIF backbone on it in
isolation -- no intra-plane average, no inter-plane gossip.  Output
NPZ uses the same schema as ``run_smoke.py`` cells so it can be
consumed unchanged by ``cloud_removal_v1/plot_paper_figs.py::fig8``.

Why: the federated ``F_snn`` / ``F_snn_alpha001`` runs measure the
combined robustness of *model + FedBN aggregation*.  A clean
"model-only" robustness number requires removing the aggregation
layer entirely, hence training a single satellite.

Usage:
    python cloud_removal_v2/run_single_sat.py \
        --backbone ann --partition_alpha 0.1 --num_epoch 200 \
        --run_name single_ann_alpha01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Enable both `python -m cloud_removal_v2.run_single_sat` and the
# repo-root `python cloud_removal_v2/run_single_sat.py` invocation
# style: in the latter case __package__ is empty and the repo root is
# not yet on sys.path, so `import cloud_removal_v2.config` would fail.
if __package__ in (None, ""):
    _this = Path(__file__).resolve()
    _parent = _this.parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))

# Reuse the v2 config parser + dataset builders to keep parity with
# the federated runs (same Dirichlet seed, same augment params, etc.).
from cloud_removal_v2.config import parse_v2a_cli
from cloud_removal_v2.run_smoke import (
    _ensure_omp_threads, _set_seed, _negotiate_backend,
    _load_datasets, _log,
)
from cloud_removal_v2.dataset import build_plane_satellite_partitions_v2
from cloud_removal_v1.task import CloudRemovalSNNTask
from cloud_removal_v1.evaluation import (
    evaluate_centerpatch, EvalResult, _torch_psnr, _torch_ssim, _center_crop,
)


@torch.no_grad()
def _evaluate_batch_bn(model: torch.nn.Module,
                       loader,
                       patch_size: int,
                       device: torch.device) -> "EvalResult":
    """Single-sat BN-poisoning workaround: evaluate with model in train()
    mode so BN uses *batch* statistics rather than the running stats that
    accumulated on the lone training satellite's distribution.  Otherwise
    eval-mode BN normalisation pushes test inputs into a distribution the
    network was not trained on, producing NaN predictions even when the
    training loss is healthy.
    """
    from spikingjelly.activation_based import functional
    model.train()                 # IMPORTANT: not eval()
    res = EvalResult()
    for batch in loader:
        cloudy, clear = batch
        cloudy = cloudy.to(device, non_blocking=True)
        clear  = clear.to(device,  non_blocking=True)
        cloudy = _center_crop(cloudy, patch_size)
        clear  = _center_crop(clear,  patch_size)
        functional.reset_net(model)
        pred = model(cloudy)
        functional.reset_net(model)
        for i in range(pred.shape[0]):
            res.psnr_per_image.append(_torch_psnr(pred[i], clear[i]).item())
            res.ssim_per_image.append(_torch_ssim(pred[i], clear[i]).item())
    return res


def _parse(argv=None) -> argparse.Namespace:
    """Strip our extra flags first, then delegate the rest to the v2 parser.

    Keeps parity with federated runs: every v2-A knob is honoured.
    """
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--plane_idx", type=int, default=-1,
                       help="Pin the chosen satellite's plane index. "
                            "Default -1 = auto-pick (largest Dirichlet slice).")
    extra.add_argument("--sat_idx", type=int, default=-1,
                       help="Pin the in-plane satellite index. "
                            "Default -1 = auto-pick alongside --plane_idx.")
    extra.add_argument("--eval_patch_size", type=int, default=64)
    extra.add_argument("--alphas", type=str, default="",
                       help="Comma-separated Dirichlet alphas to sweep in a "
                            "single invocation, e.g. '0.1,0.01'. Each alpha "
                            "produces v2a_<run_name_prefix>_alpha<tag>_*.npz "
                            "where tag is alpha with the dot stripped "
                            "(0.1->01, 0.01->001). When set, --partition_alpha "
                            "and --run_name are ignored.")
    extra.add_argument("--run_name_prefix", type=str, default="single",
                       help="Used only with --alphas to build per-alpha "
                            "run_name = <prefix>_<backbone>_alpha<tag>.")
    # Convenience overrides for the v2 defaults that parse_v2a_cli does
    # not expose directly; left as None so the V2A_DEFAULTS value is
    # honoured when the user does not pass them.
    extra.add_argument("--intra_plane_iters", type=int, default=None)
    extra.add_argument("--local_iters",       type=int, default=None)
    extra.add_argument("--warmup_epochs",     type=int, default=None)
    extra_ns, remaining = extra.parse_known_args(argv)

    base_ns = parse_v2a_cli(remaining)
    base_ns.plane_idx       = extra_ns.plane_idx
    base_ns.sat_idx         = extra_ns.sat_idx
    base_ns.eval_patch_size = extra_ns.eval_patch_size
    base_ns.alphas          = extra_ns.alphas
    base_ns.run_name_prefix = extra_ns.run_name_prefix
    if extra_ns.intra_plane_iters is not None:
        base_ns.intra_plane_iters = extra_ns.intra_plane_iters
    if extra_ns.local_iters is not None:
        base_ns.local_iters = extra_ns.local_iters
    if extra_ns.warmup_epochs is not None:
        base_ns.warmup_epochs = extra_ns.warmup_epochs
    return base_ns


def _alpha_tag(alpha: float) -> str:
    """0.1 -> '01', 0.01 -> '001', 1.0 -> '10'.  Matches the existing
    F_snn / F_snn_alpha001 naming convention so fig8 picks them up."""
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")   # '0.1', '0.01', '1'
    if not s or s == "0":
        return "0"
    return s.replace(".", "")                    # '0.1'->'01', '0.01'->'001'


def _pick_largest_sat(client_datasets) -> tuple:
    """Return (plane_idx, sat_idx, size) of the largest Dirichlet slice."""
    best_p, best_s, best_n = 0, 0, -1
    for p_i, row in enumerate(client_datasets):
        for s_i, ds in enumerate(row):
            n = len(ds)
            if n > best_n:
                best_p, best_s, best_n = p_i, s_i, n
    return best_p, best_s, best_n


def _run_one(args: argparse.Namespace,
             train_ms,
             test_ms,
             device: torch.device) -> None:
    """One isolated single-satellite run with the alpha already set on `args`.

    Hoisted out of main() so the --alphas sweep can reuse the same loaded
    train_ms/test_ms across alphas (Dirichlet partition is rebuilt inside,
    so no data leakage).
    """
    # Re-seed at the start of each run so model initialisation is byte-
    # identical across alpha values; otherwise the second sweep iteration
    # would inherit a downstream RNG state and start from different
    # weights, contaminating the alpha-comparison.
    _set_seed(args.seed, deterministic=getattr(args, "deterministic", False))

    aug_params = {
        "hflip_p":  args.aug_hflip_p,
        "vflip_p":  args.aug_vflip_p,
        "rot90_p":  args.aug_rot90_p,
        "rot270_p": args.aug_rot270_p,
    }
    client_datasets = build_plane_satellite_partitions_v2(
        train_ms,
        num_planes=args.num_planes,
        sats_per_plane=args.sats_per_plane,
        mode=args.partition_mode,
        alpha=args.partition_alpha,
        seed=args.partition_seed,
        min_per_client=args.min_samples_per_client,
        augment=bool(args.augment),
        augment_params=aug_params,
    )
    sizes = [[len(ds) for ds in row] for row in client_datasets]
    _log(f"per-client sizes: {sizes}")

    # --- 2. Pick the satellite ------------------------------------------
    if args.plane_idx >= 0 and args.sat_idx >= 0:
        plane_idx, sat_idx = args.plane_idx, args.sat_idx
        n_chosen = sizes[plane_idx][sat_idx]
    else:
        plane_idx, sat_idx, n_chosen = _pick_largest_sat(client_datasets)
    _log(f"chosen satellite: plane={plane_idx} sat={sat_idx} "
         f"size={n_chosen} (alpha={args.partition_alpha})")

    # --- 3. Test loader -------------------------------------------------
    # Larger eval batch than the federated default (1) so that the
    # batch-statistics BN workaround above sees enough samples to give
    # meaningful means/vars.  drop_last=False to keep every test image.
    test_loader = DataLoader(
        test_ms,
        batch_size=max(8, args.test_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- 4. One isolated task on the chosen sat -------------------------
    args.bn_local = False         # irrelevant when there is no aggregation
    task = CloudRemovalSNNTask(
        args=args,
        local_dataset=client_datasets[plane_idx][sat_idx],
        device=str(device),
    )
    n_params = sum(p.numel() for p in task.model.parameters())
    _log(f"backbone={args.backbone}  params={n_params/1e6:.2f}M")

    # --- 5. Training loop -----------------------------------------------
    history: Dict[str, List] = {
        "epochs":       [],
        "train_loss":   [],
        "eval_psnr":    [],
        "eval_ssim":    [],
        "comm_bytes":   [],
        "wall_seconds": [],
        # per-plane / per-sat fields kept for fig8 schema parity
        "per_plane_psnr": [],
        "per_plane_ssim": [],
    }

    t0 = time.time()
    eval_every = getattr(args, "eval_every", 5)
    for ep in range(1, args.num_epoch + 1):
        # Match federated work-per-round: intra_plane_iters * local_iters
        # passes through this sat's tiny dataset.
        round_losses = []
        for _ in range(args.intra_plane_iters):
            loss, _, _ = task.local_training(
                total_global_rounds=args.num_epoch,
                warmup_rounds=getattr(args, "warmup_epochs", 3),
            )
            round_losses.append(loss)
        train_loss = float(np.mean(round_losses))
        task.global_round = ep

        do_eval = (ep % eval_every == 0) or (ep == args.num_epoch)
        if do_eval:
            try:
                res = _evaluate_batch_bn(
                    task.model, test_loader,
                    patch_size=args.eval_patch_size, device=device,
                )
                psnr, ssim = float(res.mean_psnr), float(res.mean_ssim)
            except Exception as e:
                _log(f"WARN: eval failed at ep {ep} ({type(e).__name__}: {e})")
                psnr, ssim = float("nan"), float("nan")
        else:
            psnr, ssim = float("nan"), float("nan")

        history["epochs"].append(ep)
        history["train_loss"].append(train_loss)
        history["eval_psnr"].append(psnr)
        history["eval_ssim"].append(ssim)
        history["comm_bytes"].append(0)        # no communication
        history["wall_seconds"].append(time.time() - t0)
        # one "plane" containing the chosen sat — keeps fig5/fig8 happy
        history["per_plane_psnr"].append([psnr])
        history["per_plane_ssim"].append([ssim])
        _log(f"[single|{args.backbone}|alpha={args.partition_alpha}] "
             f"ep {ep:03d}/{args.num_epoch}  loss={train_loss:.4f}  "
             f"PSNR={psnr:.3f}dB  SSIM={ssim:.4f}  "
             f"wall={history['wall_seconds'][-1]:.1f}s")

    # --- 6. Save final model checkpoint (so eval can be redone offline
    #        if the live PSNR trace turns out to be unreliable) ---------
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"v2a_{args.run_name}_final.pt"
    torch.save({
        "state_dict": task.model.state_dict(),
        "config":     vars(args),
        "plane_idx":  plane_idx,
        "sat_idx":    sat_idx,
    }, ckpt_path)
    _log(f"wrote ckpt {ckpt_path}")

    # --- 7. Persist in federated-compatible NPZ -------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"v2a_{args.run_name}_fedbn_Gossip_Averaging.npz"
    np.savez_compressed(
        npz_path,
        epochs=np.asarray(history["epochs"]),
        train_loss=np.asarray(history["train_loss"]),
        eval_psnr=np.asarray(history["eval_psnr"]),
        eval_ssim=np.asarray(history["eval_ssim"]),
        comm_bytes=np.asarray(history["comm_bytes"]),
        wall_seconds=np.asarray(history["wall_seconds"]),
        per_plane_psnr=np.array(history["per_plane_psnr"], dtype=object),
        per_plane_ssim=np.array(history["per_plane_ssim"], dtype=object),
    )
    _log(f"wrote {npz_path}")

    # Lightweight JSON summary (fig8 doesn't need it but keep parity).
    summary = {
        "run_name":         args.run_name,
        "backbone":         args.backbone,
        "partition_alpha":  args.partition_alpha,
        "plane_idx":        plane_idx,
        "sat_idx":          sat_idx,
        "tile_count":       n_chosen,
        "num_epoch":        args.num_epoch,
        "PSNR_final":       history["eval_psnr"][-1],
        "SSIM_final":       history["eval_ssim"][-1],
    }
    with open(out_dir / f"v2a_{args.run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"wrote summary: PSNR_final={summary['PSNR_final']:.3f} "
         f"SSIM_final={summary['SSIM_final']:.4f}")


def main(argv=None) -> None:
    args = _parse(argv)
    _ensure_omp_threads()
    _set_seed(args.seed, deterministic=getattr(args, "deterministic", False))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")
    args.vlif_backend = _negotiate_backend(args.vlif_backend)
    _log(f"VLIFNet backend: {args.vlif_backend}")

    # Datasets are shared across the alpha sweep; only the Dirichlet
    # partition (which lives inside _run_one) is rebuilt per alpha.
    train_ms, test_ms = _load_datasets(args)

    if args.alphas.strip():
        # --- Sweep mode -------------------------------------------------
        import gc
        alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
        _log(f"alpha sweep: {alphas}  backbone={args.backbone}")
        for alpha in alphas:
            args.partition_alpha = float(alpha)
            args.run_name = (f"{args.run_name_prefix}_{args.backbone}"
                             f"_alpha{_alpha_tag(alpha)}")
            _log(f"\n========== alpha={alpha}  run_name={args.run_name} "
                 f"==========")
            _run_one(args, train_ms, test_ms, device)
            # Free the previous run's model + optimizer + dataloader
            # workers before the next alpha builds new ones.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # --- Single-alpha mode (legacy CLI) ----------------------------
        _run_one(args, train_ms, test_ms, device)


if __name__ == "__main__":
    main()
