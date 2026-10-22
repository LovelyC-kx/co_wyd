"""
v2 Path-A runner.

Sweeps 2 BN modes × 3 aggregation schemes = 6 sequential runs on the
same Dirichlet-non-IID partition of CUHK-CR1 + CUHK-CR2.  Saves each
run's per-epoch metrics, final model state_dict, and a combined summary.

Outputs
-------
    Outputs_v2/v2a_<run_name>_<bn_mode>_<scheme>.npz
        epochs, train_loss, eval_psnr, eval_ssim, comm_bytes,
        per_plane_psnr, per_plane_ssim, wall_seconds.
    Outputs_v2/ckpts/<run_name>_<bn_mode>_<scheme>_plane<p>.pt
        final state_dict per plane (FedBN → planes differ; FedAvg →
        all planes identical so we only save plane 0).
    Outputs_v2/v2a_<run_name>_summary.json
        all 6 final numbers + the config snapshot.
    Outputs_v2/tb/<run_name>/<bn_mode>/<scheme>/ (tensorboard)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Enable `python -m cloud_removal_v2.run_smoke` and `python run_smoke.py`
if __package__ in (None, ""):
    _this = Path(__file__).resolve()
    _parent = _this.parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.run_smoke import main   # noqa: E402
    # Only run main() when invoked AS a script — never on stray imports.
    # Previously the unconditional sys.exit(0) below would kill any caller
    # that happened to import with no __package__ context, even a unit test.
    if __name__ == "__main__":
        main()
        sys.exit(0)

from .config import parse_v2a_cli
from .dataset import (
    MultiSourceCloudDataset,
    build_plane_satellite_partitions_v2,
    seed_worker,
)
from .inline_logging import BnDriftLogger, CosineSimLogger

from cloud_removal_v1.constants import (
    GOSSIP, RELAYSUM, ALLREDUCE,
    SCHEMES, SCHEME_LABEL,
)
from cloud_removal_v1.constellation import CloudRemovalConstellation
from cloud_removal_v1.evaluation import evaluate_per_plane, average_eval_results


_BN_MODES: List[Tuple[str, bool]] = [
    ("fedavg", False),
    ("fedbn",  True),
]
# Stable, position-independent cell index for RNG seeding.  Computed from
# (bn_mode, scheme) so that `--only_bn fedbn` re-runs use the SAME seed
# the cell would have used in a full sweep — preserving B-REP-1's
# cross-cell-comparability fix even under partial sweeps.
_BN_TO_IDX: Dict[str, int] = {bn: i for i, (bn, _) in enumerate(_BN_MODES)}


def _stable_cell_idx(bn_mode: str, scheme: str) -> int:
    """Deterministic ID for one (bn_mode, scheme) cell, independent of
    sweep ordering or `--only_*` filters."""
    return _BN_TO_IDX[bn_mode] * len(SCHEMES) + SCHEMES.index(scheme)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG source we know about.  When `deterministic=True`,
    additionally request cuDNN to use deterministic algorithms (slower
    but bit-exact across runs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Newer torch — surface non-deterministic ops as errors.  We don't
        # error out (some ops are unavoidably non-det); use 'warn' so the
        # user knows where determinism could leak.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _ensure_omp_threads() -> None:
    """Suppress AutoDL's `libgomp: Invalid value for environment variable
    OMP_NUM_THREADS` warning by setting a sensible default if unset."""
    if "OMP_NUM_THREADS" not in os.environ or not os.environ["OMP_NUM_THREADS"].strip():
        os.environ["OMP_NUM_THREADS"] = "8"


def _log(msg: str) -> None:
    print(f"[v2a] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Atomic file writes (round-4 audit fix: survive SIGKILL / disk-full / Ctrl-C
# mid-write without leaving corrupt .npz / .pt / .json on disk).  All writes
# go to a temp file in the SAME directory first, then os.replace() — which is
# atomic on POSIX — swaps it into place.  If the process dies mid-write, only
# the tmp file is leaked; the live output is either the old version or not
# present at all.  Never a half-written file that resume/plot would silently
# mis-read.
# ---------------------------------------------------------------------------

def _atomic_write_json(path: str, obj) -> None:
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=d, delete=False,
                                     suffix=".tmp", prefix=".json_") as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _atomic_save_torch(state_dict, path: str) -> None:
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=d, delete=False,
                                     suffix=".tmp", prefix=".pt_") as tmp:
        torch.save(state_dict, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _atomic_savez(path: str, **arrays) -> None:
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    # np.savez APPENDS ".npz" to a STRING path if missing — if we passed a
    # tempfile path ending in ".tmp" numpy would create "foo.tmp.npz" and
    # our subsequent os.replace(tmp_path, path) would target the wrong file.
    # Pass a FILE HANDLE instead: numpy treats handles as-is.
    fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".tmp", prefix=".npz_")
    try:
        with os.fdopen(fd, "wb") as tmp:
            np.savez(tmp, **arrays)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _merge_summary_cell(summary_path: str, cell_key: str, history: Dict,
                         config_snapshot: Dict) -> None:
    """Incrementally merge one cell's final metrics into summary.json.

    Called AFTER each cell completes so interrupt-at-cell-4 preserves cells
    1-3's final PSNR/SSIM/comm records.  Non-atomic writes are avoided via
    _atomic_write_json.
    """
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            if "final" not in summary or not isinstance(summary["final"], dict):
                summary["final"] = {}
        except Exception:
            summary = {"final": {}}
    else:
        summary = {"final": {}}

    ep_arr = np.asarray(history["epochs"])
    psnr_arr = np.asarray(history["eval_psnr"], dtype=float)
    ssim_arr = np.asarray(history["eval_ssim"], dtype=float)
    valid = ~np.isnan(psnr_arr)
    if valid.any():
        last_valid = int(ep_arr[valid][-1])
        psnr_last  = float(psnr_arr[valid][-1])
        ssim_last  = float(ssim_arr[valid][-1])
    else:
        last_valid, psnr_last, ssim_last = -1, float("nan"), float("nan")
    summary["final"][cell_key] = {
        "PSNR_final": psnr_last,
        "SSIM_final": ssim_last,
        "final_eval_epoch": last_valid,
        "train_loss_final": float(history["train_loss"][-1]),
        "total_comm_bytes":  int(sum(history["comm_bytes"])),
        "total_wall_seconds": float(sum(history["wall_seconds"])),
    }
    summary["config"] = config_snapshot
    _atomic_write_json(summary_path, summary)


def _cupy_available() -> bool:
    try:
        import cupy   # noqa: F401
        return True
    except Exception:
        return False


def _negotiate_backend(requested: str) -> str:
    if requested == "cupy" and not _cupy_available():
        _log("WARN: cupy not importable → backend='torch'")
        return "torch"
    return requested


def _load_datasets(args) -> Tuple[MultiSourceCloudDataset, MultiSourceCloudDataset]:
    """Return (train_multisource, test_multisource).

    Both are MultiSourceCloudDatasets with source labels attached; the
    train view uses random-crop patch_size, the test view returns full
    resolution.  Falls back to a derived 8:2 split per source when no
    explicit train/test sub-folder is found in that source.
    """
    # For each configured source, try explicit split first; if absent,
    # run derived_train_test_split on the flat layout of that single
    # source and rebuild a tiny wrapper that feeds MultiSourceCloudDataset.

    # Simpler path for v2-A: trust the explicit train/test layout
    # (CUHK-CR1/2 ships with it).  If a source doesn't have it, we raise
    # a clear error — derived-split across multi-source is semantically
    # ambiguous (which split seed? shared or per-source?).
    sources_with_split = []
    for s in args.sources:
        sources_with_split.append(s)

    # CRITICAL: with_labels MUST be False for the training dataset.
    # CloudRemovalSNNTask.local_training unpacks `for cloudy, clear in loader`,
    # which would raise "too many values to unpack" if the underlying dataset
    # returned (cloudy, clear, label) 3-tuples.  Dirichlet partitioning does
    # NOT need labels to flow through __getitem__ — it queries
    # dataset.source_labels() directly, which is a method independent of the
    # with_labels flag.
    train = MultiSourceCloudDataset(sources_with_split,
                                    split=args.train_split,
                                    patch_size=args.patch_size,
                                    with_labels=False,
                                    strict=True)
    test = MultiSourceCloudDataset(sources_with_split,
                                   split=args.test_split,
                                   patch_size=None,
                                   with_labels=False,
                                   strict=True)
    _log(train.describe())
    _log(f"test: {test.describe()}")
    return train, test


# ---------------------------------------------------------------------------
# One (bn_mode, scheme) cell
# ---------------------------------------------------------------------------

def _reset_rng_for_cell(base_seed: int, cell_idx: int) -> None:
    """Reset ALL RNGs to a deterministic per-cell value so that six cells
    all see the same augmentation / optimiser noise stream, preserving
    cross-cell comparability.  Without this, torch / numpy RNG state
    drifts across the ~50 × 30 × 2 × 2 training steps of each earlier
    cell, and later cells get a different augmentation sequence."""
    cell_seed = (base_seed + 10_000 * (cell_idx + 1)) & 0x7FFFFFFF
    random.seed(cell_seed)
    np.random.seed(cell_seed)
    torch.manual_seed(cell_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cell_seed)


def _run_cell(bn_mode: str,
              bn_local: bool,
              scheme: str,
              args,
              test_loader: DataLoader,
              client_datasets,
              device: torch.device,
              writer,
              seed_state_dict: Optional[Dict[str, torch.Tensor]] = None,
              ) -> Tuple[Dict, Dict[int, Dict[str, torch.Tensor]]]:
    """Train one (bn_mode, scheme) combination for args.num_epoch rounds.

    Returns (history_dict, per_plane_ckpts_dict).  Per-plane state_dicts
    are kept regardless of BN mode — under FedAvg with Gossip or RelaySum
    aggregation, planes still diverge (each only averages with its chain
    neighbours), so saving only plane-0 would silently lose information.
    """
    _log(f"========== BN={bn_mode} | scheme={scheme} ==========")
    # Mutate args for this cell's bn_local; GUARD with try/finally so
    # an exception cannot leak a corrupt flag out to the outer loop.
    prev_bn_local = args.bn_local
    args.bn_local = bn_local
    try:
        constellation = CloudRemovalConstellation(
            num_planes=args.num_planes,
            sats_per_plane=args.sats_per_plane,
            client_datasets=client_datasets,
            args=args,
            init_state_dict=seed_state_dict,
            device=str(device),
            logger=_log,
        )

        history = {
            "bn_mode":        bn_mode,
            "scheme":         scheme,
            "epochs":         [],
            "train_loss":     [],
            "eval_psnr":      [],
            "eval_ssim":      [],
            "comm_bytes":     [],
            "per_plane_psnr": [],
            "per_plane_ssim": [],
            "wall_seconds":   [],
        }

        # Phase-1 P1.1: inline loggers for SC-16a/b/c drift + Seo24 cosine.
        # Both are cheap (< 300 ms / eval) and their output is serialised
        # into the cell's npz/JSON for post-hoc analysis (paper §VI-C).
        drift_logger = (BnDriftLogger(args.num_planes)
                        if getattr(args, "log_drift", True) else None)
        cos_logger = (CosineSimLogger(args.num_planes)
                      if getattr(args, "log_cosine_sim", True) else None)

        for ep in range(1, args.num_epoch + 1):
            t0 = time.time()
            train_loss = constellation.train_one_round(scheme)

            # B-EDGE-2: detect NaN training loss IMMEDIATELY and abort
            # the whole sweep with an actionable message.  AdamW + clip_grad
            # cannot recover from NaN gradients (clip with NaN-norm ⇒
            # NaN weights), and silently continuing to write zeros wastes
            # the rest of the run's GPU budget.
            if not (train_loss == train_loss):   # NaN check (NaN != NaN)
                raise RuntimeError(
                    f"[{bn_mode}|{scheme}] ep {ep}: training loss is NaN. "
                    f"Diagnose: (a) lower --lr (try 5e-4), "
                    f"(b) verify input range is [0,1] via dataset probe, "
                    f"(c) check for empty client subsets in this Dirichlet seed."
                )

            do_eval = (ep % args.eval_every == 0) or (ep == args.num_epoch)
            if do_eval:
                # Round-4 audit: wrap eval so a one-off shape/IO hiccup
                # (bad test image, OSError on a TIFF) records NaN for this
                # epoch instead of aborting the whole cell.  If eval is
                # structurally broken it will fail every epoch and the
                # final summary will show all-NaN — loud enough.
                try:
                    per_plane = evaluate_per_plane(
                        constellation, test_loader,
                        mode=args.eval_mode,
                        patch_size=args.eval_patch_size,
                        window=args.sliding_window,
                        stride=args.sliding_stride,
                        device=device,
                    )
                    mean = average_eval_results(per_plane)
                    psnr, ssim = mean.mean_psnr, mean.mean_ssim
                    pp_psnr = [r.mean_psnr for r in per_plane]
                    pp_ssim = [r.mean_ssim for r in per_plane]
                except Exception as e:
                    _log(f"WARN: eval failed at ep {ep} ({type(e).__name__}: {e}); "
                         f"recording NaN and continuing")
                    psnr, ssim, pp_psnr, pp_ssim = np.nan, np.nan, [], []

                # Phase-1 P1.1: drift + cosine snapshots at every eval boundary.
                # These populate history["bn_drift"] and history["cos_sim"]
                # which are saved alongside PSNR/SSIM in the cell's npz.
                # Wrap in try: failure here must not abort training.
                if drift_logger is not None:
                    try:
                        drift_logger.snapshot(constellation, ep)
                    except Exception as e:
                        _log(f"WARN: drift snapshot failed at ep {ep} "
                             f"({type(e).__name__}: {e}); disabling drift log")
                        drift_logger = None
                if cos_logger is not None:
                    try:
                        cos_logger.snapshot(constellation, ep)
                    except Exception as e:
                        _log(f"WARN: cos-sim snapshot failed at ep {ep} "
                             f"({type(e).__name__}: {e}); disabling cos log")
                        cos_logger = None
            else:
                psnr, ssim, pp_psnr, pp_ssim = np.nan, np.nan, [], []

            dt = time.time() - t0
            cb = int(constellation.round_bytes[-1])

            history["epochs"].append(ep)
            history["train_loss"].append(float(train_loss))
            history["eval_psnr"].append(float(psnr))
            history["eval_ssim"].append(float(ssim))
            history["comm_bytes"].append(cb)
            history["per_plane_psnr"].append(pp_psnr)
            history["per_plane_ssim"].append(pp_ssim)
            history["wall_seconds"].append(dt)

            _log(f"[{bn_mode:6s}|{scheme:24s}] ep {ep:02d}/{args.num_epoch}  "
                 f"loss={train_loss:.4f}  PSNR={psnr:.3f}dB  SSIM={ssim:.4f}  "
                 f"comm={cb/1e6:.1f}MB  time={dt:.1f}s")

            if writer is not None:
                # Round-4 audit: TB writes can hit OSError (disk full / FS
                # oddity on AutoDL).  Never let a logging failure abort the
                # training run — disable the writer and keep training.
                try:
                    writer.add_scalar(f"{bn_mode}/{scheme}/train_loss", train_loss, ep)
                    if not np.isnan(psnr):
                        writer.add_scalar(f"{bn_mode}/{scheme}/eval_psnr", psnr, ep)
                        writer.add_scalar(f"{bn_mode}/{scheme}/eval_ssim", ssim, ep)
                    writer.add_scalar(f"{bn_mode}/{scheme}/comm_bytes", cb, ep)
                except Exception as e:
                    _log(f"WARN: TB write failed ({type(e).__name__}: {e}); "
                         f"disabling writer for the rest of the sweep")
                    writer = None

        # Phase-1 P1.1: persist inline log histories into the cell's npz.
        # These are JSON-serialisable (dicts of lists of dicts) so np.savez
        # accepts them via dtype=object wrapping.
        if drift_logger is not None:
            history["bn_drift"] = drift_logger.get_history()
        if cos_logger is not None:
            history["cos_sim"] = cos_logger.get_history()

        # Save ALL plane state_dicts.  Under FedAvg+Gossip / FedAvg+RelaySum
        # the planes still diverge from each other (each plane only averages
        # with its chain neighbours), so plane-0 alone would silently drop
        # the per-plane variation.  45 MB × 6 cells storage is negligible.
        ckpts: Dict[int, Dict[str, torch.Tensor]] = {}
        for p in range(args.num_planes):
            ckpts[p] = constellation.planes[p][0].get_weights(cpu=True)

        # Release GPU memory
        del constellation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        # Always restore, even on exception, so the outer sweep loop
        # isn't corrupted.
        args.bn_local = prev_bn_local

    return history, ckpts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_v2a_cli(argv)
    _ensure_omp_threads()
    _set_seed(args.seed, deterministic=getattr(args, "deterministic", False))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")
    if device.type == "cuda":
        _log(f"CUDA device name: {torch.cuda.get_device_name(device)}")
    args.vlif_backend = _negotiate_backend(args.vlif_backend)
    _log(f"VLIFNet backend: {args.vlif_backend}")

    # -- Data ---------------------------------------------------------------
    train_ms, test_ms = _load_datasets(args)
    total_clients = args.num_planes * args.sats_per_plane
    assert len(train_ms) >= total_clients * args.min_samples_per_client, (
        f"train pool {len(train_ms)} too small for {total_clients} clients "
        f"with min_per_client={args.min_samples_per_client}")

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
    per_client_sizes = [[len(ds) for ds in row] for row in client_datasets]
    _log(f"per-client sizes: {per_client_sizes}")

    # Test loader (no augmentation; labels stripped by with_labels=False)
    test_loader_kwargs: Dict = dict(
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        test_loader_kwargs["worker_init_fn"] = seed_worker
    test_loader = DataLoader(test_ms, **test_loader_kwargs)

    # -- Tensorboard -------------------------------------------------------
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.run_name))
    except Exception as e:
        _log(f"tensorboard disabled ({e})")

    # -- Run 2 × 3 sweep ---------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Dict] = {}

    # Summary path is computed early so _merge_summary_cell can incrementally
    # update it after EACH cell completes.  The end-of-sweep block below is
    # kept as a belt-and-suspenders final write (re-confirms all cells).
    summary_path = os.path.join(args.output_dir,
                                f"v2a_{args.run_name}_summary.json")

    # Build a single shared initial model for all 6 cells so the
    # comparison is apples-to-apples.  We use build_vlifnet DIRECTLY (not a
    # full 50-satellite constellation) — that avoids ~30-60 s of wasted
    # GPU time setting up 50 duplicate nets only to throw them away.
    backbone = getattr(args, "backbone", "snn")
    if backbone == "plain":
        from cloud_removal_v1.models import build_plain_unet
        _log("[init] building shared seed state_dict (1 fresh PlainUNet)")
        _seed_net = build_plain_unet(
            dim=args.vlif_dim,
            en_blocks=tuple(args.en_blocks[:3]),
            de_blocks=tuple(args.de_blocks[:3]),
            inp_channels=3, out_channels=3,
        ).to(device)
    else:
        from cloud_removal_v1.models import build_vlifnet
        _log(f"[init] building shared seed state_dict (1 fresh VLIFNet, backbone={backbone})")
        _seed_net = build_vlifnet(
            dim=args.vlif_dim,
            en_num_blocks=tuple(args.en_blocks),
            de_num_blocks=tuple(args.de_blocks),
            T=args.T,
            use_refinement=False,
            inp_channels=3, out_channels=3,
            backend=args.vlif_backend,
            bn_variant=getattr(args, "bn_variant", "tdbn"),
            backbone=backbone,
        ).to(device)
    seed_state_dict: Dict = {}
    for k, v in _seed_net.state_dict().items():
        if isinstance(v, torch.Tensor):
            seed_state_dict[k] = v.detach().cpu().clone()
        else:
            seed_state_dict[k] = v   # SpikingJelly memory scalars (float / None)
    del _seed_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log("[init] seed ready")

    cell_idx_used = -1
    for bn_mode, bn_local in _BN_MODES:
        if args.only_bn and args.only_bn != bn_mode:
            continue
        for scheme in SCHEMES:
            if args.only_scheme and args.only_scheme != scheme:
                continue

            # Stable per-(bn_mode, scheme) RNG reset — does NOT depend on
            # sweep position or --only_* filters, so a partial re-run
            # yields the SAME RNG state the same cell saw in the full sweep.
            cell_idx_used = _stable_cell_idx(bn_mode, scheme)
            _reset_rng_for_cell(args.seed, cell_idx_used)

            history, ckpts = _run_cell(
                bn_mode=bn_mode,
                bn_local=bn_local,
                scheme=scheme,
                args=args,
                test_loader=test_loader,
                client_datasets=client_datasets,
                device=device,
                writer=writer,
                seed_state_dict=seed_state_dict,
            )
            tag = f"{args.run_name}_{bn_mode}_{scheme}"
            npz_path = os.path.join(args.output_dir, f"v2a_{tag}.npz")
            # Round-4 audit: atomic write via tempfile+os.replace — if the
            # process is killed mid-save (SIGKILL, disk full, node evict),
            # the live npz is either the previous version or absent.  Never
            # a half-written file that plot_results would silently mis-read.
            _atomic_savez(
                npz_path,
                epochs=np.array(history["epochs"]),
                train_loss=np.array(history["train_loss"]),
                eval_psnr=np.array(history["eval_psnr"]),
                eval_ssim=np.array(history["eval_ssim"]),
                comm_bytes=np.array(history["comm_bytes"]),
                wall_seconds=np.array(history["wall_seconds"]),
                per_plane_psnr=np.array(history["per_plane_psnr"], dtype=object),
                per_plane_ssim=np.array(history["per_plane_ssim"], dtype=object),
            )
            _log(f"wrote {npz_path}")
            # Save ALL plane checkpoints regardless of BN mode.
            # FedBN planes obviously differ; FedAvg Gossip / RelaySum
            # planes ALSO differ (each plane only averages with its chain
            # neighbours), so saving only plane-0 would silently drop the
            # per-plane variation that the paper reports mean/std over.
            for p, sd in ckpts.items():
                path = os.path.join(args.ckpt_dir, f"{tag}_plane{p}.pt")
                _atomic_save_torch(sd, path)
            _log(f"wrote {len(ckpts)} ckpts under {args.ckpt_dir}")
            all_results[f"{bn_mode}_{scheme}"] = history

            # Round-4 audit: update summary.json IMMEDIATELY after each
            # cell finishes so an interrupt at cell-4 of 6 preserves
            # cells 1-3's final PSNR/SSIM records.  The end-of-sweep
            # summary write below stays as a belt-and-suspenders safety
            # net, but the incremental write is the primary durability
            # mechanism.
            cfg_snap = {k: v for k, v in vars(args).items()
                        if isinstance(v, (int, float, str, bool, list, tuple))}
            _merge_summary_cell(summary_path, f"{bn_mode}_{scheme}",
                                history, cfg_snap)
            _log(f"incrementally merged {bn_mode}_{scheme} into {summary_path}")

    if writer is not None:
        writer.close()

    # -- Summary (belt-and-suspenders) -------------------------------------
    # Each completed cell was ALREADY merged into summary.json incrementally
    # via _merge_summary_cell (above).  This final pass re-confirms every
    # cell and writes atomically one more time — so even a redundancy-check
    # run with no cells executed leaves a consistent file on disk.
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            if "final" not in summary or not isinstance(summary["final"], dict):
                summary["final"] = {}
        except Exception as e:
            _log(f"WARN: failed to reopen summary ({e}); rebuilding from this run only")
            summary = {"final": {}}
    else:
        summary = {"final": {}}
    summary["config"] = {k: v for k, v in vars(args).items()
                         if isinstance(v, (int, float, str, bool, list, tuple))}

    for key, history in all_results.items():
        ep_arr = np.asarray(history["epochs"])
        psnr_arr = np.asarray(history["eval_psnr"], dtype=float)
        ssim_arr = np.asarray(history["eval_ssim"], dtype=float)
        valid = ~np.isnan(psnr_arr)
        if valid.any():
            last_valid = int(ep_arr[valid][-1])
            psnr_last  = float(psnr_arr[valid][-1])
            ssim_last  = float(ssim_arr[valid][-1])
        else:
            last_valid, psnr_last, ssim_last = -1, float("nan"), float("nan")
        summary["final"][key] = {
            "PSNR_final": psnr_last,
            "SSIM_final": ssim_last,
            "final_eval_epoch": last_valid,
            "train_loss_final": float(history["train_loss"][-1]),
            "total_comm_bytes":  int(sum(history["comm_bytes"])),
            "total_wall_seconds": float(sum(history["wall_seconds"])),
        }
    _atomic_write_json(summary_path, summary)
    _log(f"wrote {summary_path}")
    _log("done.")


if __name__ == "__main__":
    main()
