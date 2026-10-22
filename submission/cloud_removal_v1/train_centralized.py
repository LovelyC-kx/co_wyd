"""
Centralized cloud-removal training entry point.

Drives the §IV-B / §IV-C centralized SOTA + ablation experiments on
CUHK-CR1 or CUHK-CR2.  Reuses v1's CloudRemovalSNNTask losses,
PairedCloudDataset, and evaluation primitives — only the FL orchestrator
is replaced with a plain single-process training loop.

Supports three backbones:
  * vlif       : VLIFNet (SNN with MultiSpike-4)
  * vlif_ann   : VLIFNet with LIF → ReLU substitution (same arch, ANN)
  * plain_ann  : PlainUNet (no attention / no frequency / no spike) baseline

And four ablation modes (apply only when backbone in {vlif, vlif_ann}):
  * none           : full model
  * no_agfm        : every GatedSkipFusion → _PlainSkipFusion (drops the
                     content-dependent Conv1x1 + sigmoid gate; keeps the
                     post-fusion LIF + TDBN tail so the only variable
                     relative to the full model is the gating mechanism).
                     Affects the L1 and L2 skip connections; L3 has no
                     AGFM in the full model so is unchanged.
  * no_fsta        : FSTAModule + FreMLPBlock → Identity
  * binary_spike   : MultiSpike-4 → binary {0, 1} (vlif backbone only).
                     NOTE: VLIFNet's mem_update modules accumulate small
                     residual signals; under the default init (BN
                     gamma_init = alpha * V_th ~ 0.106) and short training
                     budgets (< ~50 ep) the membrane potential may not
                     exceed the 0.5 firing threshold, in which case both
                     MultiSpike-4 and binary_spike output all zeros and
                     the ablation is empirically silent.  Use the full
                     300-epoch training schedule for a meaningful B3
                     ablation result.
  * no_dual_group  : SRB → single-group SRB surgery (drops PixelShuffle
                     spatial group + cross-scale gate); fresh-init,
                     trained from scratch.  See _SingleGroupSRB below.

Usage
-----
    # CR1 main result (Tab 1)
    python -m cloud_removal_v1.train_centralized \
        --data_root /abs/path/CUHK-CR1 --dataset_name CR1 \
        --backbone vlif --num_epoch 600 --run_name A1_vlif_cr1

    # CR1 ablation (Tab 2): binary spike
    python -m cloud_removal_v1.train_centralized \
        --data_root /abs/path/CUHK-CR1 --dataset_name CR1 \
        --backbone vlif --ablation binary_spike \
        --num_epoch 300 --run_name B3_binary_cr1

    # CR1 ablation (Tab 1): w/o AGFM
    python -m cloud_removal_v1.train_centralized \
        --data_root /abs/path/CUHK-CR1 --dataset_name CR1 \
        --backbone vlif --ablation no_agfm \
        --num_epoch 300 --run_name B4_no_agfm_cr1

    # Plain ANN U-Net baseline
    python -m cloud_removal_v1.train_centralized \
        --data_root /abs/path/CUHK-CR1 --dataset_name CR1 \
        --backbone plain_ann --num_epoch 400 --run_name C2_plain_cr1

Outputs (under args.output_dir):
    centralized_<run_name>.npz       per-epoch arrays
    centralized_<run_name>_best.pt   best-PSNR ckpt
    centralized_<run_name>_final.pt  last-epoch ckpt
    centralized_<run_name>_summary.json  config + final numbers
    tb/<run_name>/                   tensorboard events (if available)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow `python cloud_removal_v1/train_centralized.py` as well as the -m form.
if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v1.train_centralized import main   # noqa: E402
    if __name__ == "__main__":
        main()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centralized cloud-removal training")
    # Run identity
    p.add_argument("--run_name", type=str, default="centralized")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str, default="./Outputs")
    p.add_argument("--tensorboard_dir", type=str, default="./Outputs/tb")
    # Dataset
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to CUHK-CR1 or CUHK-CR2 root.")
    p.add_argument("--dataset_name", type=str, default="CR1",
                   help="Tag used in output filenames; informational only.")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--test_split", type=str, default="test")
    # Data pipeline
    p.add_argument("--patch_size", type=int, default=64,
                   help="Train random-crop size.  Must be ≥16 and divisible by 4.")
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--test_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    # Model
    p.add_argument("--backbone", type=str, default="vlif",
                   choices=["vlif", "vlif_ann", "plain_ann"])
    p.add_argument("--ablation", type=str, default="none",
                   choices=["none", "no_fsta", "binary_spike",
                            "no_dual_group", "no_agfm"])
    p.add_argument("--vlif_dim", type=int, default=24)
    p.add_argument("--en_blocks", type=int, nargs=4, default=[2, 2, 4, 4])
    p.add_argument("--de_blocks", type=int, nargs=4, default=[2, 2, 2, 2])
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--vlif_backend", type=str, default="torch", choices=["torch", "cupy"])
    p.add_argument("--bn_variant", type=str, default="tdbn", choices=["tdbn", "bn2d"])
    # Optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_epoch", type=int, default=600)
    # Loss
    p.add_argument("--ssim_weight", type=float, default=0.1)
    p.add_argument("--charbonnier_eps", type=float, default=1e-3)
    # Evaluation
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--eval_patch_size", type=int, default=64)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    print(f"[centralized] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model construction (with ablation surgery)
# ---------------------------------------------------------------------------

def _set_submodule(parent: nn.Module, dotted: str, new: nn.Module) -> None:
    """Replace `parent.<dotted>` with `new` (mutates parent in place)."""
    parts = dotted.split(".")
    obj = parent
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


class _BinarySpike(torch.autograd.Function):
    """Drop-in binary replacement for MultiSpike4.quant4 — ablation B3.

    Forward:  output ∈ {0, 1}  (vs MultiSpike4's {0, 0.25, 0.5, 0.75, 1.0});
              fires when the integrated membrane potential exceeds the
              FIRST quantization level threshold of MultiSpike4 (mem > 0.5).
    Backward: rectangular surrogate matching MultiSpike4's window [0, 4]
              and gradient scale 1/4, so that "all else equal" holds —
              the *only* training-time difference vs MultiSpike4 is the
              forward-pass quantization granularity.  See §IV-C of paper.
    """

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return (inp > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad[inp < 0] = 0
        grad[inp > 4] = 0
        return grad / 4.0


class _BinarySpikeModule(nn.Module):
    """nn.Module wrapper around _BinarySpike, callable like MultiSpike4()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _BinarySpike.apply(x)


class _PlainSkipFusion(nn.Module):
    """Vanilla skip connection used by the no_agfm ablation.

    Drops the content-dependent Conv1x1 + sigmoid gate of GatedSkipFusion
    (vlifnet.py:489-503) and instead simply adds the encoder and decoder
    feature maps elementwise. The post-fusion LIF + TDBN tail is kept
    identical to GatedSkipFusion so the *only* variable relative to the
    full model is the gating mechanism — both fusion paths produce a
    [T, B, C, H, W] tensor with the same statistics-normalisation
    behaviour. Initialised fresh; the ablation is trained from scratch.
    """

    def __init__(self, dim: int):
        super().__init__()
        from cloud_removal_v1.models.vlifnet import (
            _make_lif_or_relu, _make_bn, v_th, alpha,
        )
        from spikingjelly.activation_based import functional

        functional.set_step_mode(self, step_mode="m")

        self.lif = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.bn = _make_bn(num_features=dim, alpha=alpha, v_th=v_th)

    def forward(self, dec, enc):
        return self.bn(self.lif(dec + enc))


class _SingleGroupSRB(nn.Module):
    """Spiking Residual Block with only the temporal-LIF group (Group 1).

    Ablation B2 (no_dual_group): surgically removes the PixelShuffle spatial
    group (Group 2) and the cross-scale gate from every SRB in the network.
    The shortcut conv, MultiDimensional attention, and FSTAModule are kept
    so the only variable relative to the full SRB is the second processing
    path.  Initialised fresh (no weight transfer from the full SRB; the
    ablation is always trained from scratch, not fine-tuned).
    """

    def __init__(self, dim: int):
        super().__init__()
        from cloud_removal_v1.models.vlifnet import (
            _make_lif_or_relu, _make_bn, v_th, alpha,
        )
        from cloud_removal_v1.models.fsta_module import FSTAModule
        from spikingjelly.activation_based import functional, layer

        functional.set_step_mode(self, step_mode="m")

        self.lif_1 = _make_lif_or_relu(v_threshold=v_th, decay_input=False)
        self.conv1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn1 = _make_bn(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        self.high_freq_scale_1 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_1 = nn.Parameter(torch.ones(1))

        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            _make_bn(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
        )
        self.attn = layer.MultiDimensionalAttention(
            T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
        self.fsta = FSTAModule(channels=dim, T=4)

    def forward(self, x):
        x_h_1 = self.lif_1(x)
        x_l_1 = x - x_h_1
        combined = (self.high_freq_scale_1 * x_h_1
                    + self.low_freq_scale_1 * x_l_1
                    + x * x_h_1)
        out = self.bn1(self.conv1(combined))

        shortcut = torch.clone(x)
        out = out + self.shortcut(shortcut)
        out = self.attn(out) + shortcut
        out = self.fsta(out)
        return out


def _apply_ablation(model: nn.Module, ablation: str, backbone: str) -> nn.Module:
    if ablation == "none":
        return model
    if backbone == "plain_ann":
        raise ValueError(f"ablation={ablation} is not applicable to backbone=plain_ann")

    # Lazy imports — only loaded for VLIFNet ablations (avoid spikingjelly
    # at import time for plain_ann CPU paths).
    from cloud_removal_v1.models.fsta_module import FSTAModule, FreMLPBlock
    from cloud_removal_v1.models.vlifnet import MultiSpike4, mem_update

    if ablation == "no_fsta":
        # Collect (parent, attr_name) pairs first; mutating during named_modules
        # iteration would skip half the targets.
        targets = []
        for name, sub in model.named_modules():
            if isinstance(sub, (FSTAModule, FreMLPBlock)):
                targets.append(name)
        for name in targets:
            _set_submodule(model, name, nn.Identity())
        _log(f"ablation=no_fsta: replaced {len(targets)} modules with Identity")
        return model

    if ablation == "binary_spike":
        n_replaced = 0
        for sub in model.modules():
            if isinstance(sub, mem_update):
                # mem_update.qtrick is the MultiSpike4 instance
                if isinstance(sub.qtrick, MultiSpike4):
                    sub.qtrick = _BinarySpikeModule()
                    n_replaced += 1
        _log(f"ablation=binary_spike: replaced {n_replaced} MultiSpike4 instances")
        return model

    if ablation == "no_agfm":
        from cloud_removal_v1.models.vlifnet import GatedSkipFusion
        targets = [(name, sub) for name, sub in model.named_modules()
                   if isinstance(sub, GatedSkipFusion)]
        for name, sub in targets:
            # gate_conv: layer.Conv2d(2*dim, dim, 1) -> dim = out_channels
            dim = sub.gate_conv.weight.shape[0]
            _set_submodule(model, name, _PlainSkipFusion(dim).to(
                next(sub.parameters()).device))
        _log(f"ablation=no_agfm: replaced {len(targets)} GatedSkipFusion → PlainSkipFusion")
        return model

    if ablation == "no_dual_group":
        from cloud_removal_v1.models.vlifnet import Spiking_Residual_Block
        targets = [(name, sub) for name, sub in model.named_modules()
                   if isinstance(sub, Spiking_Residual_Block)]
        for name, sub in targets:
            # Infer dim from the existing SRB's first conv weight
            dim = sub.conv1.weight.shape[0]
            _set_submodule(model, name, _SingleGroupSRB(dim).to(
                next(sub.parameters()).device))
        _log(f"ablation=no_dual_group: replaced {len(targets)} SRB → SingleGroupSRB")
        return model

    raise ValueError(f"unknown ablation: {ablation}")


def _build_model(args, device: torch.device) -> Tuple[nn.Module, bool]:
    """Build the model.  Returns (model, is_snn)."""
    if args.backbone == "plain_ann":
        from cloud_removal_v1.models.plain_unet import build_plain_unet
        # PlainUNet uses 3-level U-Net; only first three of en/de_blocks are used.
        model = build_plain_unet(
            dim=args.vlif_dim,
            en_blocks=tuple(args.en_blocks[:3]),
            de_blocks=tuple(args.de_blocks[:3]),
            inp_channels=3, out_channels=3,
        ).to(device)
        return model, False

    # VLIFNet variants — backbone="vlif" → SNN, backbone="vlif_ann" → ReLU
    from cloud_removal_v1.models import build_vlifnet
    sub_backbone = "snn" if args.backbone == "vlif" else "ann"
    model = build_vlifnet(
        dim=args.vlif_dim,
        en_num_blocks=tuple(args.en_blocks),
        de_num_blocks=tuple(args.de_blocks),
        T=args.T,
        use_refinement=False,
        inp_channels=3, out_channels=3,
        backend=args.vlif_backend,
        bn_variant=args.bn_variant,
        backbone=sub_backbone,
    ).to(device)
    if args.ablation != "none":
        if args.ablation == "binary_spike" and args.backbone != "vlif":
            raise ValueError("binary_spike ablation only applies to backbone=vlif")
        model = _apply_ablation(model, args.ablation, args.backbone)
    is_snn = (args.backbone == "vlif")
    return model, is_snn


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _build_loaders(args) -> Tuple[DataLoader, DataLoader]:
    from cloud_removal_v1.dataset import (
        PairedCloudDataset, derived_train_test_split, seed_worker,
    )
    try:
        train_ds: Dataset = PairedCloudDataset(
            args.data_root, split=args.train_split, patch_size=args.patch_size)
        test_ds: Dataset = PairedCloudDataset(
            args.data_root, split=args.test_split, patch_size=None)
        _log(f"explicit split: |train|={len(train_ds)}  |test|={len(test_ds)}")
    except FileNotFoundError:
        train_ds, test_ds = derived_train_test_split(
            args.data_root, args.patch_size, test_ratio=0.2, seed=args.seed)
        _log(f"flat layout (8:2 split): |train|={len(train_ds)}  |test|={len(test_ds)}")

    train_kwargs = dict(
        batch_size=args.train_batch_size, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=args.num_workers,
    )
    if args.num_workers > 0:
        train_kwargs["worker_init_fn"] = seed_worker
        train_kwargs["persistent_workers"] = True

    test_kwargs = dict(
        batch_size=args.test_batch_size, shuffle=False, drop_last=False,
        pin_memory=True, num_workers=args.num_workers,
    )
    if args.num_workers > 0:
        test_kwargs["worker_init_fn"] = seed_worker

    return DataLoader(train_ds, **train_kwargs), DataLoader(test_ds, **test_kwargs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = max(0.0, min(1.0, t))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def _reset_snn(model: nn.Module, is_snn: bool) -> None:
    if not is_snn:
        return
    from spikingjelly.activation_based import functional
    functional.reset_net(model)


def _train_one_epoch(model: nn.Module,
                     loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     criterion: nn.Module,
                     device: torch.device,
                     is_snn: bool,
                     clip_grad: float) -> Tuple[float, float, float]:
    model.train()
    n = 0
    sum_loss = sum_ch = sum_ss = 0.0
    for cloudy, clear in loader:
        cloudy = cloudy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        _reset_snn(model, is_snn)
        optimizer.zero_grad(set_to_none=True)
        pred = model(cloudy)
        loss_ch = criterion.charbonnier(pred, clear)
        loss_ss = criterion.ssim(pred, clear)
        loss = loss_ch + criterion.ssim_weight * loss_ss
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        _reset_snn(model, is_snn)
        n += 1
        sum_loss += loss.item()
        sum_ch += loss_ch.item()
        sum_ss += loss_ss.item()
    if n == 0:
        return math.nan, math.nan, math.nan
    return sum_loss / n, sum_ch / n, sum_ss / n


@torch.no_grad()
def _evaluate(model: nn.Module,
              test_loader: DataLoader,
              eval_patch_size: int,
              device: torch.device) -> Tuple[float, float]:
    """Centre-patch PSNR / SSIM evaluator.

    Reuses cloud_removal_v1.evaluation.evaluate_centerpatch, which is
    safe for ANN models (functional.reset_net is a no-op when no
    spikingjelly Memory* modules are present).
    """
    from cloud_removal_v1.evaluation import evaluate_centerpatch
    res = evaluate_centerpatch(model, test_loader, patch_size=eval_patch_size, device=device)
    return res.mean_psnr, res.mean_ssim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = _parse_args(argv)
    _set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")
    if device.type == "cuda":
        _log(f"CUDA name: {torch.cuda.get_device_name(device)}")

    # Validate patch_size
    if args.patch_size < 16 or args.patch_size % 4 != 0:
        raise ValueError(
            f"--patch_size must be >=16 and divisible by 4 (PlainUNet / VLIFNet "
            f"requirement); got {args.patch_size}")

    # Build model + loaders
    model, is_snn = _build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"backbone={args.backbone}  ablation={args.ablation}  "
         f"params={n_params/1e6:.2f}M  is_snn={is_snn}")

    train_loader, test_loader = _build_loaders(args)

    # Loss + optimizer
    from cloud_removal_v1.task import CloudLoss
    criterion = CloudLoss(ssim_weight=args.ssim_weight, eps=args.charbonnier_eps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=args.wd)

    # Tensorboard
    writer = None
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.run_name))
    except Exception as e:
        _log(f"tensorboard disabled ({e})")

    # Training loop
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    history: Dict[str, list] = {
        "epoch": [], "lr": [], "train_loss": [], "train_charbonnier": [],
        "train_ssim_loss": [], "eval_psnr": [], "eval_ssim": [], "wall_seconds": [],
    }
    best_psnr = -math.inf
    best_epoch = -1
    best_path = os.path.join(args.output_dir, f"centralized_{args.run_name}_best.pt")
    final_path = os.path.join(args.output_dir, f"centralized_{args.run_name}_final.pt")
    npz_path = os.path.join(args.output_dir, f"centralized_{args.run_name}.npz")
    summary_path = os.path.join(args.output_dir, f"centralized_{args.run_name}_summary.json")

    for ep in range(1, args.num_epoch + 1):
        t0 = time.time()
        lr = _cosine_lr(ep - 1, args.num_epoch, args.warmup_epochs, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss, train_ch, train_ss = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, is_snn, args.clip_grad)

        do_eval = (ep % args.eval_every == 0) or (ep == args.num_epoch)
        if do_eval:
            psnr, ssim = _evaluate(model, test_loader, args.eval_patch_size, device)
        else:
            psnr, ssim = float("nan"), float("nan")
        dt = time.time() - t0

        history["epoch"].append(ep)
        history["lr"].append(lr)
        history["train_loss"].append(float(train_loss))
        history["train_charbonnier"].append(float(train_ch))
        history["train_ssim_loss"].append(float(train_ss))
        history["eval_psnr"].append(float(psnr))
        history["eval_ssim"].append(float(ssim))
        history["wall_seconds"].append(dt)

        msg = (f"ep {ep:04d}/{args.num_epoch}  lr={lr:.2e}  loss={train_loss:.4f}  "
               f"PSNR={psnr:.3f}  SSIM={ssim:.4f}  time={dt:.1f}s")
        _log(msg)

        if writer is not None:
            try:
                writer.add_scalar("train/loss", train_loss, ep)
                writer.add_scalar("train/charbonnier", train_ch, ep)
                writer.add_scalar("train/ssim_loss", train_ss, ep)
                writer.add_scalar("optim/lr", lr, ep)
                if not math.isnan(psnr):
                    writer.add_scalar("eval/psnr", psnr, ep)
                    writer.add_scalar("eval/ssim", ssim, ep)
            except Exception:
                writer = None

        # Save best ckpt by PSNR
        if do_eval and not math.isnan(psnr) and psnr > best_psnr:
            best_psnr = psnr
            best_epoch = ep
            torch.save({
                "epoch": ep, "psnr": psnr, "ssim": ssim,
                "state_dict": {k: v.detach().cpu()
                               for k, v in model.state_dict().items()
                               if isinstance(v, torch.Tensor)},
                "config": vars(args),
            }, best_path)

        # Snapshot npz periodically (cheap, lets you ctrl-C and still have data)
        if ep % max(args.eval_every, 5) == 0 or ep == args.num_epoch:
            np.savez(
                npz_path,
                epoch=np.array(history["epoch"]),
                lr=np.array(history["lr"]),
                train_loss=np.array(history["train_loss"]),
                train_charbonnier=np.array(history["train_charbonnier"]),
                train_ssim_loss=np.array(history["train_ssim_loss"]),
                eval_psnr=np.array(history["eval_psnr"]),
                eval_ssim=np.array(history["eval_ssim"]),
                wall_seconds=np.array(history["wall_seconds"]),
            )

    # Final ckpt (latest weights)
    torch.save({
        "epoch": args.num_epoch,
        "state_dict": {k: v.detach().cpu()
                       for k, v in model.state_dict().items()
                       if isinstance(v, torch.Tensor)},
        "config": vars(args),
    }, final_path)

    # Summary
    summary = {
        "config": {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, list, tuple))},
        "params_M": float(n_params / 1e6),
        "best": {
            "epoch": best_epoch,
            "psnr": best_psnr if not math.isinf(best_psnr) else float("nan"),
            "ssim": history["eval_ssim"][best_epoch - 1] if best_epoch > 0 else float("nan"),
            "ckpt": best_path,
        },
        "final": {
            "epoch": args.num_epoch,
            "psnr": history["eval_psnr"][-1],
            "ssim": history["eval_ssim"][-1],
            "ckpt": final_path,
        },
        "total_wall_seconds": float(sum(history["wall_seconds"])),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"wrote {npz_path}")
    _log(f"wrote {best_path}  (best PSNR {best_psnr:.3f} at ep {best_epoch})")
    _log(f"wrote {final_path}")
    _log(f"wrote {summary_path}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
