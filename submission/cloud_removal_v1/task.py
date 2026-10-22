"""
Per-satellite cloud-removal training task (v1).

One VLIFNet + one persistent AdamW optimizer + one DataLoader per
satellite.  Mandatory reset_net before AND after every forward pass
(per SpikingJelly multi-step tutorial) — see `local_training`.

Design
------
1. `self.model` is a VLIFNet, NEVER deepcopy'd (deepcopy carries
   SpikingJelly memory buffers into the aggregation path, corrupting
   the average).  Weight I/O goes through state_dict instead.
2. `get_weights(cpu)` returns a tensor-aware copy: tensors are detached
   and cloned; non-tensor memory sentinels (scalar floats, None) are
   passed through verbatim — aggregation.py handles them.
3. `apply_global_weights(...)` goes through aggregation.apply_aggregated
   so BN-local (FedBN) semantics are enforced with a single flag flip.
4. The AdamW optimizer is built ONCE and re-used across global rounds;
   the momentum / second-moment buffers persist locally (client-side),
   matching the Adaptive Federated Optimization (ICLR'21) convention.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import functional

from .models import build_vlifnet
from .aggregation import apply_aggregated
from .dataset import seed_worker


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """ℒ(x, y) = mean(√((x - y)² + ε²)).

    Twice-differentiable L¹-approximation (Lai et al. 2018).  Preserves
    the L-smoothness assumption required by Thm 2 of the FLSNN paper.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


def _gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class SSIMLoss(nn.Module):
    """Standard SSIM on [0, 1] images; loss = 1 - SSIM.

    Window is lazily built per-(channel, device, dtype) call; caching is
    avoided on purpose so the module remains DataParallel-safe.  For v1
    this is called once per local step — the overhead is negligible.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2

    def _build_window(self, channel: int, device, dtype) -> torch.Tensor:
        g1d = _gaussian_window(self.window_size, self.sigma, device).to(dtype)
        w2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        return w2d.expand(channel, 1, self.window_size, self.window_size).contiguous()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"SSIMLoss expects [B, C, H, W]; got {tuple(x.shape)}"
        B, C, H, W = x.shape
        window = self._build_window(C, x.device, x.dtype)
        pad = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=pad, groups=C)
        mu_y = F.conv2d(y, window, padding=pad, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

        ssim_num = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        ssim_den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        return 1.0 - (ssim_num / ssim_den).mean()


class CloudLoss(nn.Module):
    """v1 composite loss: Charbonnier + λ·(1 - SSIM).  λ = 0.1 by default
    so Charbonnier dominates and the assumption for Thm 2 holds; SSIM
    acts as an auxiliary perceptual regulariser."""

    def __init__(self, ssim_weight: float = 0.1, eps: float = 1e-3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim = SSIMLoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.charbonnier(pred, target) + self.ssim_weight * self.ssim(pred, target)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def _reset_snn(model: nn.Module) -> None:
    functional.reset_net(model)


class CloudRemovalSNNTask:
    """One satellite's complete trainer.

    Parameters
    ----------
    args : argparse.Namespace
        Required fields (subset):
          lr, wd, min_lr, warmup_epochs,
          train_batch_size, num_workers,
          local_iters, T, vlif_dim, en_blocks, de_blocks,
          clip_grad, vlif_backend,
          ssim_weight, charbonnier_eps, bn_local (bool).
    local_dataset : torch.utils.data.Dataset
    init_state_dict : dict | None
    device : str | torch.device
    """

    def __init__(self,
                 args,
                 local_dataset: Dataset,
                 init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
                 device: str = "cuda"):
        self.args = args
        self.device = torch.device(device)

        backend = getattr(args, "vlif_backend", "torch")
        bn_variant = getattr(args, "bn_variant", "tdbn")
        backbone = getattr(args, "backbone", "snn")
        if backbone == "plain":
            from .models import build_plain_unet
            self.model = build_plain_unet(
                dim=args.vlif_dim,
                en_blocks=tuple(args.en_blocks[:3]),
                de_blocks=tuple(args.de_blocks[:3]),
                inp_channels=3,
                out_channels=3,
            ).to(self.device)
        else:
            self.model = build_vlifnet(
                dim=args.vlif_dim,
                en_num_blocks=tuple(args.en_blocks),
                de_num_blocks=tuple(args.de_blocks),
                T=args.T,
                use_refinement=False,
                inp_channels=3,
                out_channels=3,
                backend=backend,
                bn_variant=bn_variant,
                backbone=backbone,
            ).to(self.device)

        if init_state_dict is not None:
            # SpikingJelly's MemoryModule accepts scalar / None memory
            # entries via its custom _load_from_state_dict override.
            self.model.load_state_dict(init_state_dict, strict=True)

        self.local_dataset = local_dataset
        nw = getattr(args, "num_workers", 0)
        # B-BN-1 FIX: drop_last=True prevents the trailing single-sample batch
        # under v2-A's Dirichlet(α=0.1) partition.  Example: a client that
        # Dirichlet-hit exactly 5 samples + train_batch_size=4 yields batches
        # [4, 1].  A batch of 1 through BatchNorm2d in training mode drives
        # running_var → 0 (biased estimator with n=1), which amplifies the
        # NEXT forward pass by ~316× via x / sqrt(var+eps).  With
        # min_samples_per_client ≥ train_batch_size, drop_last=True costs
        # ≤ (batch_size-1) samples per client per epoch (tiny) and keeps
        # every batch well-conditioned for BN.
        loader_kwargs = dict(
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=nw,
        )
        if nw > 0:
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["worker_init_fn"] = seed_worker
        self.loader = DataLoader(local_dataset, **loader_kwargs)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=getattr(args, "wd", 0.0),
        )
        self.clip_grad = getattr(args, "clip_grad", 1.0)

        self.criterion = CloudLoss(
            ssim_weight=args.ssim_weight,
            eps=args.charbonnier_eps,
        ).to(self.device)

        self.last_train_loss: float = math.nan
        self.last_train_charbonnier: float = math.nan
        self.last_train_ssim: float = math.nan
        self.global_round: int = 0

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def get_weights(self, cpu: bool = True) -> Dict[str, torch.Tensor]:
        """Detached copy of the model state-dict.

        Tensors: detached + cloned (+ moved to CPU if requested).
        Non-tensors (SpikingJelly memory scalars / None): passed through.
        """
        sd = self.model.state_dict()
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().clone() if cpu else v.detach().clone()
            else:
                out[k] = v
        return out

    def apply_global_weights(self,
                             global_state: Dict[str, torch.Tensor],
                             bn_local: bool = False) -> None:
        """Copy aggregated weights into the local model in place.

        bn_local=True skips BN-tagged keys (FedBN).  Always issues a
        reset_net afterwards: the just-loaded weights are a noisy
        average of many clients, and leaving stale membrane potentials
        around would pollute the next forward pass.
        """
        apply_aggregated(self.model.state_dict(), global_state, bn_local=bn_local)
        _reset_snn(self.model)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _cosine_lr(self, step: int, total_steps: int, warmup_steps: int) -> float:
        base_lr = self.args.lr
        min_lr = getattr(self.args, "min_lr", 1e-7)
        if step < warmup_steps:
            return base_lr * (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = max(0.0, min(1.0, t))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    def local_training(self,
                       total_global_rounds: int,
                       warmup_rounds: int = 3,
                       lr_scale: float = 1.0) -> Tuple[float, float, float]:
        """Run `args.local_iters` passes over the local dataset.

        Returns (avg_total_loss, avg_charbonnier, avg_ssim_loss).

        lr_scale: scheme-dependent LR multiplier applied AFTER cosine schedule.
            FLSNN's revised_constellation.py:204-205 boosts RelaySum LR by
            2.093x to compensate for its delayed-aggregation noise
            (see v2 §25.11 audit). Default 1.0 = no scaling (Gossip/AllReduce).
        """
        self.model.train()

        lr = self._cosine_lr(self.global_round, total_global_rounds, warmup_rounds)
        lr = lr * float(lr_scale)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        n_batches = 0
        sum_loss = sum_ch = sum_ss = 0.0

        for _ in range(self.args.local_iters):
            for cloudy, clear in self.loader:
                cloudy = cloudy.to(self.device, non_blocking=True)
                clear  = clear.to(self.device,  non_blocking=True)

                _reset_snn(self.model)
                self.optimizer.zero_grad(set_to_none=True)

                pred = self.model(cloudy)
                loss_ch = self.criterion.charbonnier(pred, clear)
                loss_ss = self.criterion.ssim(pred, clear)
                loss = loss_ch + self.criterion.ssim_weight * loss_ss

                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                _reset_snn(self.model)

                n_batches += 1
                sum_loss += loss.item()
                sum_ch   += loss_ch.item()
                sum_ss   += loss_ss.item()

        if n_batches == 0:
            return math.nan, math.nan, math.nan

        self.last_train_loss        = sum_loss / n_batches
        self.last_train_charbonnier = sum_ch   / n_batches
        self.last_train_ssim        = sum_ss   / n_batches
        return self.last_train_loss, self.last_train_charbonnier, self.last_train_ssim

    @torch.no_grad()
    def forward_on(self, cloudy: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        _reset_snn(self.model)
        out = self.model(cloudy.to(self.device, non_blocking=True))
        _reset_snn(self.model)
        return out

    def cleanup_between_rounds(self) -> None:
        """Advance round counter.  GPU cache eviction is done once per
        round at the constellation level to avoid 50× redundant calls."""
        self.global_round += 1
