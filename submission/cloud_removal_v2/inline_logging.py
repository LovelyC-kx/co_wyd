"""
Inline logging helpers for 70-epoch v2-A run.

Two measurements, both cheap and computed at eval_every boundaries:

1. **BN drift** (supports Claim C16 sub-claims SC-16a, SC-16b, SC-16c).
   For each plane, iterate TDBN-class parameters and record
     - ||lambda - 1||_inf   (SC-16a)
     - ||beta||_inf          (SC-16b)
     - per-layer + aggregate stats for plane-level and intra-plane variance.

2. **Cross-plane cosine-similarity of weight deltas** (Seo24 Fig 13 proxy).
   For each eval boundary, snapshot all plane weights. On the NEXT eval
   boundary, compute per-plane delta (curr - prev) and measure pairwise
   cosine between plane pairs. Per-layer granularity is optional; default
   is whole-model flattened vector.

Usage (from run_smoke.py)::

    drift_logger = BnDriftLogger(num_planes)
    cos_logger = CosineSimLogger(num_planes)

    for epoch in 1..T:
        train_loss = constellation.train_one_round(scheme)
        if do_eval_this_epoch:
            drift_sample = drift_logger.snapshot(constellation, epoch)
            cos_sample = cos_logger.snapshot(constellation, epoch)

    # At end of training:
    history["bn_drift"] = drift_logger.get_history()
    history["cos_sim"] = cos_logger.get_history()

Both loggers are stateless w.r.t. training code — they only read weights.

Storage cost: drift ~20 KB/eval; cos_sim ~2 KB/eval. Negligible vs full
checkpoint save. Compute cost: ~200 ms/eval on CPU (one plane snapshot +
pairwise cosines).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# BN-class parameter identification (mirror of cloud_removal_v1/aggregation.py)
# ---------------------------------------------------------------------------

def _is_bn_param_key(key: str) -> bool:
    """Return True if this parameter belongs to a BN/TDBN class layer.

    We look for the tokens 'bn', 'norm', '_bn' in the path AND require it to
    end in .weight / .bias (the trainable affine parameters).
    """
    if not (key.endswith(".weight") or key.endswith(".bias")):
        return False
    lc = key.lower()
    return any(tok in lc for tok in ("bn", "norm"))


def _flatten_state_dict(sd: Dict[str, torch.Tensor],
                        bn_only: bool = False) -> torch.Tensor:
    """Flatten all float-tensor values in sd into one 1-D vector.

    Skips integer-dtype tensors (e.g. num_batches_tracked) and NaN/Inf.
    When bn_only=True, only BN-class parameters are included.
    """
    chunks: List[torch.Tensor] = []
    for k in sorted(sd.keys()):
        v = sd[k]
        if not isinstance(v, torch.Tensor):
            continue
        if not v.is_floating_point():
            continue
        if bn_only and not _is_bn_param_key(k):
            continue
        chunks.append(v.detach().flatten().float().cpu())
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


# ---------------------------------------------------------------------------
# BN drift logger
# ---------------------------------------------------------------------------

class BnDriftLogger:
    """Records per-plane, per-layer ||lambda - 1||_inf and ||beta||_inf.

    Uses plane-0 satellite as the representative (intra-plane averaging keeps
    all sats identical at eval boundary).
    """

    def __init__(self, num_planes: int) -> None:
        self.num_planes = num_planes
        self._per_epoch: List[Dict[str, Any]] = []

    def snapshot(self, constellation: Any, epoch: int) -> Dict[str, Any]:
        per_plane: List[Dict[str, Dict[str, float]]] = []
        for p in range(self.num_planes):
            sat = constellation.planes[p][0]
            sd = sat.get_weights(cpu=True)
            layer_stats: Dict[str, Dict[str, float]] = {}
            for key, val in sd.items():
                if not _is_bn_param_key(key):
                    continue
                if not isinstance(val, torch.Tensor) or not val.is_floating_point():
                    continue
                v = val.detach().flatten().float().cpu()
                stat: Dict[str, float]
                if key.endswith(".weight"):
                    # Trainable scale (lambda in TDBN notation). Deviation
                    # from init=1 is the diagnostic quantity for SC-16a.
                    stat = {
                        "lambda_minus_1_inf": float((v - 1.0).abs().max()),
                        "lambda_minus_1_l2":  float(torch.norm(v - 1.0, p=2)),
                        "lambda_mean":        float(v.mean()),
                        "lambda_std":         float(v.std(unbiased=False)),
                    }
                elif key.endswith(".bias"):
                    stat = {
                        "beta_inf":           float(v.abs().max()),
                        "beta_l2":            float(torch.norm(v, p=2)),
                        "beta_mean":          float(v.mean()),
                        "beta_std":           float(v.std(unbiased=False)),
                    }
                else:
                    continue
                layer_stats[key] = stat
            per_plane.append(layer_stats)

        record = {"epoch": int(epoch), "per_plane": per_plane}
        self._per_epoch.append(record)
        return record

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._per_epoch)


# ---------------------------------------------------------------------------
# Cross-plane cosine-similarity of weight deltas
# ---------------------------------------------------------------------------

class CosineSimLogger:
    """Pairwise cosine(delta_plane_i, delta_plane_j) between successive eval
    boundaries. Follows Seo24 Fig 13 methodology for operational
    cross-client gradient dissimilarity proxy.

    At each snapshot(), extracts one flat weight-vector per plane. On the
    second and subsequent calls, computes delta = current - previous and
    measures cosine similarity between all (num_planes choose 2) plane
    pairs, recorded by (i, j) with i < j.
    """

    def __init__(self, num_planes: int,
                 bn_only: bool = False,
                 output_head_only: bool = False) -> None:
        self.num_planes = num_planes
        self.bn_only = bn_only
        self.output_head_only = output_head_only
        self._prev_flat: Optional[List[torch.Tensor]] = None
        self._per_epoch: List[Dict[str, Any]] = []

    def _snapshot_planes(self, constellation: Any) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for p in range(self.num_planes):
            sat = constellation.planes[p][0]
            sd = sat.get_weights(cpu=True)
            if self.output_head_only:
                sd = {k: v for k, v in sd.items() if "main_out" in k or "head" in k}
            flat = _flatten_state_dict(sd, bn_only=self.bn_only)
            out.append(flat)
        return out

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        na = float(torch.norm(a, p=2))
        nb = float(torch.norm(b, p=2))
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float(torch.dot(a, b) / (na * nb))

    def snapshot(self, constellation: Any, epoch: int) -> Dict[str, Any]:
        curr = self._snapshot_planes(constellation)
        record: Dict[str, Any] = {"epoch": int(epoch), "pairs": []}
        if self._prev_flat is not None:
            # Compute delta and pairwise cosine
            deltas = [curr[p] - self._prev_flat[p]
                      for p in range(self.num_planes)]
            pairs = []
            for i in range(self.num_planes):
                for j in range(i + 1, self.num_planes):
                    pairs.append({
                        "i":   int(i),
                        "j":   int(j),
                        "cos": self._cosine(deltas[i], deltas[j]),
                    })
            record["pairs"] = pairs
            # Summary stats
            cos_values = [p["cos"] for p in pairs
                          if not (p["cos"] != p["cos"])]  # drop NaN
            if cos_values:
                record["cos_mean"] = float(sum(cos_values) / len(cos_values))
                record["cos_min"]  = float(min(cos_values))
                record["cos_max"]  = float(max(cos_values))
        self._prev_flat = curr
        self._per_epoch.append(record)
        return record

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._per_epoch)
