"""
Federated aggregation primitives.

Pure tensor-level state_dict operations.  No knowledge of the satellite
abstraction — that lives in constellation.py.

Key design decisions
--------------------
1. Every primitive is *tensor-aware*.  SpikingJelly's MemoryModule
   writes non-tensor values (scalar floats like `v = 0.0`, or None
   sentinels) to state_dict() alongside real parameters; aggregation
   must skip them instead of calling .detach() / .clone() / .to().

2. A single `is_bn_key(name)` function is the source of truth for
   Batch-Norm-like parameter detection (torch.nn BatchNorm, LayerNorm,
   GroupNorm, spikingjelly ThresholdDependentBatchNorm / TemporalBN).

3. `average_state_dicts(..., bn_local=True)` implements FedBN (ICLR'21):
   every BN-tagged key is taken verbatim from states[0] without
   averaging across clients.

4. `apply_aggregated(target, global_sd, bn_local=True)` is symmetric
   with average_state_dicts: the BN-skip logic lives on BOTH sides so
   a caller that forgets one still preserves FedBN semantics.

The __main__ block runs four self-tests (no GPU required).
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# BN key classification
# ---------------------------------------------------------------------------

# A state_dict key is a dot-separated path (e.g. "encoder.block.bn1.weight").
# A key belongs to a BN / normaliser if ANY path component (bounded by '.',
# start-of-string, or end-of-string) is one of:
#
#     bn, bn0, bn1, ..., _bn, _bn0, _bn1, ...        torch.nn BatchNorm
#     bns                                             spikingjelly TemporalBN
#     norm, norm0, norm1, ...                         GroupNorm / LayerNorm / InstanceNorm
#     threshold                                       spikingjelly TDBN scaling
#
# OR any path component starts with:
#
#     running_                 torch.nn BatchNorm running_mean / running_var
#     num_batches_tracked      torch.nn BatchNorm counter
#
# The previous substring-match approach missed keys where BN sits at the
# very root (e.g. "bn.weight"), because a leading-"." was expected.
_BN_RE = re.compile(
    r"(?:^|\.)(?:bn\d*|_bn\d*|norm\d*|bns\d*|threshold)(?:\.|$)"
    r"|(?:^|\.)running_"
    r"|(?:^|\.)num_batches_tracked(?:\.|$)",
    re.IGNORECASE,
)


def is_bn_key(key: str) -> bool:
    """True iff `key` names a BN / normaliser parameter or running-stat.

    Handles:
      * torch.nn.BatchNorm{1d,2d,3d}.{weight,bias,running_mean,running_var,
                                      num_batches_tracked}
      * spikingjelly ThresholdDependentBatchNorm2d (.threshold)
      * spikingjelly TemporalBN (bns.0.* / bns.1.* ...)
      * GroupNorm / LayerNorm / InstanceNorm (.norm[.N].*)
      * BN attached at the root (bn.weight, bn1.running_mean, ...)

    Does NOT match Conv2d / Linear parameters named .conv, .fc, .proj,
    .output, .aux_head, etc.
    """
    return _BN_RE.search(key) is not None


# ---------------------------------------------------------------------------
# Tensor-aware helpers
# ---------------------------------------------------------------------------

def _is_tensor(x) -> bool:
    """True iff x is a torch.Tensor.  Used to skip SpikingJelly's scalar /
    None memory sentinels that appear alongside tensor params in state_dict()."""
    return isinstance(x, torch.Tensor)


def state_add(acc: Dict[str, torch.Tensor],
              other: Dict[str, torch.Tensor],
              *, scale: float = 1.0) -> None:
    """acc ← acc + scale · other (per-key, in place).

    Non-tensor keys are skipped.  Integer-dtype tensors (typically
    ``num_batches_tracked`` in nn.BatchNorm*) are added WITHOUT the
    float ``scale`` multiplier, since PyTorch refuses to cast float →
    long in-place.  Counters don't need weighting anyway — they just
    need to be propagated.
    """
    for k, v in other.items():
        a = acc.get(k)
        if _is_tensor(a) and _is_tensor(v):
            if a.is_floating_point():
                a.add_(v, alpha=scale)
            else:
                a.add_(v)


def state_mul(sd: Dict[str, torch.Tensor], scale: float) -> None:
    """sd ← scale · sd (in place).  Non-tensor + integer keys skipped."""
    for v in sd.values():
        if _is_tensor(v) and v.is_floating_point():
            v.mul_(scale)


def state_div(sd: Dict[str, torch.Tensor], scale: float) -> None:
    """sd ← sd / scale (in place).  Non-tensor + integer keys skipped.

    BN's ``num_batches_tracked`` is int64; dividing a long by a float
    would fail PyTorch's in-place dtype check.  Counters are semantically
    not averageable anyway — they pass through verbatim.
    """
    assert scale != 0, "state_div: division by zero"
    inv = 1.0 / float(scale)
    for v in sd.values():
        if _is_tensor(v) and v.is_floating_point():
            v.mul_(inv)


def zeros_like_state(sd: Dict[str, torch.Tensor],
                     device: Optional[torch.device] = None,
                     ) -> Dict[str, torch.Tensor]:
    """zeros_like for every tensor value (optionally on a given device);
    non-tensor values are passed through verbatim.

    Used to initialise RelaySum relay buffers, which must live on the
    same device as the per-satellite weights they will be summed with.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if _is_tensor(v):
            if device is None:
                out[k] = torch.zeros_like(v)
            else:
                out[k] = torch.zeros_like(v, device=device)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------

def average_state_dicts(
    states: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
    *,
    bn_local: bool = False,
    amp_share: bool = False,      # v3 placeholder
) -> Dict[str, torch.Tensor]:
    """Weighted average of a list of state-dicts.

    Parameters
    ----------
    states : list[dict]
        All states must share the same keys and per-key shapes.  Values
        may be torch.Tensor, Python scalars (e.g. SpikingJelly memory
        `v = 0.0`), or None.  Non-tensor values are taken from states[0].
    weights : list[float] | None
        If None, uniform average.  Must sum to > 0.
    bn_local : bool
        v2 FedBN knob.  When True, BN-tagged keys are taken verbatim
        from states[0] (no averaging).
    amp_share : bool
        v3 HarmoFL placeholder.  v1 must pass False.

    Returns
    -------
    A fresh state-dict; input tensors are NOT mutated.  Output tensors
    live on the same device as states[0] values.
    """
    if amp_share:
        raise NotImplementedError(
            "amp_share is scheduled for v3 (HarmoFL).  v1/v2 must pass amp_share=False.")

    assert len(states) > 0, "average_state_dicts: empty input"
    ref_keys = list(states[0].keys())
    for i, sd in enumerate(states[1:], start=1):
        if list(sd.keys()) != ref_keys:
            raise KeyError(
                f"state_dict[{i}] keys differ from state_dict[0]; aggregation "
                f"would silently drop parameters.  Symmetric difference: "
                f"{set(sd.keys()).symmetric_difference(ref_keys)}")

    if weights is None:
        weights = [1.0] * len(states)
    assert len(weights) == len(states)
    total_w = float(sum(weights))
    assert total_w > 0, "average_state_dicts: weights sum to 0"

    out: Dict[str, torch.Tensor] = {}
    for k in ref_keys:
        v0 = states[0][k]
        # Non-tensor entries (SpikingJelly float scalars, None) are passed
        # through; BN-tagged tensors are also preserved verbatim when
        # bn_local=True.
        if not _is_tensor(v0) or (bn_local and is_bn_key(k)):
            out[k] = v0.detach().clone() if _is_tensor(v0) else v0
            continue
        # Integer-dtype tensors (BN num_batches_tracked) are counters;
        # averaging them by float weights would fail a PyTorch in-place
        # dtype cast.  Take states[0] verbatim.
        if not v0.is_floating_point():
            out[k] = v0.detach().clone()
            continue
        acc = v0.detach().clone() * float(weights[0])
        for i in range(1, len(states)):
            vi = states[i][k]
            if _is_tensor(vi):
                acc.add_(vi, alpha=float(weights[i]))
        acc.div_(total_w)
        out[k] = acc
    return out


# ---------------------------------------------------------------------------
# In-place apply
# ---------------------------------------------------------------------------

def apply_aggregated(
    target_sd: Dict[str, torch.Tensor],
    global_sd: Dict[str, torch.Tensor],
    *,
    bn_local: bool = False,
) -> None:
    """Copy `global_sd` into `target_sd` in place (parameter aliases).

    * Non-tensor entries in global_sd are ignored (the target's own
      SpikingJelly memory values are left unchanged; they will be
      re-set to 0 by the next functional.reset_net(...) anyway).
    * Keys in target_sd that don't appear in global_sd are left
      unchanged.  This tolerates state_dicts that were produced by
      an earlier model version with a slightly different key set.
    * bn_local=True skips BN-tagged keys (FedBN).
    """
    for k, v in global_sd.items():
        if bn_local and is_bn_key(k):
            continue
        dst = target_sd.get(k)
        if dst is None or not _is_tensor(v) or not _is_tensor(dst):
            continue
        dst.copy_(v.to(dst.device, dtype=dst.dtype, non_blocking=True))


# ---------------------------------------------------------------------------
# Self-tests (run with `python aggregation.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # ---- 1. is_bn_key classification ------------------------------------
    samples = [
        # True
        ("encoder_level1.initial_residual.bn1.threshold",        True),
        ("encoder_level1.initial_residual.bn1.weight",           True),
        ("encoder_level1.initial_residual.bn1.bias",             True),
        ("encoder_level1.initial_residual.bn1.running_mean",     True),
        ("encoder_level1.initial_residual.bn1.running_var",      True),
        ("encoder_level1.initial_residual.bn1.num_batches_tracked", True),
        ("fre_mlp.norm.weight",                                  True),
        ("fre_mlp.norm.bias",                                    True),
        # False
        ("patch_embed.proj.weight",                              False),
        ("encoder_level1.initial_residual.conv1.weight",         False),
        ("encoder_level1.initial_residual.high_freq_scale_1",    False),
        ("encoder_level1.initial_residual.cross_scale_gate",     False),
        ("output.0.weight",                                      False),
        ("aux_head_level2.weight",                               False),
    ]
    bad = 0
    for key, expected in samples:
        got = is_bn_key(key)
        if got != expected:
            bad += 1
            print(f"FAIL {key}: expected={expected} got={got}", file=sys.stderr)
    assert bad == 0, f"{bad} is_bn_key failures"
    print("[1/4] is_bn_key: 14/14 sample keys classified correctly")

    # ---- 2. average_state_dicts + non-tensor entries --------------------
    sd_a = {
        "conv.weight":       torch.ones(2, 2),
        "lif.v":             0.0,        # SpikingJelly un-init memory
        "lif.spike":         None,       # sentinel
        "bn.weight":         torch.tensor([1.0, 1.0]),
        "bn.running_mean":   torch.tensor([0.5, 0.5]),
    }
    sd_b = {k: (v.clone() * 2.0 if _is_tensor(v) else v) for k, v in sd_a.items()}
    out = average_state_dicts([sd_a, sd_b])
    assert torch.allclose(out["conv.weight"], torch.full((2, 2), 1.5))
    assert out["lif.v"] == 0.0
    assert out["lif.spike"] is None
    assert torch.allclose(out["bn.weight"], torch.full((2,), 1.5))
    assert torch.allclose(out["bn.running_mean"], torch.full((2,), 0.75))
    print("[2/4] average_state_dicts: non-tensor entries preserved, tensor averaged")

    # ---- 3. bn_local=True skips BN keys ---------------------------------
    out_bnl = average_state_dicts([sd_a, sd_b], bn_local=True)
    assert torch.allclose(out_bnl["conv.weight"], torch.full((2, 2), 1.5)), \
        "conv.weight should still be averaged when bn_local=True"
    assert torch.allclose(out_bnl["bn.weight"], torch.tensor([1.0, 1.0])), \
        "bn.weight should be states[0] verbatim when bn_local=True"
    assert torch.allclose(out_bnl["bn.running_mean"], torch.tensor([0.5, 0.5]))
    print("[3/4] average_state_dicts(bn_local=True): BN keys kept from states[0]")

    # ---- 4. apply_aggregated tolerates non-tensor entries ---------------
    tgt = {
        "conv.weight":  torch.zeros(2, 2),
        "lif.v":        0.0,
        "lif.spike":    None,
        "only_in_tgt":  torch.ones(1),
    }
    src = {
        "conv.weight":  torch.full((2, 2), 9.0),
        "lif.v":        42.0,    # should be ignored (v is preserved at 0.0)
        "lif.spike":    None,
        "only_in_src":  torch.ones(1),   # should be ignored (not in target)
    }
    apply_aggregated(tgt, src)
    assert torch.allclose(tgt["conv.weight"], torch.full((2, 2), 9.0))
    assert tgt["lif.v"] == 0.0, \
        f"lif.v should be left untouched on target, got {tgt['lif.v']}"
    assert torch.allclose(tgt["only_in_tgt"], torch.ones(1)), \
        "target-only keys must not be modified"
    print("[4/4] apply_aggregated: scalar/None/missing keys handled correctly")

    print("\nALL 4 AGGREGATION SELF-TESTS PASSED.")
