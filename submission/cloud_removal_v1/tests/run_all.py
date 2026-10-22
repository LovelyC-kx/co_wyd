"""Pure-Python self-tests for v1.

These tests DO NOT require torch's CUDA, spikingjelly, or any image data.
They cover the tensor-level aggregation logic and the dataset partition
arithmetic — the parts most likely to silently corrupt FL training.

Run with:
    python -m cloud_removal_v1.tests.run_all

Requires only: numpy, torch (cpu-only OK).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))

import numpy as np
import torch

from cloud_removal_v1.aggregation import (
    is_bn_key,
    average_state_dicts,
    apply_aggregated,
    zeros_like_state,
    state_div,
)
from cloud_removal_v1.dataset import (
    build_client_partitions,
    build_plane_satellite_partitions,
)


PASS = "\033[32mOK \033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures = 0


def _check(cond: bool, msg: str) -> None:
    global _failures
    if cond:
        print(f"{PASS} {msg}")
    else:
        _failures += 1
        print(f"{FAIL} {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# is_bn_key
# ---------------------------------------------------------------------------

def test_is_bn_key():
    print("\n=== 1. is_bn_key classification ===")
    positive = [
        "encoder_level1.initial_residual.bn1.threshold",
        "encoder_level1.initial_residual.bn1.weight",
        "encoder_level1.initial_residual.bn1.bias",
        "encoder_level1.initial_residual.bn1.running_mean",
        "encoder_level1.initial_residual.bn1.running_var",
        "encoder_level1.initial_residual.bn1.num_batches_tracked",
        "fre_mlp.norm.weight",
        "fre_mlp.norm.bias",
    ]
    negative = [
        "patch_embed.proj.weight",
        "encoder_level1.initial_residual.conv1.weight",
        "encoder_level1.initial_residual.high_freq_scale_1",
        "encoder_level1.initial_residual.cross_scale_gate",
        "output.0.weight",
        "aux_head_level2.weight",
    ]
    for k in positive:
        _check(is_bn_key(k) is True, f"is_bn_key({k!r}) → True")
    for k in negative:
        _check(is_bn_key(k) is False, f"is_bn_key({k!r}) → False")


# ---------------------------------------------------------------------------
# average_state_dicts — tensor + non-tensor
# ---------------------------------------------------------------------------

def test_average_state_dicts():
    print("\n=== 2. average_state_dicts preserves non-tensor entries ===")

    def _sd(a, b):
        return {
            "conv.weight":     torch.full((2, 2), float(a)),
            "lif.v":           0.0,      # SJ un-initialised memory
            "lif.spike":       None,
            "bn.weight":       torch.tensor([float(a), float(b)]),
            "bn.running_mean": torch.tensor([0.5, 0.5]) * float(a),
        }

    s0 = _sd(1.0, 2.0)
    s1 = _sd(3.0, 4.0)
    s2 = _sd(5.0, 6.0)
    out = average_state_dicts([s0, s1, s2])

    _check(torch.allclose(out["conv.weight"], torch.full((2, 2), 3.0)),
           "non-BN tensor averaged correctly (1+3+5)/3=3")
    _check(out["lif.v"] == 0.0, "scalar 0.0 preserved")
    _check(out["lif.spike"] is None, "None preserved")
    _check(torch.allclose(out["bn.weight"], torch.tensor([3.0, 4.0])),
           "BN weight averaged (default bn_local=False)")


def test_bn_local():
    print("\n=== 3. average_state_dicts(bn_local=True) keeps states[0] verbatim ===")

    s0 = {
        "conv.weight":     torch.ones(2, 2),
        "bn.weight":       torch.tensor([1.0, 1.0]),
        "bn.running_mean": torch.tensor([0.5, 0.5]),
    }
    s1 = {
        "conv.weight":     torch.full((2, 2), 3.0),
        "bn.weight":       torch.tensor([9.0, 9.0]),
        "bn.running_mean": torch.tensor([7.0, 7.0]),
    }
    out = average_state_dicts([s0, s1], bn_local=True)
    _check(torch.allclose(out["conv.weight"], torch.full((2, 2), 2.0)),
           "non-BN conv still averaged when bn_local=True")
    _check(torch.allclose(out["bn.weight"], torch.tensor([1.0, 1.0])),
           "bn.weight kept from states[0] when bn_local=True")
    _check(torch.allclose(out["bn.running_mean"], torch.tensor([0.5, 0.5])),
           "bn.running_mean kept from states[0] when bn_local=True")


# ---------------------------------------------------------------------------
# apply_aggregated
# ---------------------------------------------------------------------------

def test_apply_aggregated():
    print("\n=== 4. apply_aggregated skips non-tensor + mismatched keys ===")
    tgt = {
        "conv.weight":   torch.zeros(2, 2),
        "lif.v":         0.0,
        "lif.spike":     None,
        "only_in_tgt":   torch.ones(1),
    }
    src = {
        "conv.weight":   torch.full((2, 2), 9.0),
        "lif.v":         42.0,             # must be ignored
        "lif.spike":     None,
        "only_in_src":   torch.ones(1),    # not in tgt → must be ignored
    }
    apply_aggregated(tgt, src)
    _check(torch.allclose(tgt["conv.weight"], torch.full((2, 2), 9.0)),
           "conv.weight copied")
    _check(tgt["lif.v"] == 0.0,                                   "scalar 0.0 untouched")
    _check(torch.allclose(tgt["only_in_tgt"], torch.ones(1)),      "target-only key untouched")


# ---------------------------------------------------------------------------
# zeros_like_state  + device override
# ---------------------------------------------------------------------------

def test_zeros_like_state():
    print("\n=== 5. zeros_like_state preserves scalars/None, on-device for tensors ===")
    sd = {
        "conv.weight": torch.ones(2, 2),
        "lif.v":       0.0,
        "lif.spike":   None,
    }
    z = zeros_like_state(sd)
    _check(torch.all(z["conv.weight"] == 0), "tensor zeroed")
    _check(z["lif.v"] == 0.0,                 "scalar preserved")
    _check(z["lif.spike"] is None,            "None preserved")

    # device override — CPU device should still work
    z_cpu = zeros_like_state(sd, device=torch.device("cpu"))
    _check(z_cpu["conv.weight"].device.type == "cpu", "device override honoured")


# ---------------------------------------------------------------------------
# state_div divides by count, skips non-tensors
# ---------------------------------------------------------------------------

def test_state_div():
    print("\n=== 6. state_div handles mixed entries ===")
    sd = {
        "a": torch.full((2,), 6.0),
        "b": 0.0,
        "c": None,
        "d": torch.tensor([2.0, 4.0]),
    }
    state_div(sd, 2)
    _check(torch.allclose(sd["a"], torch.full((2,), 3.0)), "tensor a divided")
    _check(sd["b"] == 0.0,                                  "scalar b untouched")
    _check(sd["c"] is None,                                 "None c untouched")
    _check(torch.allclose(sd["d"], torch.tensor([1.0, 2.0])), "tensor d divided")


# ---------------------------------------------------------------------------
# Federated partitioning arithmetic
# ---------------------------------------------------------------------------

class _FakeDataset:
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return idx


def test_partitioning():
    print("\n=== 7. build_plane_satellite_partitions arithmetic ===")
    ds = _FakeDataset(n=534)    # CUHK-CR1 train
    parts = build_plane_satellite_partitions(ds, num_planes=5, sats_per_plane=10,
                                             mode="iid", seed=0)
    sizes = [[len(s) for s in row] for row in parts]
    total = sum(sum(r) for r in sizes)
    _check(total == 534, f"all samples accounted for: total={total}")

    flat = [idx for row in parts for s in row for idx in s.indices]
    _check(len(set(flat)) == 534, "partitions are disjoint (no duplicate indices)")
    _check(len(parts) == 5,                "shape — outer list = num_planes")
    _check(all(len(r) == 10 for r in parts), "shape — inner list = sats_per_plane")

    per_client = [len(s) for row in parts for s in row]
    _check(min(per_client) >= 10, f"every client has ≥10 samples (min={min(per_client)})")
    _check(max(per_client) <= 11, f"every client has ≤11 samples (max={max(per_client)})")


def test_partitioning_dirichlet_not_implemented():
    print("\n=== 8. dirichlet_cluster raises NotImplementedError in v1 ===")
    ds = _FakeDataset(100)
    try:
        build_client_partitions(ds, 5, mode="dirichlet_cluster")
    except NotImplementedError:
        _check(True, "dirichlet_cluster correctly raises NotImplementedError")
    except Exception as e:
        _check(False, f"expected NotImplementedError, got {type(e).__name__}: {e}")
    else:
        _check(False, "dirichlet_cluster did NOT raise")


# ---------------------------------------------------------------------------
# AST parse check for all v1 modules (runs without spikingjelly/torch GPU)
# ---------------------------------------------------------------------------

def test_ast_parse():
    print("\n=== 9. AST parse all v1 modules ===")
    import ast
    root = Path(__file__).resolve().parent.parent
    files = list(root.rglob("*.py"))
    _check(len(files) >= 10, f"found {len(files)} .py files to check")
    for f in files:
        try:
            ast.parse(f.read_text(encoding="utf-8"))
            print(f"{PASS} {f.relative_to(root)}")
        except SyntaxError as e:
            _check(False, f"{f.relative_to(root)}: {e}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    test_is_bn_key()
    test_average_state_dicts()
    test_bn_local()
    test_apply_aggregated()
    test_zeros_like_state()
    test_state_div()
    test_partitioning()
    test_partitioning_dirichlet_not_implemented()
    test_ast_parse()

    print()
    if _failures == 0:
        print(f"ALL TESTS PASSED.")
        return 0
    print(f"{_failures} TEST(S) FAILED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
