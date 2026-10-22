"""
Multi-source paired-image dataset + synchronized augmentation +
Dirichlet-over-source non-IID partitioning.

Three building blocks, used together by run_smoke.py:

1. ``MultiSourceCloudDataset``
     Wraps N independent PairedCloudDatasets (one per dataset root /
     per cloud type) and exposes them as one flat index space with
     per-sample source labels.  __getitem__ returns (cloudy, clear,
     source_label) when ``with_labels=True`` so the outer partitioner
     can see which sample came from which source.

2. ``AugmentedPairedCloudDataset``
     Drop-in wrapper around any 2- or 3-tuple-returning paired dataset
     that applies identical geometric augmentations (horizontal flip,
     vertical flip, 90°/270° rotation) to (cloudy, clear).  NO colour
     jitter — remote-sensing absolute radiance has physical meaning.

3. ``dirichlet_source_partition``
     Given per-sample source labels and Dirichlet α, allocates samples
     to ``num_clients`` such that each client's per-source mixture is
     drawn from Dir(α).  Small α (<1) ⇒ heavy non-IID feature shift.

Together they produce a ``List[List[Subset]]`` shaped
(num_planes, sats_per_plane) exactly as v1's loader did, so the
existing constellation orchestrator can consume it unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

# Reuse v1's low-level paired-image loader unchanged
from cloud_removal_v1.dataset import (
    PairedCloudDataset,
    derived_train_test_split,
    seed_worker,    # re-exported for convenience
)


# ---------------------------------------------------------------------------
# Multi-source wrapper
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    root: str
    label: int
    name: str = ""


class MultiSourceCloudDataset(Dataset):
    """Concatenation of several PairedCloudDatasets, with a per-sample
    integer source label.

    Parameters
    ----------
    sources : Sequence[SourceSpec | dict]
        One entry per dataset root (e.g. CUHK-CR1 at label 0 "thin",
        CUHK-CR2 at label 1 "thick").  Dict entries must at minimum
        contain ``root`` and ``label`` keys.
    split : str | None
        Passed through to PairedCloudDataset (``'train'`` / ``'test'`` /
        ``None``).  If any source lacks the requested sub-split, the
        whole source is skipped with a warning.
    patch_size : int | None
        Passed through.  Training splits should pass ``patch_size=N``;
        test splits should pass ``None`` for full-resolution.
    with_labels : bool
        When True, __getitem__ returns (cloudy, clear, source_label);
        when False, returns the plain (cloudy, clear) pair compatible
        with v1-style DataLoaders.  `dirichlet_source_partition` needs
        labels; `CloudRemovalSNNTask` does not and should use False.
    pair_by : str
        Passed through.
    """

    def __init__(self,
                 sources: Sequence,
                 split: Optional[str] = None,
                 patch_size: Optional[int] = 64,
                 with_labels: bool = False,
                 pair_by: str = "name",
                 strict: bool = True):
        super().__init__()
        normalized: List[SourceSpec] = []
        for s in sources:
            if isinstance(s, SourceSpec):
                normalized.append(s)
            else:
                normalized.append(SourceSpec(
                    root=s["root"], label=int(s["label"]),
                    name=s.get("name", f"src{s['label']}")))
        self.sources_spec: List[SourceSpec] = normalized     # full config
        self.loaded_spec:  List[SourceSpec] = []             # only survivors
        self.split = split
        self.patch_size = patch_size
        self.with_labels = with_labels
        self.pair_by = pair_by

        self._datasets: List[PairedCloudDataset] = []
        self._labels: List[int] = []
        self._offsets: List[int] = [0]  # cumulative lengths
        for spec in normalized:
            try:
                ds = PairedCloudDataset(spec.root, split=split,
                                        patch_size=patch_size,
                                        pair_by=pair_by)
            except FileNotFoundError as e:
                msg = (f"[MultiSourceCloudDataset] source {spec.root!r} "
                       f"(split={split!r}, name={spec.name!r}) NOT LOADED: {e}")
                if strict:
                    # Loud failure so a misconfigured source isn't quietly
                    # ignored — the v2 sweep depends on BOTH CR1 + CR2 for
                    # the non-IID Dirichlet partition.
                    raise FileNotFoundError(msg) from e
                print("WARN:", msg)
                continue
            self._datasets.append(ds)
            self._labels.append(spec.label)
            self._offsets.append(self._offsets[-1] + len(ds))
            self.loaded_spec.append(spec)                    # keep aligned with _datasets

        if len(self._datasets) == 0:
            raise FileNotFoundError(
                "MultiSourceCloudDataset: no usable source datasets were "
                "discovered; check every ``root`` exists and contains "
                "input/target sub-folders.")

    # ---- Dataset protocol -----------------------------------------------

    def __len__(self) -> int:
        return self._offsets[-1]

    def _locate(self, idx: int) -> Tuple[PairedCloudDataset, int, int]:
        """Return (per-source dataset, local_idx, source_label) for a
        flat index."""
        assert 0 <= idx < len(self), f"index {idx} out of range for {len(self)}"
        # Binary-friendly linear search — O(len(sources)), sources are few.
        for i, off in enumerate(self._offsets[1:], start=1):
            if idx < off:
                local = idx - self._offsets[i - 1]
                return self._datasets[i - 1], local, self._labels[i - 1]
        raise AssertionError("unreachable")   # pragma: no cover

    def __getitem__(self, idx: int):
        ds, local, label = self._locate(idx)
        cloudy, clear = ds[local]
        if self.with_labels:
            return cloudy, clear, int(label)
        return cloudy, clear

    # ---- Diagnostics -----------------------------------------------------

    def source_labels(self) -> np.ndarray:
        """Return a [N] int array of source labels for every sample,
        suitable for passing into `dirichlet_source_partition`."""
        out = np.empty(len(self), dtype=np.int64)
        for i in range(len(self._datasets)):
            lo, hi = self._offsets[i], self._offsets[i + 1]
            out[lo:hi] = self._labels[i]
        return out

    def describe(self) -> str:
        lines = [f"MultiSourceCloudDataset(n={len(self)}, split={self.split}, "
                 f"patch_size={self.patch_size}, with_labels={self.with_labels})"]
        # Iterate `loaded_spec` (parallel to `_datasets`) — NOT `sources_spec`,
        # which may contain dropped entries and would misalign with zip().
        for spec, ds in zip(self.loaded_spec, self._datasets):
            lines.append(f"  - {spec.name:12s} (label {spec.label}) -> "
                         f"n={len(ds)} at {ds.root}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synchronized augmentation
# ---------------------------------------------------------------------------

class AugmentedPairedCloudDataset(Dataset):
    """Wrap a paired dataset, apply IDENTICAL geometric augmentations to
    the (cloudy, clear) pair.  Works with either the 2-tuple output of
    v1's PairedCloudDataset or the 3-tuple output of
    MultiSourceCloudDataset(with_labels=True) (label is passed through).

    The random draw is tied to `torch.rand` via a per-item generator,
    which means:
      * inside a worker, successive crops are uncorrelated;
      * across workers, the `seed_worker` from v1 (re-exported here)
        is enough for reproducibility;
      * `cloudy` and `clear` receive the *same* flip / rotation.

    Parameters
    ----------
    base : Dataset
        The underlying dataset (PairedCloudDataset, Subset thereof, or
        MultiSourceCloudDataset).  Must return a 2- or 3-tuple whose
        first two entries are tensors shaped [C, H, W].
    hflip_p, vflip_p : float in [0, 1]
        Horizontal / vertical flip probabilities (independent).
    rot90_p, rot270_p : float in [0, 1]
        Probability of +90° and −90° rotation respectively.
        Their sum must be ≤ 1; with probability 1 - sum, no rotation
        is applied.  180° rotation can be obtained by stacking flips,
        so omitted here to keep the draw space small and testable.
    """

    def __init__(self, base: Dataset,
                 hflip_p: float = 0.5,
                 vflip_p: float = 0.5,
                 rot90_p: float = 0.25,
                 rot270_p: float = 0.25):
        super().__init__()
        assert 0.0 <= hflip_p <= 1.0
        assert 0.0 <= vflip_p <= 1.0
        assert 0.0 <= rot90_p <= 1.0 and 0.0 <= rot270_p <= 1.0
        assert rot90_p + rot270_p <= 1.0, \
            f"rot90_p + rot270_p must ≤ 1; got {rot90_p + rot270_p}"
        self.base = base
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rot90_p = rot90_p
        self.rot270_p = rot270_p

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _apply(t: torch.Tensor, flip_h: bool, flip_v: bool,
               rot: int) -> torch.Tensor:
        """Apply fixed geometric transforms to a single [C, H, W] tensor."""
        if flip_h:
            t = torch.flip(t, dims=[-1])
        if flip_v:
            t = torch.flip(t, dims=[-2])
        if rot != 0:
            # rot ∈ {+1, -1}: +1 = 90° CCW, -1 = 90° CW
            t = torch.rot90(t, k=rot, dims=[-2, -1])
        return t

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if len(item) == 3:
            cloudy, clear, label = item
            has_label = True
        else:
            cloudy, clear = item
            has_label = False

        flip_h = torch.rand(()).item() < self.hflip_p
        flip_v = torch.rand(()).item() < self.vflip_p
        r = torch.rand(()).item()
        if r < self.rot90_p:
            rot = 1
        elif r < self.rot90_p + self.rot270_p:
            rot = -1
        else:
            rot = 0

        cloudy = self._apply(cloudy, flip_h, flip_v, rot)
        clear  = self._apply(clear,  flip_h, flip_v, rot)

        if has_label:
            return cloudy, clear, label
        return cloudy, clear


# ---------------------------------------------------------------------------
# Dirichlet-over-source partitioning
# ---------------------------------------------------------------------------

def dirichlet_source_partition(
    source_labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 0,
    min_per_client: int = 5,
) -> List[List[int]]:
    """Partition samples into `num_clients` subsets with Dirichlet(α)
    per-client source mixture.

    Algorithm:
        1. For each source `s` independently, draw a Dirichlet(α) over
           `num_clients` — this is the *fraction* of source-s samples
           that each client will receive.
        2. Shuffle source-s's sample indices, then cut them at the
           cumulative-fraction boundaries to produce per-client slices.
        3. Concatenate slices across sources to get each client's final
           index list.
        4. If any client's list is shorter than `min_per_client`, redraw
           a small balance transfer from the largest client to bring it
           up to the minimum (keeps the Dirichlet spirit without trashing
           the overall distribution).

    This is the classical "Dirichlet over LABELS" partitioner from
    McMahan-adjacent FL papers (see [Hsu et al., 2019,
    arXiv:1909.06335]), specialised here to source labels instead of
    class labels.

    Parameters
    ----------
    source_labels : np.ndarray of shape [N], dtype integer
        source_labels[i] is the source id of sample i.
    num_clients : int
    alpha : float > 0
        Smaller α ⇒ more heterogeneous (clients more skewed to one
        source).  α → ∞ approaches IID.
    seed : int
        RNG seed.
    min_per_client : int

    Returns
    -------
    List[List[int]]
        `len(partition) == num_clients`; each inner list is the flat
        sample indices assigned to that client.  The partition is a
        permutation of `range(N)` — no duplicates, no drops.
    """
    source_labels = np.asarray(source_labels, dtype=np.int64)
    N = source_labels.shape[0]
    assert N > 0
    assert num_clients > 0
    assert alpha > 0
    assert min_per_client * num_clients <= N, (
        f"can't give each of {num_clients} clients ≥{min_per_client} samples "
        f"from a pool of {N}")

    rng = np.random.RandomState(seed)
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]
    unique_labels = np.unique(source_labels)

    # B-EDGE-1: With only ONE source label, Dirichlet-over-source degenerates
    # to a (still random) per-client size split — the *feature distribution*
    # is identical across clients regardless of α, defeating the v2 non-IID
    # premise.  Surface this loudly so a misconfigured `--source_root_*` (or
    # a single-source `--data_root`) doesn't silently produce IID-equivalent
    # data and waste a 6-cell sweep that's supposed to test BN/aggregation
    # interactions under feature shift.
    if unique_labels.size <= 1:
        import warnings as _warnings
        _warnings.warn(
            f"dirichlet_source_partition: only {unique_labels.size} unique "
            f"source label(s) found — partition will be size-heterogeneous "
            f"but FEATURE-IID across clients.  v2-A's non-IID premise "
            f"requires ≥2 sources (e.g. CUHK-CR1 + CUHK-CR2).",
            RuntimeWarning, stacklevel=2)

    for label in unique_labels:
        mask = source_labels == label
        indices = np.where(mask)[0]
        rng.shuffle(indices)
        n_label = indices.size

        # Draw Dirichlet over clients for this label
        proportions = rng.dirichlet([alpha] * num_clients)
        cum = np.cumsum(proportions)
        # Convert to integer cut points; force the LAST cut to exactly
        # n_label so floating-point round-down cannot drop a sample.
        cuts = (cum * n_label).astype(np.int64)
        cuts[-1] = n_label
        prev = 0
        for c in range(num_clients):
            cut = int(cuts[c])
            if cut < prev:
                cut = prev           # monotonicity guard after the forced last cut
            client_idx[c].extend(indices[prev:cut].tolist())
            prev = cut

    # ---- Rebalance to enforce min_per_client ------------------------------
    sizes = [len(c) for c in client_idx]
    iters_used = 0
    while min(sizes) < min_per_client and iters_used < 10000:
        smallest = int(np.argmin(sizes))
        largest  = int(np.argmax(sizes))
        if sizes[largest] <= min_per_client:
            raise RuntimeError(
                "Cannot enforce min_per_client: no donor client has "
                "enough samples.  This should be unreachable thanks to "
                "the earlier assert.")
        # Transfer one random sample from largest → smallest
        donor_idx = rng.randint(0, sizes[largest])
        sample = client_idx[largest].pop(donor_idx)
        client_idx[smallest].append(sample)
        sizes = [len(c) for c in client_idx]
        iters_used += 1
    if min(sizes) < min_per_client:
        raise RuntimeError(
            f"Failed to balance after {iters_used} iterations; "
            f"smallest client has {min(sizes)} < {min_per_client}")

    # ---- Sanity: no duplicates, covers 0..N-1 -----------------------------
    flat = [i for c in client_idx for i in c]
    assert len(flat) == N, f"partition size mismatch: {len(flat)} != {N}"
    assert len(set(flat)) == N, "duplicate index detected"
    return client_idx


def build_plane_satellite_partitions_v2(
    dataset: MultiSourceCloudDataset,
    num_planes: int,
    sats_per_plane: int,
    mode: str = "dirichlet_source",
    alpha: float = 0.1,
    seed: int = 0,
    min_per_client: int = 5,
    augment: bool = True,
    augment_params: Optional[Dict[str, float]] = None,
) -> List[List[Dataset]]:
    """Two-level (plane, satellite) partition of a MultiSourceCloudDataset.

    Returned entries are already wrapped in `AugmentedPairedCloudDataset`
    when `augment=True`, so the constellation orchestrator can pass them
    straight into DataLoaders without a separate wrapper step.

    The source-label dimension of `dataset` is used under
    `mode='dirichlet_source'`; for `mode='iid'` it falls back to an IID
    random split (same semantics as v1's `build_plane_satellite_partitions`).
    """
    total_clients = num_planes * sats_per_plane

    if mode == "iid":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(dataset))
        base = len(dataset) // total_clients
        rem  = len(dataset) % total_clients
        flat_indices: List[List[int]] = []
        cursor = 0
        for c in range(total_clients):
            size = base + (1 if c < rem else 0)
            flat_indices.append(perm[cursor:cursor + size].tolist())
            cursor += size
    elif mode == "dirichlet_source":
        labels = dataset.source_labels()
        flat_indices = dirichlet_source_partition(
            source_labels=labels,
            num_clients=total_clients,
            alpha=alpha,
            seed=seed,
            min_per_client=min_per_client,
        )
    elif mode == "dirichlet_cluster":
        raise NotImplementedError(
            "dirichlet_cluster requires a feature extractor (v3 scope). "
            "Use 'dirichlet_source' with CUHK-CR1 + CUHK-CR2 for v2-A.")
    else:
        raise ValueError(f"Unknown partition_mode: {mode}")

    # Wrap each client's slice as a Subset, then (optionally) in an
    # AugmentedPairedCloudDataset.  Labels pass through via the
    # with_labels flag on the underlying MultiSourceCloudDataset.
    aug_params = augment_params or {}
    def _wrap(subset: Subset) -> Dataset:
        if not augment:
            return subset
        return AugmentedPairedCloudDataset(subset, **aug_params)

    grouped: List[List[Dataset]] = []
    k = 0
    for _ in range(num_planes):
        row = []
        for _ in range(sats_per_plane):
            sub = Subset(dataset, flat_indices[k])
            row.append(_wrap(sub))
            k += 1
        grouped.append(row)
    return grouped


# ---------------------------------------------------------------------------
# CLI probe (run directly to inspect sources + partition)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 2:
        print("usage: python -m cloud_removal_v2.dataset <path containing CUHK-CR1/ and CUHK-CR2/>")
        sys.exit(1)
    root = sys.argv[1]
    sources = []
    for i, name in enumerate(("CUHK-CR1", "CUHK-CR2")):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            sources.append({"root": p, "label": i, "name": name})
    if not sources:
        sources = [{"root": root, "label": 0, "name": "only"}]
    print(f"probing sources: {[s['name'] for s in sources]}")

    # Train split with labels
    train = MultiSourceCloudDataset(sources, split="train",
                                    patch_size=64, with_labels=True)
    print(train.describe())

    # Partition preview
    parts = build_plane_satellite_partitions_v2(
        train, num_planes=5, sats_per_plane=10,
        mode="dirichlet_source", alpha=0.1, seed=0)
    print()
    print("Per-client sizes + (thin, thick) mixture (Dirichlet α=0.1):")
    for p_idx, plane in enumerate(parts):
        line = f"  plane {p_idx}:"
        for s_idx, subset in enumerate(plane):
            # Subset's .indices attribute names differ after augment wrap
            underlying = subset
            while hasattr(underlying, "base"):
                underlying = underlying.base
            idx_list = underlying.indices if hasattr(underlying, "indices") else list(range(len(underlying)))
            labels = train.source_labels()[np.asarray(idx_list, dtype=np.int64)]
            thin = int((labels == 0).sum())
            thick = int((labels == 1).sum())
            line += f" [{len(idx_list)}|{thin}t/{thick}T]"
        print(line)

    # Sample an item to verify shape.  The probe built the train dataset
    # with with_labels=True (so Dirichlet partitioning can see labels),
    # so AugmentedPairedCloudDataset transparently forwards a 3-tuple
    # (cloudy, clear, label).  Unpack all three.
    sample = parts[0][0][0]
    cloudy, clear = sample[0], sample[1]
    label = sample[2] if len(sample) >= 3 else None
    print(f"\nfirst-sat sample shapes: {tuple(cloudy.shape)} / {tuple(clear.shape)}"
          f"  dtype={cloudy.dtype} range=[{float(cloudy.min()):.3f}, "
          f"{float(cloudy.max()):.3f}]  source_label={label}")
