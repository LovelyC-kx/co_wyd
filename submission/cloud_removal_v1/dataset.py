"""
Paired cloud-removal image dataset + federated partitioning (v1).

Supports CUHK-CR1 / CUHK-CR2 (Sui et al., TGRS 2024) and RICE1/RICE2
(Lin et al., 2019) with unified auto-discovery of on-disk layouts.

On-disk expectations (any of):

    <root>/train/input/   and  <root>/train/target/
    <root>/test/input/    and  <root>/test/target/

    <root>/input/         and  <root>/target/           (flat)

    <root>/cloudy/        and  <root>/clear/            (CUHK-CR variant)
    <root>/cloud/         and  <root>/label/ or /gt/    (seen in mirrors)

Files may be .png / .jpg / .tif.  CUHK-CR ships as 512×512 tiles; the
paper experiments actually use 256×256, but the v1 random crop to 64²
works unchanged for either resolution.

Returned tensors are [3, H, W] float32 in [0, 1], matching VLIFNet's
input contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Worker RNG seeding
# ---------------------------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn that re-seeds numpy + random.

    PyTorch re-seeds `torch.manual_seed(...)` in each worker via
    `torch.initial_seed()`, but leaves numpy and Python `random` at
    whatever state the worker was forked/spawned with.  Our random
    crop uses `np.random.randint`, so we explicitly re-seed.
    """
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base + worker_id)
    import random as _r
    _r.seed(base + worker_id)


# ---------------------------------------------------------------------------
# Folder-layout auto-discovery
# ---------------------------------------------------------------------------

_INPUT_NAMES  = ("input", "cloudy", "cloud")
_TARGET_NAMES = ("target", "clear", "label", "gt")
_IMAGE_EXTS   = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in _IMAGE_EXTS and p.is_file()])


def _pick(folder: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = folder / name
        if p.is_dir():
            return p
    return None


def _discover_folders(root: Path, split: Optional[str]
                      ) -> Tuple[Path, Path]:
    """Resolve (input_dir, target_dir) for root + optional split.

    Raises FileNotFoundError with a helpful message if no layout matches.
    """
    root = Path(root)
    bases: List[Path] = []
    if split:
        bases.append(root / split)
    bases.append(root)

    for base in bases:
        if not base.is_dir():
            continue
        in_dir  = _pick(base, _INPUT_NAMES)
        tar_dir = _pick(base, _TARGET_NAMES)
        if in_dir is not None and tar_dir is not None:
            return in_dir, tar_dir

    raise FileNotFoundError(
        f"Could not find paired input/target folders under '{root}' "
        f"(split={split!r}).  Tried input names={_INPUT_NAMES}, "
        f"target names={_TARGET_NAMES}.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _read_rgb(path: Path) -> np.ndarray:
    """Load as float32 HWC in [0, 1], forced to 3 channels."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


class PairedCloudDataset(Dataset):
    """Paired cloudy / clear image dataset.

    Parameters
    ----------
    root : str | Path
    split : {'train', 'test', None}
    patch_size : int | None
        Random crop size during training; pass None for eval / inference
        (returns full-resolution image).
    pair_by : {'name', 'order'}
        'name' matches by filename stem (recommended for CUHK-CR).  Falls
        back to sorted-order if < 90% of stems match (typical when files
        use _gt, _clean, etc. suffixes).
    """

    def __init__(self, root: str, split: Optional[str] = None,
                 patch_size: Optional[int] = 64, pair_by: str = "name"):
        super().__init__()
        assert pair_by in ("name", "order")
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.pair_by = pair_by

        in_dir, tar_dir = _discover_folders(self.root, split)
        in_files  = _list_images(in_dir)
        tar_files = _list_images(tar_dir)

        if pair_by == "name":
            tar_by_name = {p.stem: p for p in tar_files}
            pairs = [(p, tar_by_name[p.stem]) for p in in_files
                     if p.stem in tar_by_name]
            if len(pairs) < min(len(in_files), len(tar_files)) * 0.9:
                pairs = list(zip(in_files, tar_files))
        else:
            pairs = list(zip(in_files, tar_files))

        assert len(pairs) > 0, (
            f"No image pairs discovered in {in_dir} / {tar_dir}.  "
            f"|input|={len(in_files)}  |target|={len(tar_files)}")
        self.pairs = pairs
        self.in_dir = in_dir
        self.tar_dir = tar_dir

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        in_path, tar_path = self.pairs[idx]
        cloudy = _read_rgb(in_path)
        clear  = _read_rgb(tar_path)

        h, w = cloudy.shape[:2]
        if self.patch_size is not None:
            p = self.patch_size
            assert h >= p and w >= p, (
                f"Image {in_path} is smaller ({h}x{w}) than patch_size={p}")
            r = np.random.randint(0, h - p + 1)
            c = np.random.randint(0, w - p + 1)
            cloudy = cloudy[r:r + p, c:c + p]
            clear  = clear[r:r + p, c:c + p]

        cloudy_t = torch.from_numpy(np.ascontiguousarray(cloudy.transpose(2, 0, 1)))
        clear_t  = torch.from_numpy(np.ascontiguousarray(clear.transpose(2, 0, 1)))
        return cloudy_t, clear_t

    def describe(self) -> str:
        return (f"PairedCloudDataset(root={self.root}, split={self.split}, "
                f"patch_size={self.patch_size}, pair_by={self.pair_by}, "
                f"n={len(self)}, input={self.in_dir.name}, target={self.tar_dir.name})")


# ---------------------------------------------------------------------------
# Train / test split helper (when root has no explicit split subfolder)
# ---------------------------------------------------------------------------

def derived_train_test_split(root: str,
                             patch_size_train: int,
                             test_ratio: float = 0.2,
                             seed: int = 0,
                             ) -> Tuple[Subset, Subset]:
    """Build two views of the same flat-layout root: the train view uses
    random-crop patch_size_train, the test view returns full-resolution
    images.  Indices are split disjointly 8:2 with a fixed seed.

    This is the v1 fallback when the user's data layout does not have
    explicit train/ and test/ subfolders.  Both views share the same
    file list; Subset indices are reconciled against that list.
    """
    train_base = PairedCloudDataset(root, split=None, patch_size=patch_size_train)
    test_base  = PairedCloudDataset(root, split=None, patch_size=None)
    assert len(train_base) == len(test_base), \
        "Flat-layout views must share the same file list"

    n = len(train_base)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_ratio)))
    test_idx  = idx[:n_test].tolist()
    train_idx = idx[n_test:].tolist()
    return Subset(train_base, train_idx), Subset(test_base, test_idx)


# ---------------------------------------------------------------------------
# Federated partitioning
# ---------------------------------------------------------------------------

def build_client_partitions(dataset: Dataset,
                            num_clients: int,
                            mode: str = "iid",
                            seed: int = 0,
                            min_per_client: int = 1,
                            ) -> List[Subset]:
    """Split `dataset` into `num_clients` disjoint Subsets.

    v1 only implements 'iid'.  'dirichlet_cluster' is reserved for v2
    (k-means on a pretrained feature extractor, then Dirichlet-distribute
    cluster labels) and raises NotImplementedError.
    """
    n = len(dataset)
    if mode == "iid":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        base = n // num_clients
        rem  = n % num_clients
        if base < min_per_client:
            raise ValueError(
                f"dataset size {n} too small for {num_clients} clients "
                f"(needs ≥ {num_clients * min_per_client} samples)")
        subsets: List[Subset] = []
        cursor = 0
        for c in range(num_clients):
            size = base + (1 if c < rem else 0)
            subsets.append(Subset(dataset, perm[cursor:cursor + size].tolist()))
            cursor += size
        return subsets

    if mode == "dirichlet_cluster":
        raise NotImplementedError(
            "Dirichlet-cluster partitioning is scheduled for v2.  v1 must use mode='iid'.")

    raise ValueError(f"Unknown partition mode: {mode}")


def build_plane_satellite_partitions(dataset: Dataset,
                                     num_planes: int,
                                     sats_per_plane: int,
                                     mode: str = "iid",
                                     seed: int = 0,
                                     ) -> List[List[Subset]]:
    """Two-level partition shaped (num_planes, sats_per_plane)."""
    flat = build_client_partitions(dataset, num_planes * sats_per_plane,
                                   mode=mode, seed=seed)
    grouped: List[List[Subset]] = []
    k = 0
    for _ in range(num_planes):
        grouped.append(flat[k:k + sats_per_plane])
        k += sats_per_plane
    return grouped


# ---------------------------------------------------------------------------
# CLI probe
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./data/CUHK-CR1"
    print(f"Probing {root} ...")

    try:
        train = PairedCloudDataset(root, split="train", patch_size=64)
        test  = PairedCloudDataset(root, split="test",  patch_size=None)
        print("train:", train.describe())
        print("test: ", test.describe())
    except FileNotFoundError:
        ds = PairedCloudDataset(root, split=None, patch_size=None)
        print("flat:", ds.describe())
        train_sub, test_sub = derived_train_test_split(root, 64, 0.2, seed=0)
        print(f"derived 8:2 split: |train|={len(train_sub)}  |test|={len(test_sub)}")
        parts = build_plane_satellite_partitions(train_sub, 5, 10, mode="iid")
        for i, plane in enumerate(parts):
            sizes = [len(s) for s in plane]
            print(f"  plane {i}: sizes={sizes}  total={sum(sizes)}")

    x, y = ds[0] if "ds" in dir() else train[0]
    print("sample shapes:", x.shape, y.shape, "dtype:", x.dtype,
          "range:", float(x.min()), float(x.max()))
