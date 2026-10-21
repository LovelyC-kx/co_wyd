from dataclasses import replace
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from dataset_load import DIV2KDataset, PairedImageDataset, degrade
from thz_protocol import Protocol, fingerprint
from metrics import rgb_to_y, y_metrics


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(77)
    for split, ids in [('train', (1, 2)), ('valid', (801, 802))]:
        folder = tmp_path / f'DIV2K_{split}_HR'
        folder.mkdir()
        for image_id in ids:
            Image.fromarray(rng.integers(0, 256, (49, 57, 3), dtype=np.uint8)).save(
                folder / f'{image_id:04d}.png')
    return tmp_path


@pytest.mark.parametrize('scale', (2, 3, 4))
def test_native_lr_hr_dimensions_and_fixed_validation(images, scale):
    protocol = Protocol(scale=scale, hr_patch_size=24)
    train = DIV2KDataset(images, 'train', protocol, allow_partial=True)
    valid = DIV2KDataset(images, 'valid', protocol, allow_partial=True)
    pair = train[0]
    assert pair['hr'].shape == (3, 24, 24)
    assert pair['lr'].shape == (3, 24 // scale, 24 // scale)
    expected = (3, 49 - 49 % scale, 57 - 57 % scale)
    assert valid[0]['hr'].shape == expected
    first = valid[0]['lr'].clone()
    valid.set_epoch(100)
    torch.randn(1000)  # unrelated model RNG cannot change noise
    assert torch.equal(first, valid[0]['lr'])
    train.set_epoch(1)
    assert not torch.equal(pair['lr'], train[0]['lr'])


def numpy_collate(samples):
    # Transport plain arrays: this test host restricts tensor-IPC Unix sockets.
    # Pair generation still runs inside real spawned DataLoader workers.
    return [{'name': sample['name'], 'lr': sample['lr'].numpy(), 'hr': sample['hr'].numpy()}
            for sample in samples]


def test_worker_independent_training_pairs(images):
    dataset = DIV2KDataset(images, 'train', Protocol(hr_patch_size=24), allow_partial=True)
    dataset.set_epoch(3)
    plain = list(DataLoader(dataset, num_workers=0, batch_size=1, collate_fn=numpy_collate))
    # Windows uses spawn; exercise that start method explicitly on Linux too.
    workers = list(DataLoader(dataset, num_workers=2, batch_size=1, multiprocessing_context='spawn', timeout=30, collate_fn=numpy_collate))
    for a, b in zip(plain, workers):
        np.testing.assert_array_equal(a[0]['lr'], b[0]['lr'])
        np.testing.assert_array_equal(a[0]['hr'], b[0]['hr'])


def test_strict_split_and_no_leakage(images):
    with pytest.raises(ValueError, match='expected 800'):
        DIV2KDataset(images, 'train', Protocol())
    with pytest.raises(ValueError, match='expected 100'):
        DIV2KDataset(images, 'valid', Protocol())
    (images / 'DIV2K_train_HR' / '0801.png').write_bytes(
        (images / 'DIV2K_valid_HR' / '0801.png').read_bytes())
    with pytest.raises(ValueError, match='Wrong DIV2K IDs'):
        DIV2KDataset(images, 'train', Protocol(), allow_partial=True)


def test_full_800_100_id_validation(tmp_path):
    # Tiny generated fixtures only; these are not claimed as real DIV2K images.
    for split, ids in [('train', range(1, 801)), ('valid', range(801, 901))]:
        folder = tmp_path / f'DIV2K_{split}_HR'
        folder.mkdir()
        for image_id in ids:
            Image.new('RGB', (24, 24)).save(folder / f'{image_id:04d}.png')
    train = DIV2KDataset(tmp_path, 'train', Protocol(hr_patch_size=24))
    valid = DIV2KDataset(tmp_path, 'valid', Protocol(hr_patch_size=24))
    assert train.complete and valid.complete
    assert len(train) == 800 and len(valid) == 100
    assert set(train.files).isdisjoint(valid.files)


def test_noise_is_added_after_sampling_without_clipping():
    protocol = Protocol(hr_patch_size=24, noise_std=.1)
    hr = torch.zeros(3, 24, 24)
    lr = degrade(hr, protocol, torch.Generator().manual_seed(19))
    expected = .1 * torch.randn((3, 12, 12), generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(lr, expected, rtol=0, atol=0)
    assert lr.min() < 0
    clipped = degrade(hr, replace(protocol, clip_lr=True), torch.Generator().manual_seed(19))
    assert clipped.min() == 0
    constant = degrade(torch.full_like(hr, .4), replace(protocol, noise_std=0), torch.Generator())
    torch.testing.assert_close(constant, torch.full((3, 12, 12), .4))


def test_blur_reduces_high_frequency_and_manifest_tracks_content(images):
    hr = torch.zeros(3, 48, 48)
    hr[:, 24, 24] = 1
    sharp = degrade(hr, Protocol(noise_std=0, blur_sigma=.1), torch.Generator())
    blurred = degrade(hr, Protocol(noise_std=0, blur_sigma=2), torch.Generator())
    assert blurred.max() < sharp.max()
    dataset = DIV2KDataset(images, 'valid', Protocol(), allow_partial=True)
    before = fingerprint(dataset.manifest())
    Image.new('RGB', (49, 57)).save(dataset.files[0])
    assert before != fingerprint(dataset.manifest())


def test_y_metrics_known_colors_identity_and_no_mutation():
    colors = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]])
    np.testing.assert_allclose(rgb_to_y(colors)[0],
        np.array([81.481, 144.553, 40.966, 235.]) / 255)
    ref = np.zeros((20, 20, 3))
    pred = np.full_like(ref, .1)
    ref_before, pred_before = ref.copy(), pred.copy()
    value = y_metrics(pred, ref)
    expected = -20 * np.log10(.1 * 219 / 255)
    assert value['psnr_y'] == pytest.approx(expected)
    assert y_metrics(ref, ref) == {'psnr_y': float('inf'), 'ssim_y': 1.0}
    np.testing.assert_equal(ref, ref_before)
    np.testing.assert_equal(pred, pred_before)
    with pytest.raises(ValueError, match='at least 11'):
        y_metrics(pred, ref, border=5)


def test_real_inputs_match_names_shapes_and_no_reference(images, tmp_path):
    inputs, targets = tmp_path / 'input', tmp_path / 'target'
    inputs.mkdir()
    targets.mkdir()
    Image.new('L', (15, 17), 100).save(inputs / 'key.png')
    Image.new('RGB', (30, 34)).save(targets / 'key.png')
    no_gt = PairedImageDataset(inputs, 2)
    assert 'hr' not in no_gt[0]
    assert no_gt[0]['lr'].shape == (3, 17, 15)
    paired = PairedImageDataset(inputs, 2, targets)
    assert paired[0]['hr'].shape == (3, 34, 30)
    Image.new('RGB', (31, 34)).save(targets / 'key.png')
    with pytest.raises(ValueError, match='target must'):
        paired[0]
    (targets / 'key.png').rename(targets / 'other.png')
    with pytest.raises(ValueError, match='stems must match'):
        PairedImageDataset(inputs, 2, targets)
