import importlib
import json
import numpy as np
from PIL import Image
import pytest
import torch
from spikingjelly.activation_based import layer, functional
from submission.models_div2k import build_model
from submission.train_div2k import main as submission_train
from submission.test_div2k import main as submission_test
from train import main as original_train
from infer_measurement import main as infer_measurement


@pytest.mark.parametrize('backbone,count', [('snn', 2302901), ('ann', 2302901), ('plain', 1311603)])
def test_submission_backbone_preserved_and_backward(backbone, count):
    network = build_model(backbone)
    assert sum(p.numel() for p in network.parameters()) == count
    output = network(torch.rand(2, 3, 24, 24))
    assert output.shape == (2, 3, 24, 24)
    output.square().mean().backward()
    grads = {name: parameter.grad for name, parameter in network.named_parameters()}
    unused = [name for name, value in grads.items() if value is None]
    # Uploaded auxiliary heads are intentionally not used by the normal forward.
    assert all(name.startswith(('aux_head_level2.', 'aux_head_level3.')) for name in unused)
    assert all(torch.isfinite(value).all() for value in grads.values() if value is not None)
    functional.reset_net(network)


def test_submission_import_does_not_patch_other_model():
    assert layer.MultiDimensionalAttention.__module__.startswith('spikingjelly.')
    import model as first_model
    importlib.reload(first_model)  # simulate the opposite import order too
    assert sum(p.numel() for p in first_model.model().parameters()) == 14023911
    assert layer.ThresholdDependentBatchNorm2d._check_input_dim.__module__.startswith('torch.')


def test_identical_data_fingerprint_checkpoint_and_key_inference(tmp_path):
    rng = np.random.default_rng(44)
    data = tmp_path / 'DIV2K'
    for split, ids in [('train', (1, 2)), ('valid', (801, 802))]:
        folder = data / f'DIV2K_{split}_HR'
        folder.mkdir(parents=True)
        for image_id in ids:
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / f'{image_id:04d}.png')
    common = ['--data-root', str(data), '--allow-partial', '--epochs', '1',
              '--batch-size', '2', '--patch-size', '24', '--workers', '0',
              '--max-train-steps', '1', '--max-val-images', '1', '--device', 'cpu',
              '--tile-size', '24', '--overlap', '8']
    first, second = tmp_path / 'first', tmp_path / 'second'
    original_train(common + ['--output', str(first)])
    submission_train(common + ['--output', str(second)])
    read = lambda path: json.loads(path.read_text())
    one, two = read(first / 'manifest.json'), read(second / 'manifest.json')
    assert one['protocol'] == two['protocol']
    assert one['data_fingerprint'] == two['data_fingerprint']
    assert one['execution'] == two['execution']
    assert two['parameter_count'] == 2302901
    expected = {'submission/models_div2k.py', 'submission/cloud_removal_v1/models/vlifnet.py',
                'submission/cloud_removal_v1/models/_sj_compat.py',
                'submission/cloud_removal_v1/models/fsta_module.py'}
    assert expected <= two['model_sources'].keys()
    weights = second / 'model_best.pth'
    result = tmp_path / 'evaluation'
    submission_test(['--data-root', str(data), '--weights', str(weights), '--output', str(result),
                     '--allow-partial', '--max-images', '1', '--device', 'cpu'])
    assert read(result / 'metrics.json')['mean'] == pytest.approx(read(second / 'validation_latest.json')['mean'])
    # Run each model's checkpoint through the same real signed-matrix entry point.
    values = np.linspace(-.02, .001, 15 * 17).reshape(15, 17)
    matrix = tmp_path / 'detector.xls'
    np.savetxt(matrix, values, delimiter='\t')
    for label, checkpoint in [('first', first / 'model_best.pth'), ('second', weights)]:
        out = tmp_path / ('raw_' + label)
        infer_measurement(['--matrix', str(matrix), '--normalization', 'minmax',
                           '--weights', str(checkpoint), '--output', str(out), '--device', 'cpu'])
        report = read(out / 'measurement.json')
        assert report['output_shape'] == [30, 34]
        assert report['metrics'] is None and report['reference_available'] is False
        assert not report['synthetic_degradation_applied']
        np.testing.assert_array_equal(np.load(out / 'input_signal.npy'), values)
        restored = np.load(out / 'restored_signal.npy')
        assert restored.shape == (30, 34) and np.isfinite(restored).all()
