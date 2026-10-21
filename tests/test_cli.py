import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from train import main as train_main
from test import main as eval_main


def test_train_checkpoint_reloaded_evaluation_and_no_gt_inference(tmp_path):
    rng = np.random.default_rng(9)
    root = tmp_path / 'DIV2K'
    for split, ids in [('train', (1, 2)), ('valid', (801, 802))]:
        folder = root / f'DIV2K_{split}_HR'
        folder.mkdir(parents=True)
        for image_id in ids:
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                folder / f'{image_id:04d}.png')
    output = tmp_path / 'run'
    args = ['--data-root', str(root), '--output', str(output), '--allow-partial',
            '--epochs', '1', '--batch-size', '2', '--patch-size', '24',
            '--max-train-steps', '1', '--max-val-images', '1', '--workers', '0',
            '--device', 'cpu', '--tile-size', '24', '--overlap', '8']
    train_main(args)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['smoke_test'] and not manifest['author_pipeline_exact']
    assert manifest['parameter_count'] == 14023911
    assert manifest['protocol']['learning_rate'] == 5e-6
    checkpoint = output / 'model_best.pth'
    assert checkpoint.is_file() and (output / 'model_last.pth').is_file()
    with pytest.raises(FileExistsError):
        train_main(args)
    result = tmp_path / 'eval'
    eval_main(['--data-root', str(root), '--weights', str(checkpoint), '--output', str(result),
               '--allow-partial', '--max-images', '1', '--device', 'cpu'])
    train_scores = json.loads((output / 'validation_latest.json').read_text())['mean']
    test_scores = json.loads((result / 'metrics.json').read_text())
    assert test_scores['mean'] == pytest.approx(train_scores, abs=1e-8)
    assert test_scores['tiling_matches_training_validation']
    # Inference on unpaired gray inputs must never invent reference metrics.
    inputs = tmp_path / 'real'
    inputs.mkdir()
    Image.new('L', (13, 11), 100).save(inputs / 'key.png')
    real_result = tmp_path / 'real_result'
    eval_main(['--input-dir', str(inputs), '--weights', str(checkpoint),
               '--output', str(real_result), '--device', 'cpu'])
    report = json.loads((real_result / 'metrics.json').read_text())
    assert report['mean'] == {} and report['image_count'] == 1
    assert Image.open(real_result / 'images' / 'key.png').size == (26, 22)
    # Replacing a validation image must be detected, even with unchanged filenames.
    Image.new('RGB', (32, 32)).save(root / 'DIV2K_valid_HR' / '0801.png')
    with pytest.raises(ValueError, match='Validation files differ'):
        eval_main(['--data-root', str(root), '--weights', str(checkpoint),
                   '--output', str(tmp_path / 'mismatch'), '--allow-partial', '--device', 'cpu'])
