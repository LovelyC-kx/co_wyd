import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from measurement_io import (read_numeric_matrix, load_measurement, project_preview,
                            SignalMapping, read_origin_sheet)
from infer_measurement import main as infer_main

ROOT = Path(__file__).resolve().parents[1]


def test_real_repository_key_is_101_square_and_not_axis_labeled_excel():
    raw, info = load_measurement(opju_path=ROOT / '钥匙(1).opju')
    assert raw.shape == (101, 101)
    assert raw.min() == pytest.approx(-.0233003)
    assert raw.max() == pytest.approx(.000236512)
    assert raw[0, 0] == pytest.approx(-.0210038)
    assert info['known_repository_pair_hashes_match']
    assert not info['origin_project']['numeric_project_read']
    assert not info['companion_verified_against_project_values']
    assert project_preview(ROOT / '钥匙(1).opju').startswith(b'\x89PNG')


def test_pair_mismatch_and_unknown_project_do_not_silently_use_wrong_matrix(tmp_path):
    path = tmp_path / 'other.xls'
    np.savetxt(path, np.arange(16).reshape(4, 4))
    with pytest.raises(ValueError, match='different matrix export'):
        load_measurement(path, ROOT / '钥匙(1).opju')
    project = tmp_path / 'other.opju'
    project.write_bytes(b'CPYUA test\nunknown project')
    with pytest.raises(ValueError, match='Unknown OPJU'):
        load_measurement(opju_path=project)
    matrix, info = load_measurement(path, project)
    assert matrix.shape == (4, 4) and not info['companion_verified_against_project_values']


@pytest.mark.parametrize('polarity', ['positive', 'negative'])
def test_signed_mapping_round_trip_preserves_sign_and_unclipped_predictions(polarity):
    raw = np.array([[-.02, -.01], [0., .001]])
    mapping = SignalMapping(raw.min(), raw.max(), polarity)
    norm = mapping.forward(raw)
    np.testing.assert_allclose(mapping.inverse(norm), raw, rtol=0, atol=1e-17)
    assert norm.min() == 0 and norm.max() == 1
    # Inverse is affine even outside [0,1]; numeric outputs are not silently clipped.
    assert abs(mapping.inverse(1.5) - mapping.inverse(.5)) == pytest.approx(.021)
    with pytest.raises(ValueError):
        SignalMapping(1, 1)


def test_text_and_npy_parsing_reject_missing_and_binary_excel(tmp_path):
    raw = np.array([[-1.25, 0., 2.5], [3., -4., 5.]])
    for name, delimiter in [('x.xls', '\t'), ('x.csv', ',')]:
        file = tmp_path / name
        np.savetxt(file, raw, delimiter=delimiter)
        np.testing.assert_array_equal(read_numeric_matrix(file), raw)
    path = tmp_path / 'x.npy'
    np.save(path, raw)
    np.testing.assert_array_equal(read_numeric_matrix(path), raw)
    path = tmp_path / 'missing.tsv'
    np.savetxt(path, [[1, np.nan], [2, 3]])
    with pytest.raises(ValueError, match='NaN/Inf'):
        read_numeric_matrix(path)
    path = tmp_path / 'binary.xls'
    path.write_bytes(b'\xd0\xcf\x11\xe0' + b'0' * 100)
    with pytest.raises(ValueError, match='actual Excel'):
        read_numeric_matrix(path)


def test_inspect_real_key_does_not_invent_reconstruction_or_metrics(tmp_path):
    result = tmp_path / 'inspect'
    infer_main(['--opju', str(ROOT / '钥匙(1).opju'), '--normalization', 'minmax',
                '--inspect-only', '--output', str(result)])
    report = json.loads((result / 'measurement.json').read_text())
    assert report['input_shape'] == [101, 101] and report['metrics'] is None
    assert not report['mapping']['physical_calibration']
    assert not (result / 'restored_signal.npy').exists()
    assert (result / 'origin_project_preview.png').exists()
    raw = np.load(result / 'input_signal.npy')
    np.testing.assert_array_equal(raw, read_numeric_matrix(ROOT / 'wyd01(1).xls'))


def test_origin_api_contract_and_companion_equality_without_windows(monkeypatch, tmp_path):
    # API-contract check only. A real Windows/Origin installation is not present in CI.
    import measurement_io
    raw = read_numeric_matrix(ROOT / 'wyd01(1).xls')
    calls = []
    stub = SimpleNamespace(open=lambda *a, **kw: calls.append(kw) or True,
                           find_sheet=lambda kind, ref: SimpleNamespace(to_list2=lambda: raw.tolist()),
                           oext=False)
    monkeypatch.setitem(sys.modules, 'originpro', stub)
    monkeypatch.setattr(measurement_io, 'os', SimpleNamespace(name='nt'))
    array, info = load_measurement(ROOT / 'wyd01(1).xls', ROOT / '钥匙(1).opju', '[Book1]Sheet1')
    np.testing.assert_array_equal(array, raw)
    assert calls[0] == {'readonly': True, 'asksave': True}
    assert info['origin_project']['numeric_project_read']
    assert info['companion_verified_against_project_values']
    stub.find_sheet = lambda *a: SimpleNamespace(to_list2=lambda: (raw + .01).tolist())
    with pytest.raises(ValueError, match='worksheet and companion'):
        load_measurement(ROOT / 'wyd01(1).xls', ROOT / '钥匙(1).opju', '[Book1]Sheet1')
