"""Read signed THz matrices and pair an OPJU project with its numeric export.

OPJU thumbnails are never used as measurements. Direct project numeric reading
uses Origin's official Python API on Windows; the supplied ASCII .xls export
provides a platform-independent path. No physical units/calibration are assumed.
"""
from dataclasses import asdict, dataclass
import io
import json
import os
from pathlib import Path
import struct
import zlib
import numpy as np
from thz_protocol import file_sha256

ROOT = Path(__file__).resolve().parent


def validate_matrix(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError('Measurement must be a 2-D numeric matrix, at least 2x2')
    if not np.isfinite(array).all():
        raise ValueError('Measurement contains NaN/Inf; supply an explicit missing-data treatment')
    return array


def read_numeric_matrix(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return validate_matrix(np.load(path, allow_pickle=False))
    prefix = path.read_bytes()[:16]
    if prefix.startswith((b'PK\x03\x04', b'\xd0\xcf\x11\xe0')):
        raise ValueError('This is an actual Excel binary file; export a header-free numeric TSV/CSV first')
    try:
        text = path.read_text(encoding='utf-8-sig')
        first_line = next(line for line in text.splitlines() if line.strip())
        delimiter = ',' if ',' in first_line else None
        matrix = np.loadtxt(io.StringIO(text), delimiter=delimiter, ndmin=2)
    except (UnicodeError, ValueError, StopIteration) as error:
        raise ValueError(f'{path}: expected a rectangular, header-free numeric text matrix') from error
    return validate_matrix(matrix)


def project_info(path):
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith((b'CPYUA', b'CPYA')):
        raise ValueError(f'{path}: unsupported Origin project signature')
    return {'path': str(path.resolve()), 'sha256': file_sha256(path),
            'header': data.split(b'\n', 1)[0].decode('ascii', errors='replace'),
            'numeric_project_read': False}


def project_preview(path):
    """Optional embedded PNG for provenance only, not input pixels."""
    data = Path(path).read_bytes()
    start = data.find(b'\x89PNG\r\n\x1a\n')
    if start < 0:
        return None
    end = start + 8
    while end + 12 <= len(data):
        length = struct.unpack('>I', data[end:end + 4])[0]
        tag = data[end + 4:end + 8]
        stop = end + 12 + length
        if stop > len(data):
            return None
        chunk = data[end + 4:end + 8 + length]
        expected_crc = struct.unpack('>I', data[end + 8 + length:stop])[0]
        if zlib.crc32(chunk) & 0xffffffff != expected_crc:
            return None
        end = stop
        if tag == b'IEND':
            return data[start:end]
    return None


def read_origin_sheet(path, sheet, sheet_type='w', matrix_index=0):
    """Official Origin API. Optional; requires Windows + licensed Origin.

    Worksheet data uses to_list2() so no pandas dependency or column-as-index
    conversion can drop the first data column. Matrix data uses to_np2d().
    Open read-only; never save/modify the source project.
    """
    if os.name != 'nt':
        raise RuntimeError('Direct OPJU reading requires Windows and Origin; use the companion --matrix on this machine')
    if sheet_type not in ('w', 'm') or matrix_index < 0:
        raise ValueError('Origin sheet type must be w/m and matrix index nonnegative')
    try:
        import originpro as op
    except ImportError as error:
        raise RuntimeError('Install originpro in the Windows Origin environment, or use --matrix') from error
    try:
        if not op.open(str(Path(path).resolve()), readonly=True, asksave=True):
            raise RuntimeError('Origin failed to open the project')
        source = op.find_sheet(sheet_type, sheet)
        if source is None:
            raise ValueError(f'Origin sheet not found: {sheet}')
        values = source.to_list2() if sheet_type == 'w' else source.to_np2d(matrix_index)
        return validate_matrix(values)
    finally:
        # In embedded Origin Python keep its GUI alive; external Python owns its session.
        if getattr(op, 'oext', False):
            op.exit()


def load_measurement(matrix_path=None, opju_path=None, origin_sheet=None,
                     sheet_type='w', matrix_index=0):
    if not matrix_path and not opju_path:
        raise ValueError('Supply --matrix or --opju')
    info = {'data_kind': 'signed numeric measurement', 'units': 'not supplied',
            'physical_pixel_pitch': None, 'orientation': 'rows=y, columns=x; no transpose/flip'}
    project = project_info(opju_path) if opju_path else None
    known = json.loads((ROOT / 'configs/key_wyd01.json').read_text(encoding='utf-8'))
    known_project = project is not None and project['sha256'] == known['project_sha256']
    auto_companion = False
    if matrix_path is None and opju_path and not origin_sheet:
        if not known_project:
            raise ValueError('Unknown OPJU project: provide its numeric --matrix export, or --origin-sheet on Windows')
        matrix_path = Path(opju_path).resolve().parent / known['matrix']
        auto_companion = True
        if not matrix_path.is_file():
            raise FileNotFoundError(f'Companion export not found: {matrix_path}; provide --matrix or --origin-sheet')
    exported = read_numeric_matrix(matrix_path) if matrix_path else None
    if matrix_path:
        info['matrix'] = {'path': str(Path(matrix_path).resolve()), 'sha256': file_sha256(matrix_path),
                          'format': 'npy' if Path(matrix_path).suffix.lower() == '.npy' else 'numeric text (not Excel)',
                          'automatic_companion': auto_companion}
    if origin_sheet:
        if not opju_path:
            raise ValueError('--origin-sheet requires --opju')
        array = read_origin_sheet(opju_path, origin_sheet, sheet_type, matrix_index)
        if exported is not None:
            if array.shape != exported.shape or not np.allclose(array, exported, rtol=5e-6, atol=5e-8):
                raise ValueError('Origin worksheet and companion numeric matrix differ in shape or values')
            info['companion_verified_against_project_values'] = True
        project.update(numeric_project_read=True, sheet=origin_sheet, sheet_type=sheet_type,
                       matrix_index=matrix_index)
        info['numeric_source'] = 'Origin project worksheet/matrix'
    else:
        array = exported
        info['numeric_source'] = 'numeric export'
        info['companion_verified_against_project_values'] = False
        if known_project:
            if info['matrix']['sha256'] != known['matrix_sha256']:
                raise ValueError('Known key OPJU was paired with a different matrix export; use --origin-sheet to verify new data')
            info['known_repository_pair_hashes_match'] = True
            if list(array.shape) != known['shape']:
                raise ValueError('Known key matrix shape mismatch')
    if project:
        info['origin_project'] = project
    info.update(shape=list(array.shape), minimum=float(array.min()), maximum=float(array.max()),
                mean=float(array.mean()), negative_count=int((array < 0).sum()),
                positive_count=int((array > 0).sum()))
    return array, info


@dataclass(frozen=True)
class SignalMapping:
    low: float
    high: float
    polarity: str = 'positive'
    method: str = 'minmax'

    def __post_init__(self):
        if not np.isfinite([self.low, self.high]).all() or self.high <= self.low:
            raise ValueError('Input mapping needs finite low < high; constant measurements cannot use min-max')
        if self.polarity not in ('positive', 'negative'):
            raise ValueError('Polarity must be positive or negative')

    def forward(self, array):
        values = (np.asarray(array, dtype=np.float64) - self.low) / (self.high - self.low)
        return values if self.polarity == 'positive' else 1 - values

    def inverse(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.polarity == 'negative':
            values = 1 - values
        return values * (self.high - self.low) + self.low

    def to_dict(self):
        return {**asdict(self), 'physical_calibration': False,
                'meaning': 'affine input mapping only; original signal units have not been supplied'}
