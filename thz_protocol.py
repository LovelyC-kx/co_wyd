"""THz-Net paper settings and explicitly chosen, non-author degradation settings.

Source: OE_THz_Net_Submission.pdf, Sec. 2.1, 3.2 and Table 4.
The paper does NOT specify rho, beta, kernel size, sampler, or metric borders.
The bundled values for these are reproducible experimental choices, NOT an
exact reproduction of the authors' unpublished pipeline. See docs/THZ_PROTOCOL.md.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import math


@dataclass(frozen=True)
class Protocol:
    protocol_version: int = 1
    scale: int = 2
    hr_patch_size: int = 192
    blur_sigma: float = 1.6
    kernel_size: int = 21
    noise_std: float = 0.01  # Normalized [0, 1] RGB units, not 0..255.
    seed: int = 1234
    augment: bool = False
    clip_lr: bool = False
    metric_border: int = 0
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 5e-6

    def __post_init__(self):
        if self.protocol_version != 1:
            raise ValueError('Unsupported protocol version')
        if self.scale not in (2, 3, 4):
            raise ValueError('Use a separate experiment for scale 2, 3, or 4')
        if self.hr_patch_size < 16 or self.hr_patch_size % self.scale:
            raise ValueError('HR patch size must be >=16 and divisible by scale')
        if self.kernel_size < 1 or self.kernel_size % 2 != 1:
            raise ValueError('kernel_size must be a positive odd integer')
        for name in ('blur_sigma', 'noise_std', 'learning_rate'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.blur_sigma <= 0 or self.noise_std < 0 or self.learning_rate <= 0:
            raise ValueError('Invalid blur/noise/learning rate')
        if self.metric_border < 0 or self.seed < 0:
            raise ValueError('Border and seed must be nonnegative')
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError('epochs and batch_size must be positive')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return asdict(self)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
