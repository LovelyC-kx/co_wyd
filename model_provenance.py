"""Track local model components, including nested submission attention code."""
import importlib
import inspect
from pathlib import Path
from thz_protocol import file_sha256

ROOT = Path(__file__).resolve().parent


def model_sources(network, spec):
    paths = set()
    factory_module = importlib.import_module(spec['factory'].split(':')[0])
    if getattr(factory_module, '__file__', None):
        paths.add(Path(factory_module.__file__).resolve())
    for module in network.modules():
        try:
            paths.add(Path(inspect.getfile(type(module))).resolve())
        except (TypeError, OSError):
            pass
    return {str(path.relative_to(ROOT)).replace('\\', '/'): file_sha256(path)
            for path in sorted(paths) if path.is_relative_to(ROOT) and path.suffix == '.py'}


def verify_model_sources(checkpoint):
    for relative, expected in checkpoint.get('model_sources', {}).items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f'Model component differs from checkpoint: {relative}')
