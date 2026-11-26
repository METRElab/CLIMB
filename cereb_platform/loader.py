"""Loading granule-layer models from .py files."""

import importlib.util
import inspect
import sys
from pathlib import Path

from .granulelayer import GranuleLayer


def load_model(spec):
    """Load the GranuleLayer subclass from 'path.py' or 'path.py:ClassName'."""
    spec = str(spec)
    path, cls_name = spec, None
    if not spec.endswith('.py') and ':' in spec:
        path, cls_name = spec.rsplit(':', 1)
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'model file not found: {path}')
    mod_name = f'_cereb_model_{path.stem}_{abs(hash(str(path))) % 10**8}'
    mod_spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_name] = module
    mod_spec.loader.exec_module(module)
    found = [c for _, c in inspect.getmembers(module, inspect.isclass)
             if issubclass(c, GranuleLayer) and c is not GranuleLayer
             and c.__module__ == module.__name__ and not inspect.isabstract(c)]
    if cls_name:
        found = [c for c in found if c.__name__ == cls_name]
    if len(found) != 1:
        raise ValueError(f'{spec}: expected one GranuleLayer subclass, '
                         f'found {[c.__name__ for c in found]} (use path.py:ClassName)')
    cls = found[0]
    cls.check_defaults()
    return cls


def discover_models(folder):
    """{display name: spec} for every loadable model file in ``folder``, plus
    {file: error} for files that failed to load."""
    models, errors = {}, {}
    folder = Path(folder)
    for f in sorted(folder.glob('*.py')):
        if f.name.startswith('_'):
            continue
        try:
            cls = load_model(f)
        except Exception as e:     # a broken model file shouldn't stop the server
            errors[f.name] = f'{type(e).__name__}: {e}'
            continue
        name = cls.name if cls.name not in models else f'{cls.name} ({f.stem})'
        models[name] = str(f.resolve())
    return models, errors
