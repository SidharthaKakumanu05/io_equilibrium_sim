"""Optional, provenance-checked exact-operation native backend.

Build with ``python3 -m sim.build_native``. An absent build uses the Python
reference. An incomplete, stale or incompatible build fails explicitly; it never
silently changes the backend of a frozen campaign or resumed checkpoint.
"""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

HERE=Path(__file__).resolve().parent

def _load():
    manifest=HERE/'native_build.json'
    binaries=list(HERE.glob('_io_native*.so'))
    if not manifest.exists():
        if binaries:
            raise RuntimeError('Native binary has no build manifest; run python3 -m sim.build_native')
        return None, None, None, {'backend':'python_reference'}
    info=json.loads(manifest.read_text())
    for name,key in (('_io_native.c','source_sha256'),(info['binary'],'binary_sha256')):
        path=HERE/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=info[key]:
            raise RuntimeError('Native build missing or stale: '+str(path))
    if info['python']!=sys.version or info['numpy']!=np.__version__ or info['machine']!=platform.machine():
        raise RuntimeError('Native build environment changed; rebuild and revalidate before use')
    module=importlib.import_module('sim._io_native')
    return module.advance, module.bernoulli, module.discrete, {'backend':'native_exact','build':info}

advance_io, bernoulli, discrete_update, BACKEND_IDENTITY=_load()
