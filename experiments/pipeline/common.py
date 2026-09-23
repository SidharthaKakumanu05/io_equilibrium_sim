"""Filesystem, provenance and exact-state utilities for trusted local artifacts."""
import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def fsync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with open(tmp, 'x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); fsync_dir(path.parent)

def identity():
    stat = Path('/proc/self/stat').read_text()
    return dict(pid=os.getpid(), ppid=os.getppid(), sid=os.getsid(0),
                proc_start_ticks=stat.rsplit(')', 1)[1].split()[19],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                hostname=platform.node(), proc_stat=stat, utc=utc(),
                pid_namespace=os.readlink('/proc/self/ns/pid'),
                command=sys.argv, affinity=sorted(os.sched_getaffinity(0)))

def lock(path):
    f = open(path, 'a+')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close(); raise RuntimeError('Live owner holds lock: ' + str(path))
    return f

def source_hashes(root=ROOT):
    paths = [root / 'config.py'] + sorted((root / 'sim').glob('*.py'))
    paths += sorted((root / 'sim').glob('*.c'))
    paths += sorted((root / 'sim').glob('_io_native*.so'))
    if (root / 'sim' / 'native_build.json').exists():
        paths.append(root / 'sim' / 'native_build.json')
    paths += sorted((root / 'experiments' / 'pipeline').glob('*.py'))
    paths += [root / 'experiments' / 'run_background_campaign.py']
    return {str(p.relative_to(root)): digest(p) for p in paths if p.exists()}

def environment():
    from sim.native import BACKEND_IDENTITY
    return dict(native_backend=BACKEND_IDENTITY, python=sys.version, executable=sys.executable, numpy=np.__version__,
                platform=platform.platform(), thread_env={k:os.environ.get(k) for k in
                ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})

def check_space(root, policy, extra=128 << 20):
    root=Path(root)
    free=shutil.disk_usage(root).free
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if free - extra < policy['min_free_disk_bytes']:
        raise RuntimeError('Disk reserve would be violated: free=' + str(free))
    if used + extra > policy['max_storage_bytes']:
        raise RuntimeError('Campaign storage cap would be violated: bytes=' + str(used))
    return dict(free_bytes=free, campaign_bytes=used)

def assert_finite(obj, path='state', seen=None):
    if seen is None: seen=set()
    if id(obj) in seen: return
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in 'fc' and not np.isfinite(obj).all():
            raise ValueError('Nonfinite array: ' + path)
    elif isinstance(obj, (float,np.floating)):
        if not np.isfinite(obj): raise ValueError('Nonfinite scalar: ' + path)
    elif isinstance(obj, dict):
        for k,v in obj.items(): assert_finite(v,path+'.'+str(k),seen)
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj): assert_finite(v,path+f'[{i}]',seen)
    elif hasattr(obj,'__dict__'):
        assert_finite(vars(obj),path,seen)

def exact_equal(left, right):
    """Compare all mutable state, all RNG states, arrays' bits and aliases."""
    seen_l={}; seen_r={}; keep=[]; counts=dict(arrays=0,rngs=0,aliases=0,scalars=0)
    def compare(x,y,path):
        if type(x) is not type(y): raise AssertionError(path+': type')
        if isinstance(x,type):
            assert x is y,path+': immutable type identity'
            return
        if isinstance(x,(str,int,float,bool,type(None),np.generic)):
            counts['scalars']+=1
            if isinstance(x,(float,np.floating)):
                assert np.asarray(x).tobytes()==np.asarray(y).tobytes(), path+': scalar bits'
            else: assert x==y,path+': scalar'
            return
        if id(x) in seen_l or id(y) in seen_r:
            counts['aliases']+=1
            assert seen_l.get(id(x))==seen_r.get(id(y)),path+': alias'
            return
        seen_l[id(x)]=path;seen_r[id(y)]=path
        if isinstance(x,np.ndarray):
            counts['arrays']+=1
            assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),path+': array bits'
        elif isinstance(x,np.random.Generator):
            counts['rngs']+=1
            xx=x.bit_generator.state; yy=y.bit_generator.state;keep.extend([xx,yy]);compare(xx,yy,path+'.rng')
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),path+': keys'
            for k in sorted(x): compare(x[k],y[k],path+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y),path+': length'
            for i,(xx,yy) in enumerate(zip(x,y)): compare(xx,yy,path+f'[{i}]')
        else: compare(vars(x),vars(y),path)
    compare(left,right,'state')
    return counts
