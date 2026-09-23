"""Full-object/RNG checkpoints with immutable record generations and atomic cursors.

Pickles are trusted local artifacts only. Checksum/config/source/environment checks
precede unpickling. No simulation defaults or methods are replaced or modified.
"""
import dataclasses
import gzip
import json
import os
from pathlib import Path
import pickle
import resource
import signal
import time
import uuid
import numpy as np
from config import SimConfig, LIFParams, IOChannelParams
from sim.simulate import Simulation
from experiments.pipeline.common import (ROOT,atomic_json,assert_finite,check_space,digest,
    environment,fsync_dir,identity,lock,source_hashes,utc)
from experiments.pipeline.record import Recorder

def config_from_dict(d):
    d=dict(d)
    for k in ('pkj','dcn'): d[k]=LIFParams(**d[k])
    d['io_channels']=IOChannelParams(**d['io_channels'])
    return SimConfig(**d)

def load_generation(out, name):
    p=Path(out)/'generations'/name
    m=json.loads((p/'manifest.json').read_text())
    for k,v in m['files'].items():
        if digest(p/k)!=v['sha256'] or (p/k).stat().st_size!=v['bytes']:
            raise ValueError('Corrupt checkpoint/records: '+str(p/k))
    with gzip.open(p/'state.pkl.gz','rb') as f: state=pickle.load(f)
    return state,m

def current_state(out):
    ptr=json.loads((Path(out)/'CURRENT.json').read_text())
    if digest(Path(out)/'generations'/ptr['generation']/'manifest.json')!=ptr['manifest_sha256']:
        raise ValueError('Current manifest checksum mismatch')
    state,m=load_generation(out,ptr['generation'])
    return state,m,ptr['generation']

def checkpoint(out,state,parent,spec,fault=None):
    # Serialize reserve checking and publication across every campaign worker.
    import fcntl
    with open(Path(spec['storage_root'])/'STORAGE.lock','a+') as budget_guard:
        fcntl.flock(budget_guard,fcntl.LOCK_EX)
        return _checkpoint(out,state,parent,spec,fault)

def _checkpoint(out,state,parent,spec,fault=None):
    """Commit record bytes, full precision state and their manifest in one generation."""
    check_space(spec['storage_root'],spec['policy'])
    assert_finite(state['sim'])
    state['sim'].plasticity.validate_state(state['sim'].weights)
    arrays=state['rec'].arrays() if state['rec'] else {}
    for k,v in arrays.items(): assert_finite(v,k)
    name=f"g-{state['phase']}-{state['burn_step']:08d}-{state['step']:09d}-{uuid.uuid4().hex[:12]}"
    pending=Path(out)/'generations'/('.pending-'+name)
    pending.mkdir()
    if arrays:
        with open(pending/'records.npz','xb') as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    if fault=='after_record_write': os.kill(os.getpid(),signal.SIGKILL)
    if state['rec']: state['rec'].clear()
    with open(pending/'state.pkl.gz','xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',compresslevel=1,mtime=0) as z:
            pickle.dump(state,z,protocol=5)
        f.flush();os.fsync(f.fileno())
    files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in pending.iterdir()}
    m=dict(schema_version=1,parent=parent,phase=state['phase'],step=state['step'],burn_step=state['burn_step'],
           files=files,spec_sha256=digest(Path(out)/'spec.json'),utc=utc(),
           pending_pf=int(np.count_nonzero(state['sim'].plasticity._pf_ring)),
           plasticity_write_ptr=int(state['sim'].plasticity._write_ptr),
           record_rows={k:len(v) for k,v in arrays.items()})
    atomic_json(pending/'manifest.json',m);fsync_dir(pending)
    os.rename(pending,pending.with_name(name));fsync_dir(pending.parent)
    if fault=='before_cursor': os.kill(os.getpid(),signal.SIGKILL)
    atomic_json(Path(out)/'CURRENT.json',dict(generation=name,manifest_sha256=digest(pending.with_name(name)/'manifest.json')))
    if fault=='after_cursor': os.kill(os.getpid(),signal.SIGKILL)
    return name

def run(spec_path,out,resume=False,fault=None,fault_step=None,stop_step=None):
    spec_path=Path(spec_path); out=Path(out)
    spec=json.loads(spec_path.read_text())
    if source_hashes()!=spec['source_hashes']: raise ValueError('Source provenance mismatch')
    env=environment()
    for k in ('python','numpy','executable','thread_env','native_backend'):
        if env[k]!=spec['environment'][k]: raise ValueError('Environment mismatch: '+k)
    if not resume:
        out.mkdir() # existing output is an error, never overwrite a dataset
        (out/'generations').mkdir();atomic_json(out/'spec.json',spec)
    guard=lock(out/'RUN.lock')
    if json.loads((out/'spec.json').read_text())!=spec: raise ValueError('Configuration/spec mismatch')
    resource.setrlimit(resource.RLIMIT_AS,(spec['policy']['memory_per_job_limit_bytes'],)*2)
    os.sched_setaffinity(0,{spec['cpu']})
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    stop_requested=[]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop_requested.append(True))
    started=time.monotonic(); prior_wall=0.; parent=None
    ident=identity();attempt=uuid.uuid4().hex
    atomic_json(out/f'attempt-{attempt}.json',dict(identity=ident,started=utc(),resume=resume,spec_path=str(spec_path)))
    try:
        check_space(spec['storage_root'],spec['policy'])
        if resume:
            if (out/'COMPLETE').exists(): raise ValueError('Run already complete; use analysis command only')
            state,_,parent=current_state(out); prior_wall=state['wall_s']
        else:
            sim=Simulation(config_from_dict(spec['config']));sim._choose_recorded();sim._plasticity_on=False
            state=dict(sim=sim,rec=None,phase='burn',burn_step=0,step=0,wall_s=0.)
            parent=checkpoint(out,state,None,spec)
        sim=state['sim']; total=spec['total_steps'];burn=spec['burn_steps'];interval=spec['checkpoint_steps']
        def save(fault_now=None):
            nonlocal parent
            state['wall_s']=prior_wall+time.monotonic()-started
            parent=checkpoint(out,state,parent,spec,fault_now)
            atomic_json(out/'status.json',dict(status='running',identity=ident,phase=state['phase'],step=state['step'],
                burn_step=state['burn_step'],checkpoint=parent,wall_s=state['wall_s'],updated=utc(),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        while state['burn_step']<burn:
            sim._step();state['burn_step']+=1
            if fault=='burn_buffer' and state['burn_step']==fault_step: os.kill(os.getpid(),signal.SIGKILL)
            if state['burn_step']%1000==0:
                save()
                if stop_requested: raise InterruptedError('Signal during burn-in; checkpoint saved')
        if state['phase']=='burn':
            sim._plasticity_on=True;sim.t_ms=0.;state['phase']='active'
            state['rec']=Recorder(sim,total,spec['trace_seconds']);save()
        while state['step']<total:
            events=sim._step();sim.t_ms+=sim.cfg.dt_ms;state['step']+=1
            state['rec'].record(sim,state['step'],events)
            if fault=='buffer' and state['step']==fault_step: os.kill(os.getpid(),signal.SIGKILL)
            if state['step']%1000==0:
                if prior_wall+time.monotonic()-started>spec['policy']['wall_clock_budget_per_trial_s']:
                    save();raise RuntimeError('Wall-time budget reached')
            if state['step']%interval==0 or state['step']==total or stop_requested or state['step']==stop_step:
                save(fault if state['step']==fault_step else None)
                if stop_requested or state['step']==stop_step: raise InterruptedError('Stopped at durable checkpoint')
        state['phase']='finished';save()
        atomic_json(out/'SIMULATION_DONE.json',dict(step=state['step'],wall_s=state['wall_s'],identity=ident,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,utc=utc(),checkpoint=parent))
        from experiments.pipeline.validate import validate
        result=validate(out)
        atomic_json(out/'COMPLETE',result)
        atomic_json(out/'status.json',dict(status='validated',step=total,identity=ident,checkpoint=parent,updated=utc()))
        return result
    except BaseException as exc:
        atomic_json(out/'status.json',dict(status='stopped' if isinstance(exc,InterruptedError) else 'failed',
            error=repr(exc),identity=ident,checkpoint=parent,updated=utc()))
        raise
    finally:
        guard.close()
