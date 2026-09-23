"""Paired synthetic resolved-event assay; not a network or shared-CF simulation.

At each synapse PF counts per second are Binomial(1000,.02), matching the
1ms Bernoulli PF marginal. IID LTD verdicts and transition uniforms are assigned
to each event. Independent synapses can be processed by within-synapse event
rank without changing any synapse's chronological order. Recorded time is s.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(os.environ.get('CASCADE_SOURCE', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT))
import numpy as np
from sim.plasticity import Plasticity

PHASES = [('background', 120), ('learn', 120), ('retain', 600),
          ('reverse', 120), ('retain_reversed', 600)]
VARIANTS = [('additive_original', 'additive', 0., 1., .9, .1),
            ('additive_matched_range', 'additive', .25, .55, .9, .1)]
for scale, pd, pu in [('native', .9, .1), ('step_matched', .009/(.55-.25), .001/(.55-.25))]:
    for mode in ('binary', 'abbott_cascade', 'mauk_cascade'):
        VARIANTS.append((mode+'_'+scale, mode, .25, .55, pd, pu))

def make(variant, seed, n):
    name, mode, lo, hi, pd, pu = variant
    obj = Plasticity(1, n, 1., 100., .001, .009, w_min=lo, w_max=hi,
                     mode=mode, p_ltd=pd, p_ltp=pu, transition_seed=[seed,702])
    weights = np.zeros((1,n), dtype=np.float64)
    obj.initialize_weights(weights, seed=[seed,701])
    return obj, weights

def verify_reordering():
    """Exact comparison against chronologically delivered identical events."""
    rng = np.random.default_rng(831)
    events = rng.random((1000,32)) < .02
    direction = np.where(rng.random(events.shape) < .1, -1, 1)
    uniform = rng.random(events.shape)
    for variant in VARIANTS:
        a, wa = make(variant, 11, 32)
        b, wb = make(variant, 11, 32)
        for t in range(1000):
            ids = np.flatnonzero(events[t])
            a.apply_resolved(wa, ids, direction[t,ids], uniform[t,ids])
        by_synapse = [np.flatnonzero(events[:,j]) for j in range(32)]
        for rank in range(max(map(len,by_synapse))):
            ids = np.array([j for j,ts in enumerate(by_synapse) if len(ts)>rank])
            ts = np.array([by_synapse[j][rank] for j in ids])
            b.apply_resolved(wb, ids, direction[ts,ids], uniform[ts,ids])
        assert np.array_equal(wa,wb), variant[0]
        assert a.n_ltd_events == b.n_ltd_events and a.n_ltp_events == b.n_ltp_events
        if variant[1] != 'additive':
            assert np.array_equal(a.states,b.states)
            assert np.array_equal(a.switch_counts,b.switch_counts)
            assert np.array_equal(a.cascade_counters(),b.cascade_counters())
    return dict(status='passed',variants=len(VARIANTS),rtol=0,atol=0)

def run_seed(seed, out, n=512):
    total = sum(duration for _,duration in PHASES)
    models = [make(v,seed,n) for v in VARIANTS]
    history = [np.empty((total+1,n)) for _ in models]
    states = [np.full((total+1,n),255,dtype=np.uint8) for _ in models]
    switches = [np.zeros((total+1,n),dtype=np.int32) for _ in models]
    counts = [np.zeros((total+1,7),dtype=np.int64) for _ in models]
    for j,(p,w) in enumerate(models):
        history[j][0] = w[0]
        if p.mode != 'additive': states[j][0] = p.states[0]
    rng = np.random.default_rng([seed,801])
    event_digest = hashlib.sha256()
    sec = 0
    started = time.monotonic()
    for phase,duration in PHASES:
        probability = np.full(n,.1)
        if phase in ('learn','reverse'):
            probability[:n//2] = .02 if phase=='learn' else .18
            probability[n//2:] = .18 if phase=='learn' else .02
        for _ in range(duration):
            number = rng.binomial(1000,.02,size=n)
            rank_count = int(number.max())
            directions = np.where(rng.random((rank_count,n)) < probability,-1,1).astype(np.int8)
            uniforms = rng.random((rank_count,n))
            for x in (number,directions,uniforms): event_digest.update(x.tobytes())
            for rank in range(rank_count):
                ids = np.flatnonzero(number>rank)
                ds,us = directions[rank,ids],uniforms[rank,ids]
                for p,w in models: p.apply_resolved(w,ids,ds,us)
            sec += 1
            for j,(p,w) in enumerate(models):
                history[j][sec] = w[0]
                counts[j][sec,:2] = p.n_ltd_events,p.n_ltp_events
                if p.mode != 'additive':
                    states[j][sec] = p.states[0]
                    switches[j][sec] = p.switch_counts[0]
                    counts[j][sec,2:] = p.cascade_counters()
        print(json.dumps(dict(seed=seed,phase=phase,second=sec,wall_s=time.monotonic()-started)),flush=True)
    artifacts=[]
    for j,(p,w) in enumerate(models):
        p.validate_state(w)
        path=out/(VARIANTS[j][0]+'_seed'+str(seed)+'.npz')
        np.savez_compressed(path,time_s=np.arange(total+1),weights=history[j],
                            states=states[j],switch_counts=switches[j],counts=counts[j])
        artifacts.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(seed=seed,event_sha256=event_digest.hexdigest(),artifacts=artifacts,wall_s=time.monotonic()-started)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    verified=verify_reordering()
    if args.verify_only:
        print(json.dumps(verified));return
    args.out.mkdir(parents=True,exist_ok=False)
    metadata=dict(schema_version=1,kind='synthetic IID resolved-event rule assay',
                  phases=PHASES,variants=VARIANTS,synapses=512,seeds=args.seeds,
                  dt_ms=1.,pf_hz=20.,background_ltd_probability=.1,
                  learning_ltd_probabilities=[.02,.18],sampling_s=1,
                  omitted='Shared CF timing correlations, neuron/network feedback, eligibility delay.',
                  state_sentinel_for_additive=255,counter_columns=['LTD','LTP','null','state_changes','weight_switches','LTD_switches','LTP_switches'],
                  validation=verified,source={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in [ROOT/'sim/plasticity.py']},
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=[])
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for seed in args.seeds:
        metadata['runs'].append(run_seed(seed,args.out))
        (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    metadata['status']='complete'
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.out/'COMPLETE.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256((args.out/'manifest.json').read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__': main()
