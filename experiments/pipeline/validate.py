"""Strict checksum, schema, timestamp, count, unit and complete-state audit."""
import dataclasses
import json
from pathlib import Path
import numpy as np
from experiments.pipeline.common import assert_finite,digest,utc
from experiments.pipeline.runner import current_state

def chain(out):
    out=Path(out); name=json.loads((out/'CURRENT.json').read_text())['generation']; result=[]; seen=set()
    while name:
        if name in seen: raise ValueError('Checkpoint cycle')
        seen.add(name);p=out/'generations'/name;m=json.loads((p/'manifest.json').read_text())
        result.append((p,m));name=m['parent']
    return list(reversed(result))

def validate(out):
    out=Path(out);spec=json.loads((out/'spec.json').read_text());state,final,name=current_state(out)
    sim=state['sim'];rec=state['rec'];assert_finite(state)
    assert state['phase']=='finished' and state['step']==spec['total_steps'], 'Incomplete horizon'
    assert state['burn_step']==spec['burn_steps'], 'Incomplete burn-in'
    assert sim.t_ms==spec['total_steps']*sim.cfg.dt_ms, 'Final simulated clock mismatch'
    assert sim._plasticity_on and dataclasses.asdict(sim.cfg)==spec['config'], 'Configuration mismatch'
    assert np.all(sim.weights>=sim.cfg.w_min) and np.all(sim.weights<=sim.cfg.w_max), 'Invalid weights'
    assert spec['activity']=='background only; no US, CS, or conditioning', 'Activity contract'
    assert spec['units']=={'time_step':'dt_ms milliseconds','voltage':'mV','calcium':'uM','io_gaba':'mS/cm^2',
                           'io_igap':'uA/cm^2','weight':'dimensionless','event_counts':'events'}, 'Units contract'
    homeostatic_keys = ('homeostatic_rate_hz', 'homeostatic_log_factor',
                       'homeostatic_weight_change', 'homeostatic_clipped_updates') if sim.homeostasis is not None else ()
    last_homeostatic = {}
    discrete = sim.plasticity.mode != 'additive'
    cascade_keys = ('cascade_occupancy','tracked_state','tracked_switch_count','cascade_counts') if discrete else ()
    last_cascade = {}
    sim.plasticity.validate_state(sim.weights)
    if homeostatic_keys:
        assert spec['homeostatic_units']==dict(zip(homeostatic_keys,
            ['Hz','dimensionless','summed dimensionless weight change per cell','synapse updates'])), 'Homeostatic units'
        h=sim.homeostasis
        assert h.n_updates == ((spec['burn_steps']+spec['total_steps'])//h.update_steps
                               - spec['burn_steps']//h.update_steps), 'Homeostatic update count'
        assert h.bin_steps == (spec['burn_steps']+spec['total_steps'])%h.update_steps, 'Homeostatic partial bin'
    assert rec.bin_start==spec['total_steps'] and all(not v.any() for v in rec.counts.values()), 'Partial bin'
    assert len(np.unique(rec.tracked,axis=0))==len(rec.tracked), 'Duplicate sampled synapses'
    expected_weights=np.r_[0,np.arange(rec.cadence,spec['total_steps']+1,rec.cadence)]
    if expected_weights[-1]!=spec['total_steps']: expected_weights=np.r_[expected_weights,spec['total_steps']]
    weight_steps=[];bin_starts=[];bin_ends=[];trace_steps=[]
    totals={k:np.zeros(n,np.uint64) for k,n in rec.sizes.items()};cf_counts=np.zeros(rec.sizes['io'],np.uint64)
    files={};prev_step=0;last_cf=(-1,-1);prev_plasticity=np.zeros(2,dtype=np.int64)
    for p,m in chain(out):
        assert m['spec_sha256']==digest(out/'spec.json'), 'Checkpoint spec mismatch'
        assert m['step']>=prev_step,'Nonmonotonic checkpoint'
        for fn,info in m['files'].items():
            f=p/fn;assert f.stat().st_size==info['bytes'] and digest(f)==info['sha256'],str(f)+' integrity'
            files[str(f.relative_to(out))]=info
        files[str((p/'manifest.json').relative_to(out))]=dict(bytes=(p/'manifest.json').stat().st_size,sha256=digest(p/'manifest.json'))
        if 'records.npz' not in m['files']:
            assert m['step']==0;continue
        with np.load(p/'records.npz',allow_pickle=False) as d:
            assert set(d.files)==set(rec.schemas), 'Recording schema keys'
            for k,(typ,shape) in rec.schemas.items():
                a=d[k]; assert a.dtype==np.dtype(typ) and a.shape[1:]==shape, k+' schema'
                assert len(a)==m['record_rows'][k],k+' row count';assert_finite(a,k)
            weight_steps.extend(d['weight_step'].tolist());bin_starts.extend(d['bin_start'].tolist());bin_ends.extend(d['bin_end'].tolist());trace_steps.extend(d['trace_step'].tolist())
            for k in ('pkj_mean','pkj_sd','pkj_frac_lo','pkj_frac_hi','tracked_w','weight_hist','plasticity_counts') + homeostatic_keys + cascade_keys:
                assert len(d[k])==len(d['weight_step']), k+' aligned rows'
            if discrete and len(d['weight_step']):
                occ=d['cascade_occupancy']; states=d['tracked_state']; switches=d['tracked_switch_count']; counts=d['cascade_counts']
                assert np.all(occ>=0) and np.all(occ.sum(axis=2)==sim.cfg.n_pf_per_pkj), 'Cascade occupancy'
                assert np.all(states<=7), 'Tracked cascade states'
                if sim.plasticity.mode=='binary':
                    assert np.all((states==3)|(states==4)) and np.all(occ[:,:,[0,1,2,5,6,7]]==0), 'Binary states'
                assert np.array_equal(d['tracked_w'],np.where(states<4,sim.plasticity.low_weight,sim.plasticity.high_weight)), 'Tracked state weight consistency'
                for key in ('tracked_switch_count','cascade_counts'):
                    previous=last_cascade.get(key,np.zeros(d[key].shape[1:],dtype=np.int64))
                    assert np.all(np.diff(np.concatenate([previous[None],d[key]]),axis=0)>=0), key+' monotonicity'
                assert np.all(counts[:,2]==counts[:,3]+counts[:,4]), 'Switch direction counts'
                assert np.all(counts[:,2]<=counts[:,1]) and np.all(counts[:,1]<=d['plasticity_counts'].sum(axis=1)), 'Transition eligibility counts'
                assert np.all(counts[:,3:5]<=d['plasticity_counts']), 'Directional switch eligibility'
                assert np.all(switches.sum(axis=1)<=counts[:,2]), 'Tracked switch subset'
                last_cascade={k:d[k][-1].copy() for k in cascade_keys}
            if homeostatic_keys and len(d['weight_step']):
                assert np.all(d['homeostatic_rate_hz']>=0), 'Negative homeostatic rate'
                clips=d['homeostatic_clipped_updates']
                previous=last_homeostatic.get('homeostatic_clipped_updates', np.zeros(sim.conn.n_pkj, dtype=np.int64))
                assert np.all(np.diff(np.vstack([previous,clips]),axis=0)>=0), 'Homeostatic clipping monotonicity'
                last_homeostatic={k:d[k][-1].copy() for k in homeostatic_keys}
            for k in ('pkj_mean','tracked_w'):
                assert np.all(d[k]>=sim.cfg.w_min) and np.all(d[k]<=sim.cfg.w_max), k+' bounds'
            assert np.all(d['pkj_sd']>=0),'Negative dispersion'
            for k in ('pkj_frac_lo','pkj_frac_hi'): assert np.all((d[k]>=0)&(d[k]<=1)),k+' fraction'
            assert np.all(d['weight_hist'].sum(axis=1)==sim.weights.size),'Histogram population'
            if len(d['plasticity_counts']):
                assert np.all(np.diff(np.vstack([prev_plasticity,d['plasticity_counts']]),axis=0)>=0),'Plasticity monotonicity'
                prev_plasticity=d['plasticity_counts'][-1]
            for k in rec.sizes:
                assert len(d[k+'_counts'])==len(d['bin_end']), 'Counts aligned'
                totals[k]+=d[k+'_counts'].sum(axis=0,dtype=np.uint64)
            for k in ('io_v','io_ca','io_gaba','io_igap','pkj_v','dcn_v'):
                assert len(d[k])==len(d['trace_step']),'Trace alignment'
            for prefix,pop in [('cf','io'),('pkj_window','pkj'),('dcn_window','dcn'),('pf_window','pf')]:
                t=d[prefix+'_step'];c=d[prefix+'_cell']
                assert len(t)==len(c),'Event alignment'
                assert np.all((t>prev_step)&(t<=m['step'])),'Event commit coverage'
                assert np.all((c>=0)&(c<rec.sizes[pop])),'Cell identity'
                if len(t)>1: assert np.all((np.diff(t)>0)|((np.diff(t)==0)&(np.diff(c)>0))),'Duplicate/unordered event'
                if prefix=='cf' and len(t):
                    assert (int(t[0]),int(c[0]))>last_cf,'Repeated CF samples'
                    last_cf=(int(t[-1]),int(c[-1]));cf_counts+=np.bincount(c,minlength=rec.sizes['io']).astype(np.uint64)
                elif len(t):
                    assert all(any(a<x<=b for a,b in rec.windows) for x in np.unique(t)), 'Event outside windows'
        prev_step=m['step']
    assert np.array_equal(weight_steps,expected_weights),'Missing/duplicated weight snapshots'
    assert bin_starts==[0]+bin_ends[:-1] and bin_ends[-1]==spec['total_steps'],'Missing/duplicate bins'
    assert np.array_equal(bin_ends,expected_weights[1:]),'Bin cadence'
    expected_trace=np.concatenate([np.arange(a+1,b+1) for a,b in rec.windows]) if rec.windows else np.array([],int)
    assert np.array_equal(trace_steps,expected_trace),'Missing/duplicate trace samples'
    for k in totals: assert np.array_equal(totals[k],rec.total_counts[k]),k+' total counts'
    assert np.array_equal(cf_counts,totals['io']),'CF events versus rate bins'
    assert np.array_equal(prev_plasticity,[sim.plasticity.n_ltd_events,sim.plasticity.n_ltp_events]),'Plasticity totals'
    if discrete:
        p=sim.plasticity
        expected=(np.stack([(p.states==i).sum(axis=1) for i in range(8)],axis=1),
                  p.states[rec.tracked[:,0],rec.tracked[:,1]],
                  p.switch_counts[rec.tracked[:,0],rec.tracked[:,1]],p.cascade_counters())
        for key,value in zip(cascade_keys,expected):
            assert np.array_equal(last_cascade[key],value), key+' final state versus records'
    if homeostatic_keys:
        for key,value in zip(homeostatic_keys,[h.rate_hz,h.cumulative_log_factor,
                            h.cumulative_weight_change,h.clipped_synapse_updates]):
            assert np.array_equal(last_homeostatic[key],value), key+' final state versus records'
    return dict(schema_version=1,status='passed',validated_at=utc(),final_step=state['step'],burn_steps=state['burn_step'],
                duration_s=sim.t_ms/1000,checkpoint=name,config_sha256=digest(out/'spec.json'),files=files,
                weight_samples=len(weight_steps),rate_bins=len(bin_ends),trace_samples=len(trace_steps),
                total_events={k:int(v.sum()) for k,v in totals.items()},
                units=spec['units'],source_hashes=spec['source_hashes'],rtol=0,atol=0)
