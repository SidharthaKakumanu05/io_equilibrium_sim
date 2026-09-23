"""Basic deterministic analysis; writes outside immutable raw data."""
import json
from pathlib import Path
import numpy as np
from experiments.pipeline.common import atomic_json,digest,utc,source_hashes,environment
from experiments.pipeline.validate import chain,validate

def collect(out,keys):
    parts={k:[] for k in keys}
    for p,m in chain(out):
        if 'records.npz' not in m['files']:continue
        with np.load(p/'records.npz') as d:
            for k in keys:parts[k].append(d[k])
    return {k:np.concatenate(v) for k,v in parts.items()}

def analyze(out,dest):
    out=Path(out).resolve();dest=Path(dest).resolve()
    if dest==out or out in dest.parents:raise ValueError('Analysis must be outside raw directory')
    if not (out/'COMPLETE').exists():raise ValueError('No validated COMPLETE marker')
    spec=json.loads((out/'spec.json').read_text())
    if source_hashes()!=spec['source_hashes']:raise ValueError('Analysis source mismatch')
    env=environment()
    for k in ('python','numpy','executable','thread_env','native_backend'):
        if env[k]!=spec['environment'][k]:raise ValueError('Analysis environment mismatch: '+k)
    import os,resource
    os.sched_setaffinity(0,{spec['cpu']})
    resource.setrlimit(resource.RLIMIT_AS,(spec['policy']['memory_per_job_limit_bytes'],)*2)
    validate(out) # Detect post-validation corruption before any interpretation
    dest.mkdir(parents=True,exist_ok=True)
    spec=json.loads((out/'spec.json').read_text());dt=spec['config']['dt_ms']/1000
    keys=['weight_step','pkj_mean','pkj_sd','pkj_frac_lo','pkj_frac_hi','tracked_w','plasticity_counts','bin_start','bin_end',
          'io_counts','pkj_counts','dcn_counts','pf_counts','cf_step','cf_cell']
    homeostasis=spec['config'].get('homeostatic_scaling',False)
    if homeostasis:
        keys+=['homeostatic_rate_hz','homeostatic_log_factor','homeostatic_weight_change','homeostatic_clipped_updates']
    discrete=spec['config'].get('plasticity_mode','additive')!='additive'
    if discrete:
        keys+=['cascade_occupancy','tracked_state','tracked_switch_count','cascade_counts']
    d=collect(out,keys);t=d['weight_step']*dt
    duration=spec['total_steps']*dt
    rates={k:d[k+'_counts'].sum(axis=0)/duration for k in ('io','pkj','dcn','pf')}
    mean=d['pkj_mean'].mean(axis=1);within=np.sqrt((d['pkj_sd']**2).mean(axis=1))
    tail=t>=duration*.8
    slope=float(np.polyfit(t[tail]/3600,mean[tail],1)[0]) if tail.sum()>1 else None
    cf=d['cf_step']*dt; cells=d['cf_cell'];isi=[]
    for cell in range(spec['config']['n_io']):isi.extend(np.diff(cf[cells==cell]).tolist())
    summary=dict(condition=spec['condition'],seed=spec['config']['seed'],duration_s=duration,
        measured=dict(population_rate_hz={k:float(v.mean()) for k,v in rates.items()},
            per_cell_rate_hz={k:v.tolist() for k,v in rates.items()},
            final_mean_weight=float(mean[-1]),final_within_pkj_rms_sd=float(within[-1]),
            final_between_pkj_sd=float(d['pkj_mean'][-1].std()),
            final_floor_fraction=float(d['pkj_frac_lo'][-1].mean()),final_ceiling_fraction=float(d['pkj_frac_hi'][-1].mean()),
            mean_weight_slope_last_20_percent_per_hour=slope,
            tracked_rms_displacement_from_initial=float(np.sqrt(np.mean((d['tracked_w'][-1]-d['tracked_w'][0])**2))),
            ltd_events=int(d['plasticity_counts'][-1,0]),ltp_events=int(d['plasticity_counts'][-1,1]),
            cf_isi_under_100ms=sum(v<.1 for v in isi),cf_isi_count=len(isi)),
        limitations=['One matched-seed realization per condition; no between-seed uncertainty estimate.',
            'PF activity represents 40 fixed sampled synapses; 400 random unique synapses supply trajectories.',
            'Null-event counter unsupported; null_window_ms is zero in campaign.',
            'CF events are modeled Ca spikes; no burst-size model or inferred burst counts.',
            'PKJ/DCN/PF full-time rasters unavailable; per-cell one-second counts and window events retained.',
            'Stable population means do not establish stability of individual weights.'],
        raw_complete_sha256=digest(out/'COMPLETE'),utc=utc())
    if discrete:
        c=d['cascade_counts'][-1]
        span=spec['config']['cascade_high_weight']-spec['config']['cascade_low_weight']
        summary['cascade']=dict(mode=spec['config']['plasticity_mode'],
            counters_order=['null_events','state_transitions','weight_switches','ltd_switches','ltp_switches'],
            counters=c.tolist(), final_state_occupancy=d['cascade_occupancy'][-1].sum(axis=0).tolist(),
            final_per_cell_state_occupancy=d['cascade_occupancy'][-1].tolist(),
            tracked_switch_counts=d['tracked_switch_count'][-1].tolist(),
            tracked_fraction_ever_switched=float(np.mean(d['tracked_switch_count'][-1]>0)),
            weight_range_normalized_rms_displacement=summary['measured']['tracked_rms_displacement_from_initial']/span,
            measured_total_ltp_weight_change=float(c[4]*span),
            measured_total_ltd_weight_change=float(-c[3]*span))
        summary['limitations']=[v for v in summary['limitations'] if not v.startswith('Null-event counter unsupported')]
        summary['limitations'] += [
            'LTD/LTP event totals count eligibility, not successful transitions or expressed weight changes.',
            'Snapshot state trajectories omit within-bin transitions; cumulative switch counts retain all switches.',
            'This ports CbmSim transition tables into the existing PF/CF scheduling, not its complete scheduling model.',
            'Two fixed expressed weights and shallow 50/50 initialization differ from the original additive configuration.']
    if homeostasis:
        target=spec['config']['homeostatic_target_hz']
        summary['homeostatic_scaling']=dict(
            target_hz=target,rate_tau_s=spec['config']['homeostatic_rate_tau_s'],
            scaling_tau_s=spec['config']['homeostatic_tau_s'],
            final_per_cell_smoothed_rate_hz=d['homeostatic_rate_hz'][-1].tolist(),
            final_rms_rate_error_hz=float(np.sqrt(np.mean((d['homeostatic_rate_hz'][-1]-target)**2))),
            final_per_cell_cumulative_log_factor=d['homeostatic_log_factor'][-1].tolist(),
            total_homeostatic_weight_change=float(d['homeostatic_weight_change'][-1].sum()),
            clipped_synapse_updates=int(d['homeostatic_clipped_updates'][-1].sum()))
        summary['limitations'] += [
            'Phenomenological homeostatic rule; target and time constants are modeling assumptions, not Purkinje-specific fitted biology.',
            'Cumulative log factors describe attempted scaling; clipping and concurrent LTP/LTD also affect final weights.',
            'Multiplicative scaling preserves within-cell ratios only when weights are not clipped; it does not guarantee individual-synapse stability.']
    atomic_json(dest/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(4 if homeostasis else 3,1,figsize=(10,13 if homeostasis else 10),constrained_layout=True)
    ax[0].plot(t,mean,label='population mean');ax[0].plot(t,within,label='within-PKJ RMS SD');ax[0].legend();ax[0].set_ylabel('weight (dimensionless)')
    for col in range(min(20,d['tracked_w'].shape[1])):ax[1].plot(t,d['tracked_w'][:,col],lw=.5)
    ax[1].set_ylabel('20 fixed synaptic weights')
    for k in ('io','pkj','dcn'):
        ax[2].plot(d['bin_end']*dt,d[k+'_counts'].mean(axis=1)/((d['bin_end']-d['bin_start'])*dt),label=k)
    ax[2].set_ylabel('rate (Hz / cell)');ax[2].set_xlabel('biological time (s)');ax[2].legend()
    if homeostasis:
        ax[3].plot(t,d['homeostatic_rate_hz'].mean(axis=1),label='mean sensed PKJ rate')
        ax[3].fill_between(t,d['homeostatic_rate_hz'].min(axis=1),d['homeostatic_rate_hz'].max(axis=1),alpha=.2,label='cell range')
        ax[3].axhline(target,color='black',ls='--',label='target')
        ax[3].set_ylabel('homeostatic rate (Hz)');ax[3].set_xlabel('biological time (s)');ax[3].legend()
    fig.suptitle(spec['condition']);fig.savefig(dest/'overview.png',dpi=120);plt.close(fig)
    atomic_json(dest/'ANALYSIS_COMPLETE.json',dict(status='passed',utc=utc(),summary_sha256=digest(dest/'summary.json'),
        figure_sha256=digest(dest/'overview.png'),raw_complete_sha256=digest(out/'COMPLETE')))
    return summary
