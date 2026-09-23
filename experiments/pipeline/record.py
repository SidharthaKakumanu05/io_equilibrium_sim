"""Bounded chunk recording; original simulation state is only read here."""
import numpy as np

class Recorder:
    def __init__(self, sim, total_steps, trace_seconds=5.0):
        self.dt=sim.cfg.dt_ms
        self.cadence=round(1000/self.dt)
        self.total_steps=total_steps
        self.selection_rng=np.random.default_rng([sim.cfg.seed,419])
        flat=self.selection_rng.choice(sim.weights.size,400,replace=False)
        self.tracked=np.column_stack(np.unravel_index(flat,sim.weights.shape))
        self.pf=sim.pf_recorded.copy()
        self.sizes={'io':sim.cfg.n_io,'pkj':sim.conn.n_pkj,'dcn':sim.conn.n_dcn,'pf':len(self.pf)}
        self.counts={k:np.zeros(n,dtype=np.uint32) for k,n in self.sizes.items()}
        self.total_counts={k:np.zeros(n,dtype=np.uint64) for k,n in self.sizes.items()}
        self.bin_start=0
        n=min(round(trace_seconds*1000/self.dt),total_steps//4)
        self.windows=[(0,n),(total_steps//2,total_steps//2+n),(total_steps-n,total_steps)] if n else []
        self.hist_edges=np.linspace(sim.cfg.w_min,sim.cfg.w_max,51)
        self.schemas={
            'weight_step':(np.int32,()), 'pkj_mean':(np.float64,(sim.conn.n_pkj,)),
            'pkj_sd':(np.float64,(sim.conn.n_pkj,)), 'pkj_frac_lo':(np.float64,(sim.conn.n_pkj,)),
            'pkj_frac_hi':(np.float64,(sim.conn.n_pkj,)), 'tracked_w':(np.float64,(400,)),
            'weight_hist':(np.uint32,(50,)), 'plasticity_counts':(np.int64,(2,)),
            'bin_start':(np.int32,()),'bin_end':(np.int32,()),
            'cf_step':(np.int32,()),'cf_cell':(np.int16,()),
            'trace_step':(np.int32,()),
            'io_v':(np.float64,(sim.cfg.n_io,)), 'io_ca':(np.float64,(sim.cfg.n_io,)),
            'io_gaba':(np.float64,(sim.cfg.n_io,)), 'io_igap':(np.float64,(sim.cfg.n_io,)),
            'pkj_v':(np.float64,(sim.conn.n_pkj,)), 'dcn_v':(np.float64,(sim.conn.n_dcn,)),
        }
        for k,nc in self.sizes.items():
            self.schemas[k+'_counts']=(np.uint32,(nc,))
            if k!='io':
                self.schemas[k+'_window_step']=(np.int32,()); self.schemas[k+'_window_cell']=(np.int16,())
        if sim.homeostasis is not None:
            for k in ('homeostatic_rate_hz', 'homeostatic_log_factor', 'homeostatic_weight_change'):
                self.schemas[k]=(np.float64,(sim.conn.n_pkj,))
            self.schemas['homeostatic_clipped_updates']=(np.int64,(sim.conn.n_pkj,))
        if sim.plasticity.mode != 'additive':
            self.schemas.update(cascade_occupancy=(np.int64,(sim.conn.n_pkj,8)),
                tracked_state=(np.uint8,(400,)), tracked_switch_count=(np.int64,(400,)),
                cascade_counts=(np.int64,(5,)))
        self.data={k:[] for k in self.schemas}
        self.snapshot(sim,0)

    def snapshot(self,sim,step):
        w=sim.weights
        self.data['weight_step'].append(step)
        self.data['pkj_mean'].append(w.mean(axis=1))
        self.data['pkj_sd'].append(w.std(axis=1))
        self.data['pkj_frac_lo'].append((w<=sim.cfg.w_min).mean(axis=1))
        self.data['pkj_frac_hi'].append((w>=sim.cfg.w_max).mean(axis=1))
        self.data['tracked_w'].append(w[self.tracked[:,0],self.tracked[:,1]].copy())
        self.data['weight_hist'].append(np.histogram(w,bins=self.hist_edges)[0])
        self.data['plasticity_counts'].append([sim.plasticity.n_ltd_events,sim.plasticity.n_ltp_events])
        if sim.plasticity.mode != 'additive':
            p=sim.plasticity
            self.data['cascade_occupancy'].append(np.stack([(p.states==i).sum(axis=1) for i in range(8)],axis=1))
            self.data['tracked_state'].append(p.states[self.tracked[:,0],self.tracked[:,1]].copy())
            self.data['tracked_switch_count'].append(p.switch_counts[self.tracked[:,0],self.tracked[:,1]].copy())
            self.data['cascade_counts'].append(p.cascade_counters())
        if sim.homeostasis is not None:
            h=sim.homeostasis
            for k,v in [('homeostatic_rate_hz',h.rate_hz),
                        ('homeostatic_log_factor',h.cumulative_log_factor),
                        ('homeostatic_weight_change',h.cumulative_weight_change),
                        ('homeostatic_clipped_updates',h.clipped_synapse_updates)]:
                self.data[k].append(v.copy())

    def record(self,sim,step,events):
        pf,pkj,dcn,cf=events
        activity={'pf':pf[self.pf[:,0],self.pf[:,1]],'pkj':pkj,'dcn':dcn,'io':cf}
        for k,sp in activity.items():
            self.counts[k]+=sp
            self.total_counts[k]+=sp
        idx=np.flatnonzero(cf)
        self.data['cf_cell'].extend(idx);self.data['cf_step'].extend([step]*len(idx))
        if any(a<step<=b for a,b in self.windows):
            self.data['trace_step'].append(step)
            for k,v in [('io_v',sim.io.V),('io_ca',sim.io.Ca),('io_gaba',sim.io_gaba.g),
                        ('pkj_v',sim.pkj.V),('dcn_v',sim.dcn.V)]:
                self.data[k].append(v.copy())
            v=sim.io.V
            gap=np.zeros(sim.cfg.n_io) if sim.io.g_gap is None else sim.io.g_gap@v-sim.io._gap_row_sum*v
            self.data['io_igap'].append(gap)
            for k in ('pf','pkj','dcn'):
                ids=np.flatnonzero(activity[k])
                self.data[k+'_window_cell'].extend(ids)
                self.data[k+'_window_step'].extend([step]*len(ids))
        if step%self.cadence==0 or step==self.total_steps:
            self.data['bin_start'].append(self.bin_start);self.data['bin_end'].append(step)
            for k in self.sizes:
                self.data[k+'_counts'].append(self.counts[k].copy());self.counts[k].fill(0)
            self.bin_start=step
            self.snapshot(sim,step)

    def arrays(self):
        return {k:np.asarray(v,dtype=self.schemas[k][0]).reshape((-1,)+self.schemas[k][1]) for k,v in self.data.items()}

    def clear(self):
        self.data={k:[] for k in self.schemas}
