"""The main simulation loop.

This file has three parts:

  SpikeRecorder  collects spike times while the simulation runs
  SimLog         the container for everything a run records (returned by run())
  Simulation     builds the network, then steps it forward 1 ms at a time

How the network is stored: each cell type is ONE population object holding all
of its cells as arrays (all 320 Purkinje cells in one PKJPopulation, and so on).
Connections between populations are 0/1 matrices, so "deliver every PKJ spike
to the DCN cells it connects to" is a single matrix-vector product.

One time step (Simulation._step) goes around the loop in order:

    1. draw this step's PF spikes, excite the Purkinje cells, step them
    2. Purkinje spikes inhibit the DCN; step the DCN
    3. DCN spikes inhibit the olive; step the olive -> CF events
    4. CF events give their Purkinje cells a complex spike and a pause,
       and drive the learning rule

Each stage uses the spikes the previous stage produced in the SAME step, so a
change travels all the way around the loop within one step.

A run (Simulation.run) has two phases:

    burn-in   BURN_IN_S seconds with learning off and nothing recorded. Every
              cell starts at rest, and for the first moments the olive has no
              inhibition yet and fires too fast. If learning were on, that
              start-up burst would wrongly depress the weights.
    trials    NUM_TRIALS trials of TRIAL_MS each, with learning on, recording
              everything. As in CbmSim, a trial is a block of time steps: the
              network carries on from one trial to the next without being
              reset, and rasters are written to disk at the end of each trial
              (sim/rasters.py).

Times in the SimLog run continuously from t = 0 (the end of burn-in), so trial
k covers k*TRIAL_MS to (k+1)*TRIAL_MS.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from config import SimConfig
from sim.connectivity import build_connectivity
from sim.io_channels import IOPopulation
from sim.io_coupling import build_gap_junction_matrix
from sim.neurons import DCNPopulation, ExpSynapse, PKJPopulation
from sim.plasticity import Plasticity
from sim.poisson_input import generate_pf_spikes
from sim.rasters import FORMATS, CbmRasterWriter


class SpikeRecorder:
    """Records spikes for one population during a run.

    Each step, record() appends a (time, cell index) pair for every cell that
    spiked. At the end, as_trains() turns that into one array of spike times
    per cell."""

    def __init__(self, n_cells):
        self.n_cells = n_cells
        self._t = []
        self._cell = []

    def record(self, t_ms, spiked):
        idx = np.flatnonzero(spiked)
        if len(idx):
            self._t.extend([t_ms] * len(idx))
            self._cell.extend(idx.tolist())

    def as_trains(self):
        t = np.asarray(self._t, dtype=float)
        cell = np.asarray(self._cell, dtype=int)
        return [t[cell == i] for i in range(self.n_cells)]

    @property
    def n_spikes(self):
        return len(self._t)


@dataclass
class SimLog:
    """Everything recorded from one run. Times are in ms, with t = 0 at the end
    of burn-in.

    What is kept, and how much:
      * spikes: every spike of every IO, PKJ and DCN cell. For PFs, only a
        sample of cfg.n_pf_recorded fibres (all 160,000 would be far too many).
      * weights: sampled every cfg.record_every_ms. The mean per climbing-fibre
        territory, plus a few individually tracked synapses.
      * membrane voltages: every step, but only for the last cfg.trace_window_s
        seconds of the run.
    Firing rates are not stored; sim/analysis.py counts them from the spikes.
    """
    # --- spikes: a list with one array of spike times (ms) per cell ---
    io_spikes: list = field(default_factory=list)      # one per olive cell: these ARE the CF events
    pkj_spikes: list = field(default_factory=list)     # one per Purkinje cell
    dcn_spikes: list = field(default_factory=list)     # one per DCN cell
    pf_spikes: list = field(default_factory=list)      # one per RECORDED parallel fibre
    pf_recorded: np.ndarray = None                     # (n_pf_recorded, 2): which (PKJ, fibre) each one is

    # --- weights, sampled every cfg.record_every_ms ---
    t_ms: np.ndarray = None                            # (T,) sample times
    mean_weight: np.ndarray = None                     # (T, n_io) mean weight over each climbing fibre's 8 PKJ
    sample_weights: np.ndarray = None                  # (T, n_tracked) individual synapses over time
    tracked_synapses: np.ndarray = None                # (n_tracked, 2): which (PKJ, fibre) each column is

    # --- membrane state, every step over the last cfg.trace_window_s seconds ---
    trace_t_ms: np.ndarray = None                      # (S,) sample times
    trace_io_v: np.ndarray = None                      # (S, n_io) olive voltage, mV
    trace_io_ca: np.ndarray = None                     # (S, n_io) olive calcium, uM
    trace_pkj_v: np.ndarray = None                     # (S, n_pkj) PKJ voltage, mV (spikes drawn as a peak)
    trace_dcn_v: np.ndarray = None                     # (S, n_dcn) DCN voltage, mV (spikes drawn as a peak)
    trace_io_gaba: np.ndarray = None                   # (S, n_io) DCN inhibition reaching each olive cell, mS/cm^2

    # --- end state and labels ---
    final_weights: np.ndarray = None                   # (n_pkj, n_pf) every weight at the end of the run
    group_of_pkj: np.ndarray = None                    # colour label for each PKJ in plots (= its climbing fibre)
    group_of_dcn: np.ndarray = None                    # colour label for each DCN in plots (= its own index)
    io_of_pkj: np.ndarray = None                       # which olive cell's climbing fibre contacts each PKJ
    gap_matrix: np.ndarray = None                      # (n_io, n_io) gap-junction conductances used
    duration_ms: float = 0.0                           # length of the recorded part of the run (all trials)
    n_trials: int = 0
    trial_ms: float = 0.0
    trial_cf_counts: np.ndarray = None                 # (n_trials, n_io) CF events per olive cell per trial
    trial_mean_weight: np.ndarray = None               # (n_trials,) mean PF->PKJ weight at the end of each trial
    meta: dict = field(default_factory=dict)           # cell counts and key settings, for labelling


class Simulation:
    """Build the network from a SimConfig, then call run() to simulate it."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

        # --- wiring: who connects to whom (sim/connectivity.py) ---
        self.conn = build_connectivity(
            n_io=cfg.n_io, n_dcn=cfg.n_dcn, n_pkj_per_io=cfg.n_pkj_per_io,
            n_pkj_per_dcn=cfg.n_pkj_per_dcn, n_dcn_per_pkj=cfg.n_dcn_per_pkj,
            n_dcn_per_io=cfg.n_dcn_per_io, enforce_closed_loop=cfg.enforce_closed_loop,
            seed=cfg.connectivity_seed,
        )
        n_io, n_pkj, n_dcn = cfg.n_io, self.conn.n_pkj, self.conn.n_dcn

        # --- random number generators ---
        # One separate generator per use, all derived from cfg.seed. Then changing one of them
        # (say, recording different synapses) does not change the PF input the run receives.
        seed = cfg.seed
        self.rng = np.random.default_rng(seed)                                          # PF spikes
        io_rng = np.random.default_rng(None if seed is None else [seed, 1])             # olive noise (off by default)
        lif_rng = np.random.default_rng(None if seed is None else [seed, 2])            # PKJ/DCN noise and starting voltages
        self.record_rng = np.random.default_rng(None if seed is None else [seed, 3])    # which synapses/PFs get recorded

        # Each Purkinje cell has exactly one climbing fibre. io_of_pkj[p] says which olive cell it is.
        # Plots colour PKJ (and their PFs) by this. DCN cells are not grouped this way because
        # each one receives Purkinje cells from several climbing fibres.
        self.io_of_pkj = self.conn.cf_of_pkj
        self.group_of_pkj = self.io_of_pkj
        self.group_of_dcn = np.arange(n_dcn)

        # --- the olive: gap junctions, then the cells themselves (sim/io_coupling.py, sim/io_channels.py) ---
        self.gap_matrix = build_gap_junction_matrix(n_io, cfg.gap_g, cfg.gap_topology, cfg.gap_n_neighbors)
        self.io = IOPopulation(cfg.io_channels, n_io, cfg.dt_ms, rng=io_rng, g_gap=self.gap_matrix,
                               heterogeneity_cv=cfg.io_heterogeneity_cv,
                               heterogeneity_seed=cfg.io_heterogeneity_seed)

        # --- Purkinje and DCN cells (sim/neurons.py) ---
        self.pkj = PKJPopulation(
            n_pkj, cfg.dt_ms, cfg.pkj.tau_m_ms, cfg.pkj.v_th_mv, cfg.pkj.v_reset_mv, cfg.pkj.e_leak_mv,
            cfg.pkj.t_ref_ms, cfg.pkj_baseline_hz, cfg.tau_pf_pkj_ms, cfg.tau_pkj_dcn_ms,
            cfg.e_exc_mv, cfg.e_inh_mv, noise_sigma_mv=cfg.pkj.noise_sigma_mv, rng=lif_rng,
            pause_g=cfg.cf_pause_g, pause_ms=cfg.cf_pause_ms, e_pause_mv=cfg.e_inh_mv,
            burst_spikes=cfg.cf_burst_spikes, burst_isi_ms=cfg.cf_burst_isi_ms,
        )
        self.dcn = DCNPopulation(
            n_dcn, cfg.dt_ms, cfg.dcn.tau_m_ms, cfg.dcn.v_th_mv, cfg.dcn.v_reset_mv, cfg.dcn.e_leak_mv,
            cfg.dcn.t_ref_ms, cfg.dcn_baseline_hz, cfg.tau_pf_pkj_ms, cfg.tau_pkj_dcn_ms,
            cfg.e_exc_mv, cfg.e_inh_mv, noise_sigma_mv=cfg.dcn.noise_sigma_mv, rng=lif_rng,
        )
        # The DCN -> IO inhibitory synapse. Its conductance is handed to the olive on every step.
        self.io_gaba = ExpSynapse(n_io, cfg.tau_dcn_io_ms, cfg.dt_ms, cfg.io_channels.e_gaba)

        # --- connection matrices: entry [post, pre] is 1 if pre connects to post ---
        self.m_pkj_to_dcn = self.conn.pkj_to_dcn_matrix()   # (n_dcn, n_pkj)
        self.m_dcn_to_io = self.conn.dcn_to_io_matrix()     # (n_io, n_dcn)

        # Scale the two inhibitory gains by (fitted count / actual count of converging cells), so a
        # DCN or olive cell receives the same total inhibition however many cells converge on it.
        # See the gains section of config.py.
        self.pkj_dcn_scale = cfg.pkj_dcn_fitted_at_n_pkj / float(self.conn.pkj_per_dcn.mean())
        self.dcn_io_scale = cfg.dcn_io_fitted_at_n_dcn / float(self.conn.dcn_per_io.mean())
        # DCN -> IO strength per spike. The CUT_DCN_TO_IO lesion just sets this to zero: DCN cells
        # keep firing, but the olive never feels it.
        self.dcn_io_drive = (0.0 if cfg.ablate_dcn_io
                             else cfg.dcn_io_gaba_gain * self.dcn_io_scale)

        # --- the plastic PF -> PKJ weights: one row per Purkinje cell, one column per fibre ---
        self.weights = np.full((n_pkj, cfg.n_pf_per_pkj), cfg.w_init, dtype=float)
        self.plasticity = Plasticity(
            n_pkj, cfg.n_pf_per_pkj, cfg.dt_ms, cfg.ltd_window_ms, cfg.delta_plus, cfg.delta_minus,
            null_window_ms=cfg.null_window_ms, w_min=cfg.w_min, w_max=cfg.w_max,
            cf_source_of_pkj=self.io_of_pkj,   # each PKJ learns from its own climbing fibre only
        )

        self.t_ms = 0.0
        self._plasticity_on = True
        self.log = SimLog()

    # --- recording setup --------------------------------------------------

    def _choose_recorded(self):
        """Pick, at random, which synapses have their weight tracked and which
        PFs have their spikes saved. Each pick is a (PKJ row, fibre column) pair."""
        cfg, n_pkj = self.cfg, self.conn.n_pkj
        n_syn = min(cfg.n_tracked_synapses, n_pkj * cfg.n_pf_per_pkj)
        flat = self.record_rng.choice(n_pkj * cfg.n_pf_per_pkj, size=n_syn, replace=False)
        rows, cols = np.unravel_index(flat, (n_pkj, cfg.n_pf_per_pkj))
        self.tracked_synapses = np.stack([rows, cols], axis=1)          # (n_tracked, 2)

        n_pf_rec = min(cfg.n_pf_recorded, n_pkj * cfg.n_pf_per_pkj)
        flat = self.record_rng.choice(n_pkj * cfg.n_pf_per_pkj, size=n_pf_rec, replace=False)
        rows, cols = np.unravel_index(flat, (n_pkj, cfg.n_pf_per_pkj))
        self.pf_recorded = np.stack([rows, cols], axis=1)               # (n_pf_recorded, 2)

    # --- one timestep -----------------------------------------------------

    def _step(self, record_pf=False):
        """Advance the whole network by one time step (dt_ms). Returns this
        step's spikes: PF (n_pkj x n_pf), PKJ, DCN, and CF events (one per olive cell)."""
        cfg = self.cfg
        dt = cfg.dt_ms

        # 1. PF -> PKJ. Each PKJ gets excitation = gain x (sum of weights of its PFs that spiked).
        pf_spikes = generate_pf_spikes(cfg.pf_rate_hz, dt, self.rng, self.weights.shape)
        self.pkj.exc.add(cfg.pf_pkj_gain * (self.weights * pf_spikes).sum(axis=1))
        pkj_spiked = self.pkj.step()

        # 2. PKJ -> DCN. Each DCN gets inhibition = gain x (number of its PKJ inputs that spiked).
        self.dcn.inh.add(cfg.pkj_dcn_gain * self.pkj_dcn_scale * (self.m_pkj_to_dcn @ pkj_spiked))
        dcn_spiked = self.dcn.step()

        # 3. DCN -> IO. Each olive cell gets inhibition from its DCN inputs that spiked, then steps.
        #    cf_events[i] is True if olive cell i produced a CF event this step.
        self.io_gaba.add(self.dcn_io_drive * (self.m_dcn_to_io @ dcn_spiked))
        cf_events = self.io.step(dt, self.io_gaba.g)
        self.io_gaba.decay()                                         # inhibition fades between spikes

        # 4. CF -> PKJ. A CF event starts a complex spike and pause in the 8 PKJ it contacts
        #    (they happen from the next step; sim/neurons.py), and the learning rule updates
        #    the weights (sim/plasticity.py).
        if cf_events.any():
            self.pkj.trigger_complex_spike(cf_events[self.io_of_pkj])   # cf_events[io_of_pkj]: one True/False per PKJ
        if self._plasticity_on:
            self.plasticity.step(pf_spikes, cf_events, self.weights)

        return pf_spikes, pkj_spiked, dcn_spiked, cf_events

    # --- the run ----------------------------------------------------------

    def run(self, n_trials=None, trial_ms=None, record_every_ms=None, burn_in_s=None, trace_window_s=None,
            raster_paths=None, on_trial_end=None, on_step=None):
        """Burn in, run the trials, record, and return a SimLog. Arguments left
        as None use the values in the config.

        raster_paths: optional {cell type: file path} for CbmSim-format rasters,
                      cell types being "io", "pkj", "dcn" and "pf" (sim/rasters.py).
        on_trial_end: optional function called after every trial as
                      on_trial_end(trial, cf_counts, mean_weight, seconds), e.g. to print
                      progress. seconds is how long the trial took to compute.
        on_step:      optional function called after every step of every trial as
                      on_step(sim, trial, ts, t_ms, pf_spikes, cf_events, pkj_spiked, dcn_spiked),
                      e.g. the live view (sim/live_view.py).
        """
        cfg = self.cfg
        n_trials = cfg.n_trials if n_trials is None else n_trials
        trial_ms = cfg.trial_ms if trial_ms is None else trial_ms
        record_every_ms = cfg.record_every_ms if record_every_ms is None else record_every_ms
        burn_in_s = cfg.burn_in_s if burn_in_s is None else burn_in_s
        trace_window_s = cfg.trace_window_s if trace_window_s is None else trace_window_s

        # Convert times into numbers of steps.
        trial_steps = int(round(trial_ms / cfg.dt_ms))
        n_steps = n_trials * trial_steps
        record_every = max(1, int(round(record_every_ms / cfg.dt_ms)))       # sample weights every this many steps
        n_trace = min(n_steps, int(round(trace_window_s * 1000.0 / cfg.dt_ms)))
        trace_start = n_steps - n_trace                              # voltages are recorded from here to the end
        self._choose_recorded()

        # --- phase 1: burn-in, learning off, nothing recorded ---
        # This lets the cells' voltages settle. The weights are untouched; they still need a few
        # hundred seconds of the main run to reach equilibrium.
        if burn_in_s > 0:
            self._plasticity_on = False
            for _ in range(int(round(burn_in_s * 1000.0 / cfg.dt_ms))):
                self._step()
            self._plasticity_on = True
        self.t_ms = 0.0                                              # t = 0 is the end of burn-in

        # --- set up the recorders and the arrays the samples go into ---
        n_io, n_pkj, n_dcn = cfg.n_io, self.conn.n_pkj, self.conn.n_dcn
        io_rec, pkj_rec, dcn_rec = SpikeRecorder(n_io), SpikeRecorder(n_pkj), SpikeRecorder(n_dcn)
        pf_rec = SpikeRecorder(len(self.pf_recorded))
        pf_rows, pf_cols = self.pf_recorded[:, 0], self.pf_recorded[:, 1]
        tr_rows, tr_cols = self.tracked_synapses[:, 0], self.tracked_synapses[:, 1]

        n_slow = (n_steps + record_every - 1) // record_every
        t_slow = np.empty(n_slow)
        mean_w = np.empty((n_slow, n_io))                      # mean weight per climbing-fiber territory
        sample_w = np.empty((n_slow, len(self.tracked_synapses)))
        pkj_group_slices = [np.flatnonzero(self.io_of_pkj == i) for i in range(n_io)]   # PKJ rows per climbing fibre

        trial_cf = np.zeros((n_trials, n_io), dtype=int)       # CF events per olive cell, per trial
        trial_w = np.empty(n_trials)                           # mean weight at the end of each trial

        # CbmSim-format raster files. The PF raster holds only the recorded PF sample, each
        # identified by a global fibre number: PKJ row x n_pf_per_pkj + fibre column.
        writers = {kind: CbmRasterWriter(path, FORMATS[kind][1])
                   for kind, path in (raster_paths or {}).items()}
        pf_global_id = pf_rows * cfg.n_pf_per_pkj + pf_cols

        trace_t = np.empty(n_trace)
        trace_io_v, trace_io_ca = np.empty((n_trace, n_io)), np.empty((n_trace, n_io))
        trace_io_g = np.empty((n_trace, n_io))
        trace_pkj_v, trace_dcn_v = np.empty((n_trace, n_pkj)), np.empty((n_trace, n_dcn))

        # --- phase 2: the trials ---
        slow_i = 0                                                   # number of weight samples taken so far
        for trial in range(n_trials):
            trial_start = time.time()
            for w in writers.values():
                w.start_trial(trial)
            for ts in range(trial_steps):                            # ts: step within this trial
                i = trial * trial_steps + ts                         # i: step within the whole run
                pf_spikes, pkj_spiked, dcn_spiked, cf_events = self._step()
                self.t_ms += cfg.dt_ms

                # every step: save spikes, in memory and (if asked) to the raster files
                pf_sample = pf_spikes[pf_rows, pf_cols]
                io_rec.record(self.t_ms, cf_events)
                pkj_rec.record(self.t_ms, pkj_spiked)
                dcn_rec.record(self.t_ms, dcn_spiked)
                pf_rec.record(self.t_ms, pf_sample)
                trial_cf[trial] += cf_events
                for kind, spiked in (("io", cf_events), ("pkj", pkj_spiked), ("dcn", dcn_spiked)):
                    if kind in writers:
                        writers[kind].record(ts, spiked)
                if "pf" in writers:
                    writers["pf"].record(ts, pf_global_id[pf_sample])
                if on_step is not None:
                    on_step(self, trial, ts, self.t_ms, pf_spikes, cf_events, pkj_spiked, dcn_spiked)

                # every record_every steps: save the weights
                if i % record_every == 0:
                    t_slow[slow_i] = self.t_ms
                    for g, sl in enumerate(pkj_group_slices):
                        mean_w[slow_i, g] = self.weights[sl].mean()
                    sample_w[slow_i] = self.weights[tr_rows, tr_cols]
                    slow_i += 1

                # last trace_window_s seconds only: save voltages
                if i >= trace_start:
                    j = i - trace_start
                    trace_t[j] = self.t_ms
                    trace_io_v[j], trace_io_ca[j], trace_io_g[j] = self.io.V, self.io.Ca, self.io_gaba.g
                    # An integrate-and-fire cell jumps straight from threshold to reset, so its trace would
                    # show no spikes. Draw a spike as v_peak on the steps where the cell fired.
                    trace_pkj_v[j] = np.where(pkj_spiked, cfg.pkj.v_peak_mv, self.pkj.V)
                    trace_dcn_v[j] = np.where(dcn_spiked, cfg.dcn.v_peak_mv, self.dcn.V)

            # end of trial: write this trial's rasters, note its summary, report it
            for w in writers.values():
                w.end_trial()
            trial_w[trial] = self.weights.mean()
            if on_trial_end is not None:
                on_trial_end(trial, trial_cf[trial], trial_w[trial], time.time() - trial_start)

        # --- pack everything into the SimLog ---
        log = self.log
        log.io_spikes, log.pkj_spikes, log.dcn_spikes = io_rec.as_trains(), pkj_rec.as_trains(), dcn_rec.as_trains()
        log.pf_spikes, log.pf_recorded = pf_rec.as_trains(), self.pf_recorded
        log.t_ms, log.mean_weight, log.sample_weights = t_slow[:slow_i], mean_w[:slow_i], sample_w[:slow_i]
        log.tracked_synapses = self.tracked_synapses
        log.trace_t_ms, log.trace_io_v, log.trace_io_ca = trace_t, trace_io_v, trace_io_ca
        log.trace_io_gaba, log.trace_pkj_v, log.trace_dcn_v = trace_io_g, trace_pkj_v, trace_dcn_v
        log.final_weights = self.weights.copy()
        log.group_of_pkj, log.group_of_dcn = self.group_of_pkj, self.group_of_dcn
        log.io_of_pkj = self.io_of_pkj
        log.gap_matrix, log.duration_ms = self.gap_matrix, n_steps * cfg.dt_ms
        log.n_trials, log.trial_ms = n_trials, trial_steps * cfg.dt_ms
        log.trial_cf_counts, log.trial_mean_weight = trial_cf, trial_w
        log.meta = {"n_io": n_io, "n_dcn": n_dcn, "n_pkj": n_pkj,
                    "n_pkj_per_io": cfg.n_pkj_per_io, "n_pkj_per_dcn": cfg.n_pkj_per_dcn,
                    "n_pf_per_pkj": cfg.n_pf_per_pkj, "gap_topology": cfg.gap_topology, "gap_g": cfg.gap_g,
                    "n_trials": n_trials, "trial_ms": trial_ms, "duration_s": n_steps * cfg.dt_ms / 1000.0,
                    "burn_in_s": burn_in_s, "seed": cfg.seed}
        return log
