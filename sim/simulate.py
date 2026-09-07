"""Main time-stepped simulation loop, and the recording layer.

Assembles the whole microzone -- PF -> PKJ -| DCN -| IO -> CF, with CF both
pausing its Purkinje cells and gating LTD at their PF synapses -- and steps it.

Every population is held FLAT: one PKJPopulation of n_io*n_pkj_per_io cells, one
DCNPopulation, one IOPopulation. The wiring lives entirely in the 0/1 routing
matrices built from sim/connectivity.py, so each projection is a single matrix
multiply against a spike vector rather than a Python loop over cells. That
matters at this scale -- 320 Purkinje cells x 500 fibers is 160,000 plastic
synapses stepped 68,000 times over a default run.

Within a timestep the order is PF -> PKJ -> DCN -> IO -> CF/plasticity, each
population reading its upstream source's *just-updated* spikes. The paper does
not specify an integration order; this semi-implicit choice propagates a step's
effect around the whole loop in one step of latency instead of four.

`SimConfig.burn_in_s` runs the loop with plasticity and recording OFF before
t = 0. Without it every cell starts from rest, the olive transiently overshoots
toward its unopposed rate before DCN and PKJ activity ramps up to inhibit it,
and the resulting burst of spurious early LTD permanently crashes the weights.
That is a startup artifact, not modeled dynamics.
"""
from dataclasses import dataclass, field

import numpy as np

from config import SimConfig
from sim.connectivity import build_connectivity
from sim.io_channels import IOPopulation
from sim.io_coupling import build_gap_junction_matrix
from sim.neurons import DCNPopulation, ExpSynapse, PKJPopulation
from sim.plasticity import Plasticity
from sim.poisson_input import generate_pf_spikes


class SpikeRecorder:
    """Collects (time, cell) pairs as two flat lists and splits them into
    per-cell spike-time arrays at the end. Appending to flat lists costs one
    extend per step regardless of population size; the split happens once."""

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
    """Everything recorded from a run.

    Spike trains are exact -- every spike of every cell, at full dt resolution --
    and are the primary record; firing rates are derived from them in
    sim/analysis.py rather than logged as smoothed traces.

    Two things are deliberately subsampled, for readability rather than cost.
    The PF raster keeps cfg.n_pf_recorded of the 160,000 fibers. Membrane
    potentials are recorded only over cfg.trace_window_s at the END of the run,
    but at full dt there: sampling V at the slow logging cadence would alias
    away every spike and every Ca2+ spike, and sampling it at dt for a whole run
    is a lot of memory for a plot no one could read.
    """
    # --- spike trains: one array of spike times (ms) per cell ---
    io_spikes: list = field(default_factory=list)      # n_io trains -- these are the CF events
    pkj_spikes: list = field(default_factory=list)     # n_io*n_pkj_per_io trains
    dcn_spikes: list = field(default_factory=list)     # n_dcn trains
    pf_spikes: list = field(default_factory=list)      # cfg.n_pf_recorded trains (a subsample; see cfg)
    pf_recorded: np.ndarray = None                     # (n_pf_recorded, 2) the (pkj_row, pf_col) each train came from

    # --- slow traces, sampled every cfg.record_every_ms ---
    t_ms: np.ndarray = None
    mean_weight: np.ndarray = None                     # (T, n_io) mean PF->PKJ weight per climbing-fiber territory
    sample_weights: np.ndarray = None                  # (T, n_tracked) individually tracked synapses
    tracked_synapses: np.ndarray = None                # (n_tracked, 2) the (pkj_row, pf_col) each column came from

    # --- membrane potentials, sampled every dt over the final cfg.trace_window_s ---
    trace_t_ms: np.ndarray = None
    trace_io_v: np.ndarray = None                      # (S, n_io) mV
    trace_io_ca: np.ndarray = None                     # (S, n_io) uM
    trace_pkj_v: np.ndarray = None                     # (S, n_pkj) mV, spikes drawn at pkj.v_peak_mv
    trace_dcn_v: np.ndarray = None                     # (S, n_dcn) mV, spikes drawn at dcn.v_peak_mv
    trace_io_gaba: np.ndarray = None                   # (S, n_io) mS/cm^2, the DCN inhibition each IO is receiving

    # --- final state and bookkeeping ---
    final_weights: np.ndarray = None                   # (n_pkj, n_pf) at the end of the run
    group_of_pkj: np.ndarray = None                    # plot colouring for PKJ: its climbing fiber
    group_of_dcn: np.ndarray = None                    # plot colouring for DCN: its own index (it belongs to no CF)
    io_of_pkj: np.ndarray = None                       # which IO's climbing fiber owns each PKJ
    gap_matrix: np.ndarray = None                      # (n_io, n_io) coupling conductances the run used
    duration_ms: float = 0.0
    meta: dict = field(default_factory=dict)


class Simulation:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.conn = build_connectivity(
            n_io=cfg.n_io, n_dcn=cfg.n_dcn, n_pkj_per_io=cfg.n_pkj_per_io,
            n_pkj_per_dcn=cfg.n_pkj_per_dcn, n_dcn_per_pkj=cfg.n_dcn_per_pkj,
            n_dcn_per_io=cfg.n_dcn_per_io, enforce_closed_loop=cfg.enforce_closed_loop,
            seed=cfg.connectivity_seed,
        )
        n_io, n_pkj, n_dcn = cfg.n_io, self.conn.n_pkj, self.conn.n_dcn

        # Four independent RNG streams, all derived from cfg.seed. Keeping them separate means that
        # changing one source of randomness -- turning on IO noise, recording different synapses --
        # never shifts the PF spike draws, so two runs stay comparable on everything else. The
        # connectivity and heterogeneity draws are seeded separately again (cfg.connectivity_seed,
        # cfg.io_heterogeneity_seed), so the same network can be re-run under different input seeds.
        seed = cfg.seed
        self.rng = np.random.default_rng(seed)                                          # PF Poisson draws, and nothing else
        io_rng = np.random.default_rng(None if seed is None else [seed, 1])             # IO membrane noise
        lif_rng = np.random.default_rng(None if seed is None else [seed, 2])            # PKJ/DCN membrane noise + V init
        self.record_rng = np.random.default_rng(None if seed is None else [seed, 3])    # which cells/synapses get recorded

        # Which IO's climbing fiber owns each Purkinje cell -- exactly one, as in CbmSim's
        # connectIOtoPC. This is the only grouping the circuit has: PKJ->DCN and DCN->IO both
        # overlap across territories, so nuclear cells belong to no single climbing fiber. Plots
        # therefore colour PF and PKJ by climbing fiber, and DCN and IO by their own index.
        self.io_of_pkj = self.conn.cf_of_pkj
        self.group_of_pkj = self.io_of_pkj
        self.group_of_dcn = np.arange(n_dcn)

        self.gap_matrix = build_gap_junction_matrix(n_io, cfg.gap_g, cfg.gap_topology, cfg.gap_n_neighbors)
        self.io = IOPopulation(cfg.io_channels, n_io, cfg.dt_ms, rng=io_rng, g_gap=self.gap_matrix,
                               heterogeneity_cv=cfg.io_heterogeneity_cv,
                               heterogeneity_seed=cfg.io_heterogeneity_seed)

        self.pkj = PKJPopulation(
            n_pkj, cfg.dt_ms, cfg.pkj.tau_m_ms, cfg.pkj.v_th_mv, cfg.pkj.v_reset_mv, cfg.pkj.e_leak_mv,
            cfg.pkj.t_ref_ms, cfg.pkj_baseline_hz, cfg.tau_pf_pkj_ms, cfg.tau_pkj_dcn_ms,
            cfg.e_exc_mv, cfg.e_inh_mv, noise_sigma_mv=cfg.pkj.noise_sigma_mv, rng=lif_rng,
            pause_g=cfg.cf_pause_g, pause_ms=cfg.cf_pause_ms, e_pause_mv=cfg.e_inh_mv,
        )
        self.dcn = DCNPopulation(
            n_dcn, cfg.dt_ms, cfg.dcn.tau_m_ms, cfg.dcn.v_th_mv, cfg.dcn.v_reset_mv, cfg.dcn.e_leak_mv,
            cfg.dcn.t_ref_ms, cfg.dcn_baseline_hz, cfg.tau_pf_pkj_ms, cfg.tau_pkj_dcn_ms,
            cfg.e_exc_mv, cfg.e_inh_mv, noise_sigma_mv=cfg.dcn.noise_sigma_mv, rng=lif_rng,
        )
        # DCN -> IO GABA_A lives here rather than on the IO population: the IO's own class takes a
        # conductance per step and knows nothing about where it came from.
        self.io_gaba = ExpSynapse(n_io, cfg.tau_dcn_io_ms, cfg.dt_ms, cfg.io_channels.e_gaba)

        # Routing as 0/1 matrices built straight from the connection tables, so each projection is
        # one matmul against a spike vector.
        self.m_pkj_to_dcn = self.conn.pkj_to_dcn_matrix()   # (n_dcn, n_pkj)
        self.m_dcn_to_io = self.conn.dcn_to_io_matrix()     # (n_io, n_dcn)

        # Both convergent projections are normalized by how many cells actually converge, against the
        # counts the gains were fitted at. This keeps the total inhibition a DCN (and an IO) receives
        # independent of the population ratios, so changing the topology rewires the network without
        # also moving its operating point. Each factor is 1 at the fitted counts.
        #
        # Note this equalizes the MEAN only. It cannot equalize the graininess: N converging cells at a
        # given rate deliver N times as many, N times smaller, events. That distinction matters here --
        # see config.dcn_io_gaba_gain on why the olive's feedback limb depends on grainy inhibition.
        self.pkj_dcn_scale = cfg.pkj_dcn_fitted_at_n_pkj / float(self.conn.pkj_per_dcn.mean())
        self.dcn_io_scale = cfg.dcn_io_fitted_at_n_dcn / float(self.conn.dcn_per_io.mean())
        # Fold the gain, the convergence normalization and the ablation switch into one number.
        # ablate_dcn_io = True sets it to zero, which is the open loop: DCN still spike and are
        # still recorded, their inhibition simply never reaches the olive.
        self.dcn_io_drive = (0.0 if cfg.ablate_dcn_io
                             else cfg.dcn_io_gaba_gain * self.dcn_io_scale)

        self.weights = np.full((n_pkj, cfg.n_pf_per_pkj), cfg.w_init, dtype=float)
        self.plasticity = Plasticity(
            n_pkj, cfg.n_pf_per_pkj, cfg.dt_ms, cfg.ltd_window_ms, cfg.delta_plus, cfg.delta_minus,
            null_window_ms=cfg.null_window_ms, w_min=cfg.w_min, w_max=cfg.w_max,
            cf_source_of_pkj=self.io_of_pkj,   # each PKJ resolves against the one CF that contacts it
        )

        self.t_ms = 0.0
        self._plasticity_on = True
        self.log = SimLog()

    # --- recording setup --------------------------------------------------

    def _choose_recorded(self):
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
        cfg = self.cfg
        dt = cfg.dt_ms

        # --- PF -> PKJ: weighted excitatory conductance from this step's Poisson draw ---
        pf_spikes = generate_pf_spikes(cfg.pf_rate_hz, dt, self.rng, self.weights.shape)
        self.pkj.exc.add(cfg.pf_pkj_gain * (self.weights * pf_spikes).sum(axis=1))
        pkj_spiked = self.pkj.step()

        # --- PKJ -> DCN: inhibitory conductance, one increment per presynaptic spike ---
        self.dcn.inh.add(cfg.pkj_dcn_gain * self.pkj_dcn_scale * (self.m_pkj_to_dcn @ pkj_spiked))
        dcn_spiked = self.dcn.step()

        # --- DCN -> IO: GABA_A conductance on the olivary glomerulus ---
        self.io_gaba.add(self.dcn_io_drive * (self.m_dcn_to_io @ dcn_spiked))
        cf_events = self.io.step(dt, self.io_gaba.g)
        self.io_gaba.decay()

        # --- CF back onto PKJ, and the plasticity it resolves ---
        if cf_events.any():
            self.pkj.trigger_cf_pause(cf_events[self.io_of_pkj])   # each PKJ is paused by its own climbing fiber
        if self._plasticity_on:
            self.plasticity.step(pf_spikes, cf_events, self.weights)

        return pf_spikes, pkj_spiked, dcn_spiked, cf_events

    # --- the run ----------------------------------------------------------

    def run(self, duration_s=None, record_every_ms=None, burn_in_s=None, trace_window_s=None):
        cfg = self.cfg
        duration_s = cfg.duration_s if duration_s is None else duration_s
        record_every_ms = cfg.record_every_ms if record_every_ms is None else record_every_ms
        burn_in_s = cfg.burn_in_s if burn_in_s is None else burn_in_s
        trace_window_s = cfg.trace_window_s if trace_window_s is None else trace_window_s

        n_steps = int(round(duration_s * 1000.0 / cfg.dt_ms))
        record_every = max(1, int(round(record_every_ms / cfg.dt_ms)))
        n_trace = min(n_steps, int(round(trace_window_s * 1000.0 / cfg.dt_ms)))
        trace_start = n_steps - n_trace                              # the window sits at the END of the run
        self._choose_recorded()

        # --- burn-in: settle the membrane state before plasticity or recording start ---
        # See the module docstring. Note this settles the CELLS, not the weights: the weights still
        # take a few hundred seconds to reach their equilibrium from w_init, which is why drift should
        # be read off runs of 300 s or more (see the README's settling result).
        if burn_in_s > 0:
            self._plasticity_on = False
            for _ in range(int(round(burn_in_s * 1000.0 / cfg.dt_ms))):
                self._step()
            self._plasticity_on = True
        self.t_ms = 0.0                                              # t = 0 is the end of burn-in

        n_io, n_pkj, n_dcn = cfg.n_io, self.conn.n_pkj, self.conn.n_dcn
        io_rec, pkj_rec, dcn_rec = SpikeRecorder(n_io), SpikeRecorder(n_pkj), SpikeRecorder(n_dcn)
        pf_rec = SpikeRecorder(len(self.pf_recorded))
        pf_rows, pf_cols = self.pf_recorded[:, 0], self.pf_recorded[:, 1]
        tr_rows, tr_cols = self.tracked_synapses[:, 0], self.tracked_synapses[:, 1]

        n_slow = (n_steps + record_every - 1) // record_every
        t_slow = np.empty(n_slow)
        mean_w = np.empty((n_slow, n_io))                      # mean weight per climbing-fiber territory
        sample_w = np.empty((n_slow, len(self.tracked_synapses)))
        pkj_group_slices = [np.flatnonzero(self.io_of_pkj == i) for i in range(n_io)]

        trace_t = np.empty(n_trace)
        trace_io_v, trace_io_ca = np.empty((n_trace, n_io)), np.empty((n_trace, n_io))
        trace_io_g = np.empty((n_trace, n_io))
        trace_pkj_v, trace_dcn_v = np.empty((n_trace, n_pkj)), np.empty((n_trace, n_dcn))

        slow_i = 0
        for i in range(n_steps):
            pf_spikes, pkj_spiked, dcn_spiked, cf_events = self._step()
            self.t_ms += cfg.dt_ms

            io_rec.record(self.t_ms, cf_events)
            pkj_rec.record(self.t_ms, pkj_spiked)
            dcn_rec.record(self.t_ms, dcn_spiked)
            pf_rec.record(self.t_ms, pf_spikes[pf_rows, pf_cols])

            if i % record_every == 0:
                t_slow[slow_i] = self.t_ms
                for g, sl in enumerate(pkj_group_slices):
                    mean_w[slow_i, g] = self.weights[sl].mean()
                sample_w[slow_i] = self.weights[tr_rows, tr_cols]
                slow_i += 1

            if i >= trace_start:
                j = i - trace_start
                trace_t[j] = self.t_ms
                trace_io_v[j], trace_io_ca[j], trace_io_g[j] = self.io.V, self.io.Ca, self.io_gaba.g
                # An integrate-and-fire cell resets instead of producing an upstroke, so a raw V trace shows
                # only the sawtooth. Painting v_peak on spiking steps makes the trace read as a spike train.
                trace_pkj_v[j] = np.where(pkj_spiked, cfg.pkj.v_peak_mv, self.pkj.V)
                trace_dcn_v[j] = np.where(dcn_spiked, cfg.dcn.v_peak_mv, self.dcn.V)

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
        log.meta = {"n_io": n_io, "n_dcn": n_dcn, "n_pkj": n_pkj,
                    "n_pkj_per_io": cfg.n_pkj_per_io, "n_pkj_per_dcn": cfg.n_pkj_per_dcn,
                    "n_pf_per_pkj": cfg.n_pf_per_pkj, "gap_topology": cfg.gap_topology, "gap_g": cfg.gap_g,
                    "duration_s": duration_s, "burn_in_s": burn_in_s, "seed": cfg.seed}
        return log
