"""All simulation parameters, defaults, and units."""
from dataclasses import dataclass, field          # config is a plain dataclass, no validation logic
from typing import Optional                       # for fields that can be None (seed)


class _StrictParams:
    """Reject assignment to attributes that are not declared fields.

    These configs are plain dataclasses, so `cfg.duration_ms = 2000` -- a field
    that does not exist, the real one being `duration_s` -- silently creates a
    new attribute and changes nothing about the run. That is survivable when a
    human is reading the output, and dangerous when a sweep sets parameters
    programmatically: the run quietly uses the default while the sweep logs it
    as a swept condition, and the resulting data looks fine but answers a
    different question. Fail loudly instead."""

    def __setattr__(self, name, value):
        if name not in getattr(type(self), "__dataclass_fields__", ()):
            valid = ", ".join(sorted(type(self).__dataclass_fields__))
            raise AttributeError(
                f"{type(self).__name__} has no field {name!r} -- assigning it would "
                f"silently do nothing. Valid fields: {valid}")
        object.__setattr__(self, name, value)


@dataclass
class IOChannelParams(_StrictParams):
    """Ionic parameters for the conductance-based IO neuron (sim/io_channels.py).

    Gating kinetics are hard-coded in io_channels.py from the published olivary
    models (Schweighofer et al. 1999; De Gruijl et al. 2012) -- those are the
    physiology and are not meant to be tuned. What lives here is what a
    single-compartment reduction legitimately has to re-fit: maximal
    conductances, reversal potentials, Ca2+ handling, and the noise level.

    Units: V in mV, conductances in mS/cm^2, currents in uA/cm^2,
    C_m in uF/cm^2, [Ca]i in uM, time in ms.
    """
    # --- maximal conductances (retuned for one compartment; see module docstring) ---
    g_cal: float = 1.1        # low-threshold ("T-type") Ca2+ -- the rebound/oscillation driver
    g_cah: float = 1.0        # high-threshold Ca2+ -- the [Ca]i source that recruits I_KCa
    g_kca: float = 15.0       # Ca2+-activated K+ -- spike repolarization + the pacing AHP
    g_h: float = 0.30         # h current -- depolarizing sag, sets oscillation frequency
    g_leak: float = 0.06      # passive leak. On its own this is a 17 ms membrane time constant, but the cell
                              # has no true resting state: the effective tau swings from ~0.2 ms during the
                              # I_KCa-dominated AHP to ~7 ms just before a spike, as I_KCa deactivates.

    # --- reversal potentials ---
    e_ca: float = 120.0       # Ca2+ (both Ca currents)
    e_k: float = -75.0        # K+ (I_KCa)
    e_h: float = -43.0        # mixed cation (I_h)
    e_leak: float = -60.0     # leak. The source models use a depolarizing leak (+10 mV) balanced by somatic
                              # currents this reduction drops; keeping it here made the LEAK the pacemaker and left
                              # firing nearly unchanged when I_CaL was blocked. A conventional -60 mV leak restores
                              # I_CaL as the current that actually generates the rhythm -- see
                              # experiments/run_io_characterization.py's ablation panel, which checks exactly that.
    e_gaba: float = -75.0     # Cl- (DCN's GABA_A synapse onto the olivary glomerulus)

    c_m: float = 1.0          # membrane capacitance
    i_app: float = 0.0        # constant applied current; 0 = the cell is driven by its own channels alone

    # --- Ca2+ handling: d[Ca]/dt = -ca_influx * I_CaH - ca_decay * [Ca] ---
    ca_influx: float = 0.003  # uM per (uA/cm^2) of I_CaH per ms. Scaled down 1000x from the source models (and
                              # kca_ca_scale up 1000x to match, leaving the dynamics identical) purely so [Ca]i reads
                              # in real units: ~3 uM at the peak of a Ca2+ spike instead of a nominal ~13000.
    ca_decay: float = 0.075   # first-order extrusion rate, 1/ms (tau ~ 13 ms)

    # --- I_KCa [Ca]-dependence (alpha_s = min(ca_scale*[Ca], alpha_max)) ---
    kca_ca_scale: float = 0.02
    kca_alpha_max: float = 0.01
    kca_beta: float = 0.015

    cal_tau_k_ms: float = 1.0  # I_CaL activation time constant (fast; near-instantaneous vs. the rest)

    # --- membrane noise: OU current standing in for synaptic bombardment / channel noise ---
    noise_sigma: float = 0.0  # stationary SD of the noise current, uA/cm^2. ZERO by default: the cell's firing is
                              # then generated entirely by its own ionic currents, with no applied current
                              # (i_app = 0), no bias and no jitter -- it free-runs at 3.00 Hz on I_CaL/I_KCa/I_h
                              # alone, and blocking I_CaL leaves it silent at -62.6 mV.
                              # Set > 0 (2.0 gives ~3.5 mV of Vm jitter) to restore stochastic spike gating, which
                              # smooths how steeply CF rate falls off with inhibition. See the README for what the
                              # noiseless feedback limb looks like by comparison.
    noise_tau_ms: float = 5.0 # OU correlation time

    # --- CF event detection ---
    v_spike_mv: float = -35.0       # upward crossing of this = one Ca2+ spike = one CF event
    spike_refractory_ms: float = 15.0  # blocks counting a single Ca2+ spike as several CF events

    # --- initial conditions / integration ---
    v_init_mv: float = -60.0
    ca_init_um: float = 0.0   # no basal Ca2+ influx or buffering is modeled, so [Ca]i decays to ~0 between
                              # spikes; the model tracks the spike-evoked transient, which is what gates I_KCa.
    sub_dt_ms: float = 0.1          # internal integration step; the outer loop still runs at cfg.dt_ms
    display_tau_ms: float = 3000.0  # smoothing constant for the LOGGED rate estimate only


@dataclass
class LIFParams(_StrictParams):
    """Conductance-based leaky-integrate-and-fire parameters, shared shape for
    PKJ and DCN (sim/neurons.py). Units: V in mV, time in ms, conductances in
    1/ms (C_m is normalized to 1, so g_leak is just 1/tau_m and currents are in
    mV/ms).

    None of these come from the white paper, which specifies population ratios
    and plasticity constants but no membrane model. They are conventional
    cortical-neuron values; what was actually fitted for this circuit is the
    noise level and the synaptic gains in SimConfig.
    """
    tau_m_ms: float = 20.0        # membrane time constant
    v_th_mv: float = -50.0        # spike threshold
    v_reset_mv: float = -65.0     # post-spike reset
    e_leak_mv: float = -70.0      # leak reversal / resting potential with no input
    t_ref_ms: float = 2.0         # absolute refractory period
    v_peak_mv: float = 0.0        # NOT dynamics: the value a spike is drawn at in logged V traces, since an
                                  # integrate-and-fire cell resets rather than producing an upstroke of its own.
    noise_sigma_mv: float = 3.0   # SD of the OU membrane noise. See sim/neurons.py: this is what keeps the
                                  # f-I curve smooth near rheobase, and so what keeps the loop's feedback graded.


@dataclass
class SimConfig(_StrictParams):
    # --- timing ---
    dt_ms: float = 1.0                # simulation timestep, milliseconds
    duration_s: float = 60.0          # default run length if Simulation.run() isn't given one

    # --- population sizes and connectivity (CbmSim's microzone; see sim/connectivity.py) ---
    # Connection numbers are CbmSim's own, from src/cbm_state/connectivityparams.cpp; the three
    # POPULATION sizes are those numbers scaled up 10x, while per-cell convergence is not scaled at
    # all. The counts stay mutually consistent at this scale (320 PKJ x 3 targets = 960 = 80 DCN x
    # 12 inputs) and they satisfy the white paper's N_IO << N_DCN << N_PKJ ordering (40 < 80 < 320)
    # without needing any assumption the paper does not state.
    n_io: int = 40                    # CbmSim num_io (4) x10
    n_dcn: int = 80                   # CbmSim num_nc (8) x10
    n_pkj_per_io: int = 8             # CbmSim num_p_io_from_io_to_pc -- PKJ per climbing fiber.
                                      # n_pkj = n_io * n_pkj_per_io = 320 (CbmSim num_pc x10), and
                                      # every Purkinje cell has exactly one climbing fiber.
    n_pkj_per_dcn: int = 12           # CbmSim num_p_nc_from_pc_to_nc -- PKJ converging on one DCN
    n_dcn_per_pkj: int = 3            # CbmSim num_p_pc_from_pc_to_nc -- DCN reached by one PKJ
    n_dcn_per_io: int = 8             # CbmSim num_p_io_from_nc_to_io -- DCN converging on one IO.
                                      # At CbmSim's own scale this equals n_dcn, i.e. connectNCtoIO is
                                      # COMPLETE and every olivary cell sees identical inhibition. Held
                                      # fixed while the populations grew 10x, it is now 8 of 80: each
                                      # olivary cell reads its own topographic block of the nucleus.
                                      # That is what stops the olive collapsing to one repeated cell.
    n_pf_per_pkj: int = 500           # THE ONE SCALED-DOWN NUMBER. CbmSim gives each Purkinje cell
                                      # 32,768 of a million shared granule cells; this MVP gives it a
                                      # private pool of 500 Poisson fibers, which is the spec's own
                                      # simplification (private pools keep each cell's coincidence
                                      # detection independent, which is what H2 measures).
    # --- circuit ablation ---
    ablate_dcn_io: bool = False        # cut the nucleo-olivary projection: DCN spikes stop delivering GABA to the
                                       # olive, so the loop is OPEN. Everything else (wiring, plasticity, CF->PKJ
                                       # teaching, gap junctions) is untouched, which is what makes this a clean
                                       # test of H1: if the ~1 Hz equilibrium is really produced by the feedback
                                       # limb, removing only that limb should abolish it -- CF rate runs free to
                                       # the cell's intrinsic rate and the weights should be driven to a bound
                                       # instead of settling mid-range. If the rate survived the cut, the
                                       # equilibrium would have been an artifact of the operating point rather
                                       # than of feedback.

    # --- olivary cell-to-cell heterogeneity ---
    io_heterogeneity_cv: float = 0.0   # coefficient of variation of the per-cell maximal conductances
                                       # g_cal / g_kca / g_h (see IOPopulation._draw_conductances).
                                       # 0.0 = every olivary cell identical, which is CbmSim's assumption
                                       # and which -- combined with a complete DCN->IO projection and
                                       # noise_sigma = 0 -- made the cells integrate to bit-identical
                                       # traces. Real olivary cells differ; this is that difference, drawn
                                       # ONCE at construction from io_heterogeneity_seed, so the population
                                       # is fixed and the cells stay deterministic. It is variability
                                       # between cells, NOT noise within a cell: no applied drive, no bias,
                                       # no jitter is added by setting it.
    io_heterogeneity_seed: int = 0     # seeds the draw above, independently of `seed` and connectivity_seed,
                                       # so the same cell population can be re-used across input seeds

    connectivity_seed: int = 0        # seeds the randomized half of the PKJ->DCN overlap

    # --- PF Poisson input ---
    pf_rate_hz: float = 20.0          # per-PF-unit Poisson rate, Hz

    # --- PKJ / DCN membrane models ---
    pkj: "LIFParams" = field(default_factory=lambda: LIFParams())
    dcn: "LIFParams" = field(default_factory=lambda: LIFParams())

    # --- PKJ / DCN baseline rates (what each cell fires at with no synaptic input) ---
    pkj_baseline_hz: float = 50.0     # PKJ tonic rate before PF excitation / CF pause
    dcn_baseline_hz: float = 60.0     # DCN tonic rate BEFORE PKJ inhibition, so well above the ~15 Hz it actually
                                      # runs at in the loop. Deliberately so: a LIF sitting only just above rheobase
                                      # has a near-vertical f-I curve, and the nucleus could not both sit at 15 Hz and
                                      # respond gradually to PKJ. Driving it high and inhibiting it back down puts the
                                      # operating point on a gentler part of the curve, which is what the feedback limb
                                      # needs. Lower is gentler: 60/70/80/90 Hz give DCN-vs-PKJ slopes of
                                      # -0.66/-0.78/-0.87/-0.95 Hz per Hz, so the lowest was taken.
                                      # See experiments/run_calibration.py stage 2.

    # --- synaptic reversal potentials and time constants ---
    e_exc_mv: float = 0.0             # AMPA-like (PF -> PKJ)
    e_inh_mv: float = -75.0           # GABA_A / Cl- (PKJ -> DCN, and the CF pause on PKJ)
    tau_pf_pkj_ms: float = 5.0        # PF -> PKJ excitatory conductance decay
    tau_pkj_dcn_ms: float = 10.0      # PKJ -> DCN inhibitory conductance decay
    tau_dcn_io_ms: float = 50.0       # DCN -> IO inhibitory conductance decay, matching the olivary glomerulus'
                                      # slow GABA response. Deliberately NOT made longer: a longer tau averages the
                                      # 8 converging DCN inputs into a steadier conductance, which pushes the IO back
                                      # toward the near-vertical constant-conductance response it cannot regulate
                                      # against. See dcn_io_gaba_gain for that curve and why it matters.

    # --- IO conductance model (sim/io_channels.py IOPopulation) ---
    io_channels: "IOChannelParams" = field(default_factory=lambda: IOChannelParams())  # ionic parameters; see above

    # --- IO-IO gap junctions (sim/io_coupling.py) ---
    gap_g: float = 0.0143             # conductance of ONE gap junction, mS/cm^2. Set 0.0 to run the cells uncoupled.
                                      # What matters physiologically is a cell's TOTAL coupling conductance --
                                      # gap_g times its number of partners -- not gap_g alone. CbmSim couples its
                                      # 4 olivary cells all-to-all (num_p_io_in_io_to_io = 3), giving 0.019 x 3 =
                                      # 0.057, about 0.95x g_leak. At n_io = 40 all-to-all would mean 39 partners
                                      # and ~12x leak, which is not a physiological regime and would clamp the
                                      # whole olive to one potential; the topology is local instead (4 partners),
                                      # and gap_g is set to 0.057 / 4 to hold that same 0.95x leak per cell.
                                      # Re-scale this if you change n_io or the topology: run_gap_sweep.py shows
                                      # the loop starts to degrade once the total passes roughly 2x leak.
    gap_topology: str = "nearest_k"   # "all_to_all" (CbmSim's connectIOtoIO) | "ring" | "nearest_k"
    gap_n_neighbors: int = 2          # neighbours per side under "nearest_k"; ignored by the other topologies

    # --- coupling gains between populations ---
    # All three are conductance increments per presynaptic event, and all three were fitted by
    # experiments/run_calibration.py rather than guessed -- see its docstring for the targets.
    #
    # The two convergent gains are normalized by how many cells actually converge, against the counts
    # they were fitted at (below). Without that, changing a population ratio would silently change the
    # loop's operating point as well as its wiring, and every result would need re-fitting for every
    # topology. sim/simulate.py applies the scaling.
    pkj_dcn_fitted_at_n_pkj: int = 8      # pkj_dcn_gain was fitted with 8 PKJ converging on each DCN
    dcn_io_fitted_at_n_dcn: int = 2       # dcn_io_gaba_gain was fitted with 2 DCN converging on each IO
    pf_pkj_gain: float = 0.000137     # PKJ excitatory conductance (1/ms) added per unit of (weight * PF spike).
                                      # Fitted so PKJ runs 50 Hz at weight 0 and 70 Hz at weight 1. It is the SPAN
                                      # that matters: the loop has to be able to push the Purkinje rate in both
                                      # directions from w_init = 0.5, or the weights would have nowhere to settle.
    pkj_dcn_gain: float = 0.007717    # DCN inhibitory conductance (1/ms) added per presynaptic PKJ spike.
                                      # Fitted so DCN sits at 15 Hz when the 12 PKJ converging on it (CbmSim's
                                      # num_p_nc_from_pc_to_nc) fire at 60 Hz, i.e. at weight 0.5.
                                      # Scaled by pkj_dcn_fitted_at_n_pkj / n_pkj_per_dcn in sim/simulate.py.
    dcn_io_gaba_gain: float = 0.0974  # IO GABA_A conductance (mS/cm^2) added per presynaptic DCN spike. Fitted so the
                                      # IO fires 1 Hz at DCN = 15 Hz, with CbmSim's 8 nuclear cells converging on each
                                      # olivary cell. It has been re-fitted twice, and the sequence is informative:
                                      # 0.68 (2 DCN converging, with membrane noise), 0.273 (2 DCN, noiseless), 0.0974
                                      # (8 DCN, noiseless). Each step needs LESS inhibition per synapse, for the same
                                      # underlying reason -- more converging cells, and less membrane noise, both make
                                      # the conductance the cell sees steadier, and a steadier conductance suppresses
                                      # the olive more effectively. Re-run experiments/run_calibration.py after
                                      # changing either; it reads the live config, so it re-derives rather than going
                                      # stale.
                                      #
                                      # WHY THIS IS A SPIKE-DRIVEN SYNAPSE AND NOT A SCALAR. With noise off and the
                                      # conductance HELD CONSTANT, the olive fires 3.00 Hz at g = 0 and is silent at
                                      # every g >= 0.10 mS/cm^2. There is no graded range at all -- steady inhibition
                                      # either leaves the rhythm untouched or abolishes it, so a loop cannot regulate
                                      # against it. Delivered as discrete DCN events the same means give a smooth
                                      # monotone curve: 2.73 Hz at mean g = 0.029, 2.09 at 0.088, 1.05 at 0.147 (the
                                      # loop's operating point), 0.21 at 0.207, 0.01 at 0.297. The cell escapes during
                                      # the troughs of a conductance that swings around its mean, and the loop's
                                      # entire negative-feedback limb is that fluctuation. Both curves are plotted on
                                      # one axis by experiments/run_io_characterization.py.
                                      #
                                      # A corollary worth knowing before changing n_dcn_per_io or tau_dcn_io_ms:
                                      # anything that makes the inhibition SMOOTHER (more converging cells, a longer
                                      # tau) narrows the usable limb back toward that cliff. There is an upper bound
                                      # on nucleo-olivary convergence beyond which the loop cannot regulate itself
                                      # without some other noise source -- which is a real prediction of the model.

    # --- CF -> PKJ inhibitory pause ---
    cf_pause_g: float = 1.0           # inhibitory conductance (1/ms) switched on during the pause -- large enough
                                      # to dominate g_leak and clamp the cell near e_inh_mv, i.e. a real pause
    cf_pause_ms: float = 10.0         # pause duration after each CF event, ms

    # --- PF -> PKJ synaptic weights ---
    w_init: float = 0.5               # initial weight, all synapses
    w_min: float = 0.0                # lower clip bound
    w_max: float = 1.0                # upper clip bound

    # --- plasticity ---
    ltd_window_ms: float = 100.0      # window after a PF spike in which a CF event causes LTD
    null_window_ms: float = 0.0       # window after the LTD window with no weight change; 0 = 2-window mode
    delta_plus: float = 0.001         # LTP increment
    delta_minus: float = 0.009        # LTD decrement

    # --- closed-loop connectivity ---
    enforce_closed_loop: bool = True  # False rotates each olivary cell's block of the nucleus by a full block width,
                                      # so it is inhibited by nuclear cells its own climbing fiber does not drive --
                                      # the feedback is misrouted rather than removed. Needs n_dcn_per_io < n_dcn to
                                      # mean anything (at CbmSim's complete projection there is nothing to rotate,
                                      # and build_connectivity reports that in meta["loop"]). For removing the limb
                                      # outright, see ablate_dcn_io above.

    # --- recording ---
    record_every_ms: float = 10.0     # cadence for the slow traces (mean weight, tracked synapses)
    trace_window_s: float = 3.0       # length of the high-resolution membrane-potential window, taken at the END
                                      # of the run. V is sampled every dt there; sampling it at record_every_ms
                                      # for the whole run would alias every spike and every Ca2+ spike away.
    n_pf_recorded: int = 40           # PF units whose spike trains are kept for the raster. All 160,000 fibers
                                      # (320 PKJ x 500) at 20 Hz would be ~190M spike times over the default 60 s
                                      # run, for a raster no one could read.
    n_tracked_synapses: int = 15      # individual PF->PKJ synapses logged alongside the mean weight

    # --- misc ---
    seed: Optional[int] = None        # RNG seed. Seeds PF Poisson draws; the IO membrane noise, the PKJ/DCN
                                      # membrane noise and the recording subsamples each draw from their own
                                      # derived stream, so changing one never shifts the others.
    burn_in_s: float = 8.0            # duration run before plasticity/logging start, to let the loop settle
