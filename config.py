"""Every parameter of the model, in one place.

There are three parameter groups, one per class:

  IOChannelParams  the inferior-olive cell's ion channels   (used by sim/io_channels.py)
  LIFParams        the Purkinje and DCN cell membranes       (used by sim/neurons.py)
  SimConfig        everything else: cell counts, wiring, synapse strengths,
                   the learning rule, recording. It contains one IOChannelParams
                   and two LIFParams (one for PKJ, one for DCN).

run.py fills in a SimConfig from its settings block; anything run.py does not
set keeps the default written here.

Units used throughout: time in ms, voltage in mV, rates in Hz.
"""
from dataclasses import dataclass, field
from typing import Optional


class _StrictParams:
    """Make a typo in a parameter name an error instead of a silent no-op.

    A plain dataclass lets you write `cfg.n_trial = 100` (the real field is
    `n_trials`) and quietly creates a new, unused attribute. Every config
    class below inherits from this one, so that mistake raises instead."""

    def __setattr__(self, name, value):
        if name not in getattr(type(self), "__dataclass_fields__", ()):
            valid = ", ".join(sorted(type(self).__dataclass_fields__))
            raise AttributeError(
                f"{type(self).__name__} has no field {name!r} -- assigning it would "
                f"silently do nothing. Valid fields: {valid}")
        object.__setattr__(self, name, value)


@dataclass
class IOChannelParams(_StrictParams):
    """Parameters of one inferior-olive (IO) cell.

    The IO cell is a single compartment with five ionic currents, and together
    they make it oscillate and fire on its own at about 3 Hz with no input.
    The voltage dependence of each channel is written into sim/io_channels.py
    (it comes from published olive models). What can be tuned is here: how many
    channels of each kind there are (maximal conductances), the reversal
    potentials, calcium handling and noise.

    Units: V in mV, conductances in mS/cm^2, currents in uA/cm^2,
    capacitance in uF/cm^2, [Ca]i in uM, time in ms.
    """
    # --- maximal conductances: how strong each current can get ---
    g_cal: float = 1.1        # I_CaL, low-threshold Ca2+: produces the Ca2+ spike (the CF event)
    g_cah: float = 1.0        # I_CaH, high-threshold Ca2+: lets Ca2+ into the cell during a spike
    g_kca: float = 15.0       # I_KCa, Ca2+-activated K+: ends the spike and holds the cell down afterwards
    g_h: float = 0.30         # I_h, the "sag" current: slowly pulls the cell back up, setting the rhythm
    g_leak: float = 0.06      # passive leak

    # --- reversal potentials: the voltage each current pushes the cell toward ---
    e_ca: float = 120.0       # Ca2+
    e_k: float = -75.0        # K+
    e_h: float = -43.0        # I_h (mixed cations)
    e_leak: float = -60.0     # leak
    e_gaba: float = -75.0     # GABA_A (Cl-), the inhibition arriving from the DCN

    c_m: float = 1.0          # membrane capacitance
    i_app: float = 0.0        # constant injected current. 0: the cell runs on its own channels only

    # --- intracellular calcium: d[Ca]/dt = -ca_influx * I_CaH - ca_decay * [Ca] ---
    ca_influx: float = 0.003  # how much [Ca]i rises per unit of I_CaH current
    ca_decay: float = 0.075   # how fast Ca2+ is pumped out, 1/ms (time constant ~13 ms)

    # --- how [Ca]i opens I_KCa: opening rate = min(kca_ca_scale * [Ca], kca_alpha_max) ---
    kca_ca_scale: float = 0.02
    kca_alpha_max: float = 0.01
    kca_beta: float = 0.015   # closing rate

    cal_tau_k_ms: float = 1.0  # how fast I_CaL activates (fast, ~1 ms)

    # --- optional membrane noise (an Ornstein-Uhlenbeck random current) ---
    noise_sigma: float = 0.0  # size of the noise, uA/cm^2. 0 = off (default): the cell is fully
                              # deterministic. 2.0 gives a few mV of voltage jitter.
    noise_tau_ms: float = 5.0 # how quickly the noise changes

    # --- turning the voltage trace into CF events ---
    v_spike_mv: float = -35.0       # V rising through this level counts as one CF event
    spike_refractory_ms: float = 15.0  # ignore further crossings for this long, so one spike counts once

    # --- starting state and integration ---
    v_init_mv: float = -60.0
    ca_init_um: float = 0.0
    sub_dt_ms: float = 0.1          # the IO is integrated in 0.1 ms sub-steps inside each 1 ms main step
    display_tau_ms: float = 3000.0  # smoothing for a rate readout that nothing in the model uses


@dataclass
class LIFParams(_StrictParams):
    """Membrane parameters for a leaky integrate-and-fire cell (sim/neurons.py).
    One copy is used for Purkinje cells and one for DCN cells.

    The voltage relaxes toward rest. When it reaches threshold the cell spikes,
    is reset, and is held there for a short refractory period.
    Units: V in mV, time in ms.
    """
    tau_m_ms: float = 20.0        # membrane time constant: how quickly V relaxes
    v_th_mv: float = -50.0        # spike threshold
    v_reset_mv: float = -65.0     # V right after a spike
    e_leak_mv: float = -70.0      # resting potential
    t_ref_ms: float = 2.0         # refractory period after each spike
    v_peak_mv: float = 0.0        # only used for drawing: spikes are plotted at this voltage
    noise_sigma_mv: float = 3.0   # size of random voltage noise. It smooths the cell's response so
                                  # its rate changes gradually with input rather than switching on/off.


@dataclass
class SimConfig(_StrictParams):
    """Everything about one simulation run."""

    # --- timing ---
    dt_ms: float = 1.0                # main time step, ms
    n_trials: int = 12                # number of trials (recorded run = n_trials x trial_ms)
    trial_ms: float = 5000.0          # length of one trial, ms (CbmSim's trialTime)

    # --- cell counts and wiring (built in sim/connectivity.py) ---
    # The counts follow CbmSim, the lab's reference cerebellum simulator, with the
    # population sizes scaled up 10x. Defaults: 40 IO, 80 DCN, 320 PKJ.
    n_io: int = 40                    # olive cells = climbing fibres
    n_dcn: int = 80                   # deep nuclear cells
    n_pkj_per_io: int = 8             # PKJ contacted by each climbing fibre. Total PKJ = n_io * n_pkj_per_io,
                                      # and each PKJ has exactly one climbing fibre.
    n_pkj_per_dcn: int = 12           # PKJ inhibiting each DCN cell
    n_dcn_per_pkj: int = 3            # DCN cells each PKJ inhibits
    n_dcn_per_io: int = 8             # DCN cells inhibiting each olive cell. Each olive cell gets its own
                                      # block of 8 DCN, so different olive cells receive different inhibition.
    n_pf_per_pkj: int = 500           # parallel fibres per PKJ. Each PKJ has its own private set, so
                                      # 320 x 500 = 160,000 plastic synapses in total.

    # --- lesion ---
    ablate_dcn_io: bool = False       # True cuts the DCN -> IO connection (the "open loop" control). DCN cells
                                      # still fire, but their inhibition never reaches the olive. Nothing else
                                      # changes, so any difference from the intact run is due to the feedback.

    # --- making olive cells differ from each other ---
    io_heterogeneity_cv: float = 0.0  # spread of g_cal, g_kca and g_h across olive cells (0.15 = 15%).
                                      # 0 = all olive cells identical. The values are drawn once at the start
                                      # and then fixed, so this is cell-to-cell variety, not noise over time.
    io_heterogeneity_seed: int = 0    # random seed for that draw

    connectivity_seed: int = 0        # random seed for the random part of the PKJ -> DCN wiring

    # --- parallel-fibre input ---
    pf_rate_hz: float = 20.0          # firing rate of every parallel fibre, Hz

    # --- Purkinje and DCN membranes ---
    pkj: "LIFParams" = field(default_factory=lambda: LIFParams())
    dcn: "LIFParams" = field(default_factory=lambda: LIFParams())

    # --- spontaneous rates (what each cell fires at with no synaptic input) ---
    # sim/neurons.py works out the constant input current that produces each rate.
    pkj_baseline_hz: float = 50.0     # PKJ
    dcn_baseline_hz: float = 60.0     # DCN. Set high on purpose: PKJ inhibition brings it down to ~15 Hz in
                                      # the running loop, a range where DCN responds smoothly to PKJ input.

    # --- synapses: reversal potentials and decay times ---
    # Each synapse is a conductance that jumps up on every presynaptic spike and decays exponentially.
    e_exc_mv: float = 0.0             # excitatory reversal (PF -> PKJ)
    e_inh_mv: float = -75.0           # inhibitory reversal (PKJ -> DCN, and the post-complex-spike pause)
    tau_pf_pkj_ms: float = 5.0        # PF -> PKJ decay time
    tau_pkj_dcn_ms: float = 10.0      # PKJ -> DCN decay time
    tau_dcn_io_ms: float = 50.0       # DCN -> IO decay time (slow GABA in the olive)

    # --- the olive cell model ---
    io_channels: "IOChannelParams" = field(default_factory=lambda: IOChannelParams())

    # --- gap junctions between olive cells (built in sim/io_coupling.py) ---
    gap_g: float = 0.0143             # conductance of ONE junction, mS/cm^2. 0 = uncoupled.
                                      # Each cell has 4 partners by default, so its total coupling is
                                      # 4 x 0.0143 = 0.057, about the same as its leak (0.06).
    gap_topology: str = "nearest_k"   # who is coupled to whom: "nearest_k" | "ring" | "all_to_all"
                                      # (sim/io_coupling.py also has "small_world")
    gap_n_neighbors: int = 2          # for "nearest_k": partners on each side of the ring

    # --- synaptic strengths (gains) ---
    # Each gain is how much conductance one presynaptic spike adds. They were fitted so the loop
    # runs at PKJ ~60 Hz, DCN ~15 Hz and IO ~1 Hz when the weights are at 0.5.
    #
    # The two convergent gains (PKJ->DCN and DCN->IO) are rescaled in sim/simulate.py by
    # (count below / actual number of converging cells). That way, if you change how many cells
    # converge, the TOTAL input each cell gets stays the same and the loop keeps its operating point.
    pkj_dcn_fitted_at_n_pkj: int = 8      # pkj_dcn_gain is "per spike, when 8 PKJ converge"
    dcn_io_fitted_at_n_dcn: int = 2       # dcn_io_gaba_gain is "per spike, when 2 DCN converge"
    pf_pkj_gain: float = 0.000137     # PF -> PKJ, per (weight x PF spike). With it, PKJ fires 50 Hz when all
                                      # weights are 0 and 70 Hz when all are 1, so learning can push the
                                      # rate either way from the starting weight of 0.5.
    pkj_dcn_gain: float = 0.007717    # PKJ -> DCN, per PKJ spike
    dcn_io_gaba_gain: float = 0.0974  # DCN -> IO, per DCN spike (mS/cm^2).
                                      # This inhibition is delivered spike by spike rather than as a smooth
                                      # constant on purpose. A constant inhibition either leaves the olive
                                      # firing at 3 Hz or silences it, with almost nothing in between.
                                      # Spike-by-spike inhibition rises and falls, the olive fires in the
                                      # dips, and its rate then falls smoothly as inhibition grows. That
                                      # smooth fall is what the feedback loop needs to regulate the olive.

    # --- what a CF event does to its Purkinje cells (sim/neurons.py, PKJPopulation) ---
    cf_burst_spikes: int = 3          # complex spike: this many forced PKJ spikes. 0 = no complex spike
    cf_burst_isi_ms: float = 2.0      # time between them
    cf_pause_g: float = 1.0           # then a pause: a large conductance at e_inh_mv holding V down...
    cf_pause_ms: float = 10.0         # ...for this long after the last complex-spike spike

    # --- PF -> PKJ weights ---
    w_init: float = 0.5               # starting weight of every synapse
    w_min: float = 0.0                # weights are clipped to [w_min, w_max]
    w_max: float = 1.0

    # --- learning rule (sim/plasticity.py) ---
    ltd_window_ms: float = 100.0      # CF within this long after a PF spike -> that synapse is weakened
    null_window_ms: float = 0.0       # optional window after that with no change; 0 = off
    delta_plus: float = 0.001         # LTP step
    delta_minus: float = 0.009        # LTD step

    # --- wiring variant ---
    enforce_closed_loop: bool = True  # False shifts each olive cell's block of DCN inputs by one block, so
                                      # it is inhibited by DCN cells its own climbing fibre does not drive.
                                      # The feedback is misrouted rather than removed (compare ablate_dcn_io).

    # --- what gets recorded ---
    record_every_ms: float = 10.0     # how often the weights are sampled
    trace_window_s: float = 3.0       # membrane voltages are recorded at every time step, but only for this
                                      # many seconds at the very end of the run
    n_pf_recorded: int = 40           # how many of the 160,000 PFs have their spikes saved (for the raster)
    n_tracked_synapses: int = 15      # how many individual synapses have their weight saved over time

    # --- misc ---
    seed: Optional[int] = None        # random seed. None = different every run. Separate random streams are
                                      # derived from it for each use (see sim/simulate.py).
    burn_in_s: float = 8.0            # run this long with learning and recording off before t = 0
