"""PKJ and DCN neurons: conductance-based leaky integrate-and-fire.

The two populations share one model and differ only in their parameters and in
what they are wired to; PKJPopulation adds the climbing-fiber pause, DCN adds
nothing. The third cell type, the olive, is not an integrate-and-fire cell at
all -- it has its own ionic machinery in sim/io_channels.py.

Model, per cell:

    C dV/dt = -g_L (V - E_L) - g_exc (V - E_exc) - g_inh (V - E_inh) + I_tonic

with C = 1 and g_L = 1/tau_m, so conductances are in units of 1/ms and I_tonic
in mV/ms. V >= v_th emits a spike, resets to v_reset, and holds there for
t_ref. Synaptic conductances are single-exponential (`ExpSynapse`): each
presynaptic spike adds a fixed increment, which then decays with tau_syn.

Integration is exponential Euler on V, matching sim/io_channels.py: with the
conductances held over the step, dV/dt is linear in V, so

    V <- V_inf + (V - V_inf) exp(-dt/tau_eff),
    V_inf = (g_L E_L + g_exc E_exc + g_inh E_inh + I_tonic) / g_tot,
    tau_eff = C / g_tot.

That is exact for constant conductance and, unlike forward Euler, cannot
overshoot when a large synaptic conductance makes tau_eff shorter than dt --
which happens on every CF pause.

Baseline rates are configured directly as rates (`pkj_baseline_hz`,
`dcn_baseline_hz`) rather than as currents: `tonic_drive_for_rate` inverts the
LIF f-I curve to find the I_tonic that makes an otherwise-unsynapsed cell fire
at that rate. So `pkj_baseline_hz = 50` really does mean 50 Hz, and re-tuning
the membrane parameters does not silently move every baseline with it.
"""
import numpy as np


def tonic_drive_for_rate(target_hz, tau_m_ms, v_th_mv, v_reset_mv, e_leak_mv, t_ref_ms, dt_ms=0.0):
    """The constant I_tonic (mV/ms) that makes an unsynapsed LIF cell fire at
    target_hz.

    Between spikes V relaxes toward V_inf = E_L + I_tonic*tau_m, so the
    interspike interval is t_ref + tau_m*ln((V_inf - V_reset)/(V_inf - V_th)).
    Inverting for V_inf with a = exp((ISI - t_ref)/tau_m) gives
    V_inf = (a*V_th - V_reset)/(a - 1), and I_tonic = (V_inf - E_L)/tau_m.

    `dt_ms` corrects for the simulation being stepped on a grid. That formula
    is continuous-time, so it places the threshold crossing exactly on a step
    boundary -- where floating-point rounding decides whether the spike is
    detected on that step or the next one, and landing one step late costs a
    whole dt of ISI (5% of it at 50 Hz with dt = 1 ms). Shortening the target
    charge time by half a step puts the crossing in the middle of its step
    instead, which rounds to the intended step reliably. Left at 0 the function
    returns the exact continuous-time answer."""
    if target_hz <= 0.0:
        return 0.0
    isi_ms = 1000.0 / target_hz
    charge_ms = isi_ms - t_ref_ms - 0.5 * dt_ms
    if charge_ms <= 0.0:
        raise ValueError(f"target rate {target_hz} Hz needs an ISI shorter than the {t_ref_ms} ms refractory period")
    a = np.exp(charge_ms / tau_m_ms)
    v_inf = (a * v_th_mv - v_reset_mv) / (a - 1.0)
    return float((v_inf - e_leak_mv) / tau_m_ms)


class ExpSynapse:
    """Single-exponential conductance: g decays with tau, each presynaptic
    spike adds `increment`. One instance per postsynaptic population, holding
    that population's total conductance of this type."""

    def __init__(self, n_units, tau_ms, dt_ms, reversal_mv):
        self.g = np.zeros(n_units, dtype=float)
        self.tau_ms = tau_ms
        self.reversal_mv = reversal_mv
        self._decay = float(np.exp(-dt_ms / tau_ms))      # exact single-step decay factor

    def decay(self):
        self.g *= self._decay

    def add(self, amount):
        self.g += amount

    def reset(self):
        self.g[:] = 0.0


class LIFPopulation:
    """Conductance-based leaky integrate-and-fire population.

    Holds one excitatory and one inhibitory synapse pool; a subclass may add
    more (PKJPopulation adds the CF pause). `step` decays the conductances,
    advances V, and returns a boolean spike vector.
    """

    def __init__(self, n_units, dt_ms, tau_m_ms, v_th_mv, v_reset_mv, e_leak_mv,
                 t_ref_ms, baseline_hz, tau_exc_ms, tau_inh_ms, e_exc_mv, e_inh_mv,
                 noise_sigma_mv=0.0, v_init_mv=None, rng=None):
        self.n_units = n_units
        self.dt_ms = dt_ms
        self.tau_m_ms = tau_m_ms
        self.v_th_mv = v_th_mv
        self.v_reset_mv = v_reset_mv
        self.e_leak_mv = e_leak_mv
        self.t_ref_ms = t_ref_ms
        self.g_leak = 1.0 / tau_m_ms                              # C = 1, so g_L is just 1/tau_m

        self.i_tonic = tonic_drive_for_rate(baseline_hz, tau_m_ms, v_th_mv, v_reset_mv, e_leak_mv, t_ref_ms, dt_ms)
        self.baseline_hz = baseline_hz

        # One pool of each sign per population. On PKJ the excitatory pool carries PF input and the
        # inhibitory one is unused (the CF pause is a separate conductance, see PKJPopulation); on
        # DCN the inhibitory pool carries PKJ input and the excitatory one is unused, DCN's tonic
        # drive being i_tonic rather than a synapse. Both are allocated either way so the two
        # populations share one step().
        self.exc = ExpSynapse(n_units, tau_exc_ms, dt_ms, e_exc_mv)
        self.inh = ExpSynapse(n_units, tau_inh_ms, dt_ms, e_inh_mv)

        # Start each cell at a random point between reset and threshold, so the population does not
        # fire its first spike in lock-step (an artifact that would otherwise take several hundred ms to decorrelate).
        rng = rng if rng is not None else np.random.default_rng()
        if v_init_mv is None:
            self.V = rng.uniform(v_reset_mv, v_th_mv, size=n_units)
        else:
            self.V = np.full(n_units, v_init_mv, dtype=float)
        self._refractory_ms = np.zeros(n_units)
        self.spiked = np.zeros(n_units, dtype=bool)                # last step's spike vector

        # Membrane noise, as an Ornstein-Uhlenbeck process on V with stationary SD noise_sigma_mv:
        # dV += sigma*sqrt(2*dt/tau_m)*xi. It stands in for the synaptic bombardment these cells
        # receive from everything this MVP does not model (mossy fibers onto DCN, molecular-layer
        # interneurons onto PKJ). It is not decoration: a noiseless LIF has a hard rheobase, so its
        # rate collapses from tens of Hz to silence over a very narrow band of inhibitory conductance.
        # The loop's negative feedback needs DCN's rate to vary *smoothly* with PKJ drive, and noise
        # is what linearizes the f-I curve around threshold to give it that. Same role the OU current
        # plays in sim/io_channels.py, for the same reason.
        self.noise_sigma_mv = noise_sigma_mv
        self._noise_kick = noise_sigma_mv * np.sqrt(2.0 * dt_ms / tau_m_ms)
        self._rng = rng

    def extra_conductances(self):
        """(g, g*E) contributions beyond exc/inh. Overridden by PKJPopulation
        for the CF pause; the base population has none."""
        return 0.0, 0.0

    def step(self, dt_ms=None):
        dt = self.dt_ms if dt_ms is None else dt_ms

        g_e, g_i = self.exc.g, self.inh.g
        g_x, gx_ex = self.extra_conductances()

        g_tot = self.g_leak + g_e + g_i + g_x
        i_in = (self.g_leak * self.e_leak_mv + g_e * self.exc.reversal_mv
                + g_i * self.inh.reversal_mv + gx_ex + self.i_tonic)
        v_inf = i_in / g_tot
        v_new = v_inf + (self.V - v_inf) * np.exp(-dt * g_tot)   # C = 1, so tau_eff = 1/g_tot and dt/tau_eff = dt*g_tot
        if self._noise_kick:
            v_new = v_new + self._noise_kick * self._rng.standard_normal(self.n_units)

        refractory = self._refractory_ms > 0.0
        v_new = np.where(refractory, self.v_reset_mv, v_new)              # clamped at reset while refractory
        np.maximum(self._refractory_ms - dt, 0.0, out=self._refractory_ms)

        spiked = v_new >= self.v_th_mv
        v_new = np.where(spiked, self.v_reset_mv, v_new)
        self._refractory_ms[spiked] = self.t_ref_ms

        self.V = v_new
        self.spiked = spiked

        self.exc.decay()                                                   # conductances decay once per step, after use
        self.inh.decay()
        self._decay_extra()
        return spiked

    def _decay_extra(self):
        pass

    @property
    def rate_hz(self):
        """Instantaneous population rate implied by this step's spikes. Only a
        readout -- nothing in the dynamics consumes it."""
        return float(self.spiked.mean()) * 1000.0 / self.dt_ms


class PKJPopulation(LIFPopulation):
    """Purkinje cells: LIF driven by weighted PF excitation, plus the brief
    inhibitory pause that follows each climbing-fiber complex spike.

    The pause is modeled as a large inhibitory conductance switched on for
    cf_pause_ms rather than as a subtracted current, so that (like the real
    post-complex-spike pause) it both hyperpolarizes the cell and shunts
    whatever PF excitation arrives during it."""

    def __init__(self, *args, pause_g=1.0, pause_ms=10.0, e_pause_mv=-75.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pause_g = pause_g
        self.pause_ms = pause_ms
        self.e_pause_mv = e_pause_mv
        # Per cell, because one population holds every Purkinje cell in the microzone and each is
        # paused only by the single climbing fiber that contacts it.
        self._pause_remaining_ms = np.zeros(self.n_units)

    def trigger_cf_pause(self, mask=None):
        """(Re)start the pause countdown on the cells selected by `mask`
        (a bool array over the population); None pauses all of them."""
        if mask is None:
            self._pause_remaining_ms[:] = self.pause_ms
        else:
            self._pause_remaining_ms[np.asarray(mask, dtype=bool)] = self.pause_ms

    def extra_conductances(self):
        paused = self._pause_remaining_ms > 0.0
        g = np.where(paused, self.pause_g, 0.0)
        return g, g * self.e_pause_mv

    def _decay_extra(self):
        np.maximum(self._pause_remaining_ms - self.dt_ms, 0.0, out=self._pause_remaining_ms)


class DCNPopulation(LIFPopulation):
    """Deep cerebellar nuclear cells: tonically driven LIF, inhibited by the
    `n_pkj_per_dcn` Purkinje cells that converge on it. No extra machinery
    beyond LIFPopulation -- the class exists so the network's three cell types
    read as three named types at the call site."""
