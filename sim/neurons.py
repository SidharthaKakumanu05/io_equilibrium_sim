"""Purkinje (PKJ) and deep nuclear (DCN) cells: leaky integrate-and-fire.

Both cell types use the same simple model. (The olive cell is far more detailed;
it lives in sim/io_channels.py.)

  * V decays toward a resting value, and is pulled toward other values by
    synaptic conductances and a constant input current I_tonic:

        dV/dt = -g_L (V - E_L) - g_exc (V - E_exc) - g_inh (V - E_inh) + I_tonic

  * When V reaches threshold, the cell spikes, V is reset, and it is held there
    for the refractory period.

Contents, in reading order:

  tonic_drive_for_rate  works out the I_tonic that makes a cell fire at a given
                        rate, so baselines can be set in Hz
  ExpSynapse            a synaptic conductance: each input spike adds to it,
                        then it decays exponentially
  LIFPopulation         a whole population of these cells, stepped together
  PKJPopulation         adds the climbing-fibre response: a complex spike
                        (a short forced burst), then a pause
  DCNPopulation         adds nothing; it exists so the code can say "DCN"

The voltage update uses "exponential Euler": with the conductances held fixed
over one step, V moves toward its target V_inf along an exact exponential,

    V <- V_inf + (V - V_inf) * exp(-dt * g_tot)

This stays stable even when a conductance is very large (as during the CF
pause), where a plain Euler step would overshoot.
"""
import numpy as np


def tonic_drive_for_rate(target_hz, tau_m_ms, v_th_mv, v_reset_mv, e_leak_mv, t_ref_ms, dt_ms=0.0):
    """Return the constant input current (mV/ms) that makes a cell with no
    synaptic input fire at target_hz.

    With constant input, V rises from V_reset toward V_inf = E_L + I_tonic*tau_m,
    and the time between spikes is
        t_ref + tau_m * ln((V_inf - V_reset) / (V_inf - V_th)).
    Solving that for V_inf, then for I_tonic, gives the code below.

    dt_ms aims the threshold crossing at the middle of a time step instead of
    the edge of one, so rounding cannot push each spike a step late."""
    if target_hz <= 0.0:
        return 0.0
    isi_ms = 1000.0 / target_hz
    charge_ms = isi_ms - t_ref_ms - 0.5 * dt_ms               # time spent rising from reset to threshold
    if charge_ms <= 0.0:
        raise ValueError(f"target rate {target_hz} Hz needs an ISI shorter than the {t_ref_ms} ms refractory period")
    a = np.exp(charge_ms / tau_m_ms)
    v_inf = (a * v_th_mv - v_reset_mv) / (a - 1.0)
    return float((v_inf - e_leak_mv) / tau_m_ms)


class ExpSynapse:
    """A synaptic conductance for a whole population: one value per cell.

    add() is called with each step's input, and decay() shrinks every value by
    exp(-dt/tau) once per step."""

    def __init__(self, n_units, tau_ms, dt_ms, reversal_mv):
        self.g = np.zeros(n_units, dtype=float)
        self.tau_ms = tau_ms
        self.reversal_mv = reversal_mv
        self._decay = float(np.exp(-dt_ms / tau_ms))      # fraction left after one step

    def decay(self):
        self.g *= self._decay

    def add(self, amount):
        self.g += amount

    def reset(self):
        self.g[:] = 0.0


class LIFPopulation:
    """A population of n_units leaky integrate-and-fire cells.

    Each cell has one excitatory conductance (self.exc) and one inhibitory one
    (self.inh). Other code adds input to them, then calls step(), which updates
    every cell's V and returns a True/False array of which cells spiked.
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
        self.g_leak = 1.0 / tau_m_ms                              # leak conductance (capacitance is taken as 1)

        # Constant input current that gives the requested spontaneous rate.
        self.i_tonic = tonic_drive_for_rate(baseline_hz, tau_m_ms, v_th_mv, v_reset_mv, e_leak_mv, t_ref_ms, dt_ms)
        self.baseline_hz = baseline_hz

        # Synaptic conductances. PKJ uses exc (from PFs); DCN uses inh (from PKJ).
        # The other one stays at zero.
        self.exc = ExpSynapse(n_units, tau_exc_ms, dt_ms, e_exc_mv)
        self.inh = ExpSynapse(n_units, tau_inh_ms, dt_ms, e_inh_mv)

        # Start each cell at a random voltage between reset and threshold, so they don't all fire
        # their first spike at the same moment.
        rng = rng if rng is not None else np.random.default_rng()
        if v_init_mv is None:
            self.V = rng.uniform(v_reset_mv, v_th_mv, size=n_units)
        else:
            self.V = np.full(n_units, v_init_mv, dtype=float)
        self._refractory_ms = np.zeros(n_units)                    # time left in each cell's refractory period
        self.spiked = np.zeros(n_units, dtype=bool)                # which cells spiked on the last step

        # Random voltage noise, added every step. It stands in for all the inputs this model leaves
        # out. It also matters to the loop: without noise a cell's rate drops from tens of Hz to
        # zero over a tiny range of inhibition. With noise the rate changes gradually, and the
        # feedback needs that. _noise_kick is the per-step size that gives a spread of noise_sigma_mv.
        self.noise_sigma_mv = noise_sigma_mv
        self._noise_kick = noise_sigma_mv * np.sqrt(2.0 * dt_ms / tau_m_ms)
        self._rng = rng

    def extra_conductances(self):
        """Any extra conductance g and its g*E term, beyond exc and inh. None
        here; PKJPopulation overrides this to add the CF pause."""
        return 0.0, 0.0

    def forced_spikes(self):
        """Cells that must spike this step regardless of V, or None. None here;
        PKJPopulation overrides this for the complex spike."""
        return None

    def step(self, dt_ms=None):
        """Advance every cell by one step. Returns a True/False array: who spiked."""
        dt = self.dt_ms if dt_ms is None else dt_ms

        # --- 1. move V toward the voltage the conductances are pulling it to ---
        g_e, g_i = self.exc.g, self.inh.g
        g_x, gx_ex = self.extra_conductances()

        g_tot = self.g_leak + g_e + g_i + g_x                              # total conductance
        i_in = (self.g_leak * self.e_leak_mv + g_e * self.exc.reversal_mv
                + g_i * self.inh.reversal_mv + gx_ex + self.i_tonic)
        v_inf = i_in / g_tot                                               # where V is heading
        v_new = v_inf + (self.V - v_inf) * np.exp(-dt * g_tot)           # exponential Euler step
        if self._noise_kick:
            v_new = v_new + self._noise_kick * self._rng.standard_normal(self.n_units)

        # --- 2. refractory cells are held at reset ---
        refractory = self._refractory_ms > 0.0
        v_new = np.where(refractory, self.v_reset_mv, v_new)
        np.maximum(self._refractory_ms - dt, 0.0, out=self._refractory_ms)

        # --- 3. cells at threshold (or forced to fire) spike, reset, and become refractory ---
        spiked = v_new >= self.v_th_mv
        forced = self.forced_spikes()
        if forced is not None:
            spiked = spiked | forced
        v_new = np.where(spiked, self.v_reset_mv, v_new)
        self._refractory_ms[spiked] = self.t_ref_ms

        self.V = v_new
        self.spiked = spiked

        # --- 4. synaptic conductances fade before the next step ---
        self.exc.decay()
        self.inh.decay()
        self._decay_extra()
        return spiked

    def _decay_extra(self):
        pass

    @property
    def rate_hz(self):
        """Fraction of cells that spiked this step, as a rate in Hz. Not used by
        the simulation."""
        return float(self.spiked.mean()) * 1000.0 / self.dt_ms


class PKJPopulation(LIFPopulation):
    """Purkinje cells: the basic LIF cell, plus the response to a climbing-fibre event.

    The climbing fibre is excitatory. Each CF event gives its Purkinje cells:

      1. a complex spike: burst_spikes forced spikes, burst_isi_ms apart,
         starting on the next step. These are ordinary output spikes, so they
         inhibit the DCN like any other PKJ spike.
      2. a pause: a large conductance at e_pause_mv that holds V down and
         swamps PF input. It is on during the burst and for pause_ms after the
         last burst spike.

    For a CF event on step t with the defaults (3 spikes, 2 ms apart, 10 ms
    pause): forced spikes on t+1, t+3, t+5, then silent through t+15.

    burst_spikes = 0 gives the older model: no complex spike, just a pause on
    steps t+1 .. t+10.
    """

    def __init__(self, *args, pause_g=1.0, pause_ms=10.0, e_pause_mv=-75.0,
                 burst_spikes=3, burst_isi_ms=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pause_g = pause_g
        self.pause_ms = pause_ms
        self.e_pause_mv = e_pause_mv
        self.burst_spikes = int(burst_spikes)
        self.burst_isi_ms = burst_isi_ms
        self._pause_remaining_ms = np.zeros(self.n_units)                  # time left in each cell's pause
        self._burst_left = np.zeros(self.n_units, dtype=int)               # complex-spike spikes still to fire
        self._burst_wait_ms = np.zeros(self.n_units)                       # time until the next one

    def trigger_complex_spike(self, mask=None):
        """Start (or restart) a complex spike and pause on the cells where
        `mask` is True. mask=None triggers every cell."""
        cells = slice(None) if mask is None else np.asarray(mask, dtype=bool)
        if self.burst_spikes == 0:
            self._pause_remaining_ms[cells] = self.pause_ms
            return
        self._burst_left[cells] = self.burst_spikes
        self._burst_wait_ms[cells] = 0.0                                   # first spike on the next step
        # Pause from the next step until pause_ms after the last burst spike.
        self._pause_remaining_ms[cells] = ((self.burst_spikes - 1) * self.burst_isi_ms
                                           + self.dt_ms + self.pause_ms)

    def forced_spikes(self):
        """Cells due to fire their next complex-spike spike this step."""
        due = (self._burst_left > 0) & (self._burst_wait_ms <= 0.0)
        self._burst_left[due] -= 1
        self._burst_wait_ms[due] = self.burst_isi_ms
        return due

    def extra_conductances(self):
        """The pause conductance: pause_g on paused cells, 0 elsewhere."""
        paused = self._pause_remaining_ms > 0.0
        g = np.where(paused, self.pause_g, 0.0)
        return g, g * self.e_pause_mv

    def _decay_extra(self):
        """Count the pause and the wait to the next burst spike down by one step."""
        np.maximum(self._pause_remaining_ms - self.dt_ms, 0.0, out=self._pause_remaining_ms)
        np.maximum(self._burst_wait_ms - self.dt_ms, 0.0, out=self._burst_wait_ms)


class DCNPopulation(LIFPopulation):
    """Deep nuclear cells: the basic LIF cell with nothing added. They fire on
    their own (dcn_baseline_hz) and are inhibited by Purkinje cells."""
