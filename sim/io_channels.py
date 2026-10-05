"""The inferior-olive (IO) cell.

Unlike the Purkinje and DCN cells, the olive cell is modelled with real ion
channels. It oscillates and fires about 3 times a second on its own, and DCN
inhibition slows it down. Each "spike" is a calcium spike, and each calcium
spike is one climbing-fibre (CF) event.

The currents, and what each does in one cycle:

  I_CaL   low-threshold Ca2+. Switched off (inactivated) at rest and switched
          back on by hyperpolarisation. Once available, it opens and produces
          the calcium spike.
  I_CaH   high-threshold Ca2+. Opens only during the spike. Brings Ca2+ into
          the cell.
  I_KCa   Ca2+-activated K+. Opened by that Ca2+. Ends the spike and holds the
          cell hyperpolarised afterwards, which switches I_CaL back on for the
          next cycle.
  I_h     slow "sag" current, activated by hyperpolarisation. Pulls the cell
          back up from the dip and sets how long a cycle takes.
  I_leak  passive leak.
  I_GABA  inhibition from the DCN, as a conductance.
  I_gap   current through gap junctions from neighbouring olive cells
          (sim/io_coupling.py).

The channel equations come from published olive models (Schweighofer et al.
1999; De Gruijl et al. 2012). Channel densities are in config.IOChannelParams.

File layout:
  1. gating functions: for each channel, how its gates depend on voltage (or Ca2+)
  2. IOPopulation: all olive cells, integrated together
  3. IOConductanceNeuron: a one-cell wrapper (run.py does not use it)

All olive cells are stepped together because gap junctions connect them:
each cell's update needs every other cell's voltage at the same moment.
"""
import math
import numpy as np

_EXP_CLAMP = 60.0   # limit on the argument to exp(), so extreme voltages can't overflow to inf


def _exp(x):
    """exp() with its argument limited to +-60, so it never returns inf."""
    return np.exp(np.clip(x, -_EXP_CLAMP, _EXP_CLAMP))


def _unwrap(a):
    """Return a plain number for a single-value input and an array otherwise,
    so the functions below work on one voltage or on all cells at once."""
    a = np.asarray(a)
    return a[()] if a.ndim == 0 else a


# ===========================================================================
#  1. Gating functions
#
#  Each channel has one or more "gates", numbers between 0 (closed) and 1 (open).
#  Each gate moves toward a target value x_inf at a rate set by its time constant
#  tau. Both depend on voltage (or, for I_KCa, on calcium). The functions below
#  return x_inf and tau for each gate.
#
#  The `_e` argument picks which exp() to use. Called on their own they use the
#  safe _exp. IOPopulation.substep limits V itself first, so it passes the faster
#  np.exp. The result is the same.
# ===========================================================================

# --- I_CaL: low-threshold Ca2+.  Conductance = g_cal * k^3 * l -------------
# k = activation (opens fast as V rises), l = inactivation (slow; reopens when V is low).

def cal_k_inf(v_mv, _e=_exp):
    """Target value of k. Half open at -61 mV, just above rest."""
    return _unwrap(1.0 / (1.0 + _e(-(np.asarray(v_mv, dtype=float) + 61.0) / 4.2)))


def cal_l_inf(v_mv, _e=_exp):
    """Target value of l. Near 0 at rest (-60 mV), so I_CaL is mostly off,
    and only becomes available after the cell has been pushed below about
    -85 mV."""
    return _unwrap(1.0 / (1.0 + _e((np.asarray(v_mv, dtype=float) + 85.5) / 8.5)))


def cal_tau_l(v_mv, _e=_exp):
    """Time constant of l, ms: about 35 ms when depolarised, up to ~165 ms
    at -85 mV."""
    v = np.asarray(v_mv, dtype=float)
    return _unwrap(20.0 * _e((v + 160.0) / 30.0) / (1.0 + _e((v + 84.0) / 7.3)) + 35.0)


# --- I_CaH: high-threshold Ca2+.  Conductance = g_cah * r^2 ----------------

def cah_r_rates(v_mv, _e=_exp):
    """Opening rate alpha and closing rate beta of gate r. Its target is
    r_inf = alpha/(alpha+beta) and its time constant is 1/(alpha+beta).

    The beta formula is 0/0 at exactly V = -8.5 mV. Its limit there is 0.1,
    which is used directly at that point."""
    v = np.asarray(v_mv, dtype=float)
    alpha = 1.7 / (1.0 + _e(-(v - 5.0) / 13.9))
    x = v + 8.5
    singular = np.abs(x) < 1e-6
    denom = np.where(singular, 1.0, _e(x / 5.0) - 1.0)
    beta = np.where(singular, 0.1, 0.02 * x / denom)
    return alpha, beta


# Convenience wrappers around cah_r_rates.
def cah_r_alpha(v_mv, _e=_exp):
    return _unwrap(cah_r_rates(v_mv, _e)[0])


def cah_r_beta(v_mv, _e=_exp):
    return _unwrap(cah_r_rates(v_mv, _e)[1])


def cah_r_inf(v_mv, _e=_exp):
    a, b = cah_r_rates(v_mv, _e)
    return _unwrap(a / (a + b))


def cah_tau_r(v_mv, _e=_exp):
    a, b = cah_r_rates(v_mv, _e)
    return _unwrap(1.0 / (a + b))


# --- I_KCa: Ca2+-activated K+.  Conductance = g_kca * s ---------------------
# Gate s depends on calcium, not voltage. The defaults below match config.py;
# IOPopulation always passes the configured values.
_KCA_CA_SCALE, _KCA_ALPHA_MAX, _KCA_BETA = 0.02, 0.01, 0.015


def kca_s_alpha(ca_um, ca_scale=_KCA_CA_SCALE, alpha_max=_KCA_ALPHA_MAX):
    """Opening rate of s: rises with [Ca]i up to a maximum of alpha_max."""
    return _unwrap(np.minimum(ca_scale * np.asarray(ca_um, dtype=float), alpha_max))


def kca_s_inf(ca_um, ca_scale=_KCA_CA_SCALE, alpha_max=_KCA_ALPHA_MAX, beta=_KCA_BETA):
    a = kca_s_alpha(ca_um, ca_scale, alpha_max)
    return _unwrap(a / (a + beta))


def kca_tau_s(ca_um, ca_scale=_KCA_CA_SCALE, alpha_max=_KCA_ALPHA_MAX, beta=_KCA_BETA):
    a = kca_s_alpha(ca_um, ca_scale, alpha_max)
    return _unwrap(1.0 / (a + beta))


# --- I_h: the slow "sag" current.  Conductance = g_h * q --------------------

def h_q_inf(v_mv, _e=_exp):
    """Target value of q. Opens when V is LOW (q -> 1 as V falls)."""
    return _unwrap(1.0 / (1.0 + _e((np.asarray(v_mv, dtype=float) + 80.0) / 4.0)))


def h_tau_q(v_mv, _e=_exp):
    """Time constant of q, ms: hundreds of ms, so I_h acts on the slow rhythm,
    not on the spike itself."""
    v = np.asarray(v_mv, dtype=float)
    return _unwrap(1.0 / (_e(-0.086 * v - 14.6) + _e(0.070 * v - 1.87)))


# ===========================================================================
#  2. The population of olive cells
# ===========================================================================

# Inside substep, the gating functions are evaluated at V limited to this range so exp() can't
# overflow. Normal voltages are far inside it.
_V_GATING_MIN, _V_GATING_MAX = -200.0, 100.0

class IOPopulation:
    """All n_cells olive cells, stepped together.

    State per cell: V (voltage), Ca (calcium), and the gates k, l, r, s, q.

    How a step works: the main simulation calls step() once per 1 ms. step()
    splits that into 10 sub-steps of 0.1 ms (substep()), because the calcium
    spike is too fast for 1 ms steps. Each sub-step updates V, Ca and every gate
    with exponential Euler: x <- x_inf + (x - x_inf) * exp(-dt/tau).

    With the default settings the cell is fully deterministic: no injected
    current and no noise, yet it fires about 3 Hz on its own channels.

    g_gap: matrix of gap-junction conductances between cells (from
    sim/io_coupling.py), or None for no coupling.
    """

    def __init__(self, params, n_cells, dt_ms, rng=None, g_gap=None,
                 heterogeneity_cv=0.0, heterogeneity_seed=0):
        self.n = int(n_cells)
        self._p = params
        # Sub-step size: 0.1 ms by default, adjusted so a whole number of sub-steps fits in dt_ms.
        self.sub_dt_ms = min(params.sub_dt_ms, dt_ms)
        self.n_sub = max(1, int(round(dt_ms / self.sub_dt_ms)))      # sub-steps per main step (10)
        self.sub_dt_ms = dt_ms / self.n_sub
        self.rng = rng if rng is not None else np.random.default_rng()

        # --- starting state: every gate at its resting target value ---
        p = params
        self.V = np.full(self.n, p.v_init_mv, dtype=float)           # membrane potential, mV
        self.Ca = np.full(self.n, p.ca_init_um, dtype=float)         # intracellular [Ca2+], uM
        self.k = np.broadcast_to(cal_k_inf(self.V), (self.n,)).astype(float)   # I_CaL activation
        self.l = np.broadcast_to(cal_l_inf(self.V), (self.n,)).astype(float)   # I_CaL inactivation
        self.r = np.broadcast_to(cah_r_inf(self.V), (self.n,)).astype(float)   # I_CaH activation
        self.s = np.broadcast_to(                                              # I_KCa activation
            kca_s_inf(self.Ca, p.kca_ca_scale, p.kca_alpha_max, p.kca_beta), (self.n,)).astype(float)
        self.q = np.broadcast_to(h_q_inf(self.V), (self.n,)).astype(float)     # I_h activation
        self.noise = np.zeros(self.n)                                # current value of the noise current (0 when off)

        # Channel densities. With heterogeneity_cv = 0 these are single numbers shared by all
        # cells; above 0 each cell gets its own fixed value (see _draw_conductances).
        self._g_cal, self._g_cah, self._g_kca, self._g_h = self._draw_conductances(
            heterogeneity_cv, heterogeneity_seed)
        self.heterogeneity_cv = float(heterogeneity_cv)

        self._refractory_ms = np.zeros(self.n)                       # so one calcium spike counts as one CF event
        self._refresh_derived()                                      # precompute constant step factors

        self.display_rate_hz = np.zeros(self.n)                      # smoothed rate readout; not used by the model
        self._g_gaba = np.zeros(self.n)                              # last DCN inhibition received, for currents()
        self.set_gap_junctions(g_gap)

    # --- setup helpers ----------------------------------------------------

    def _draw_conductances(self, cv, seed):
        """Give each cell its own g_cal, g_kca and g_h, randomly spread by `cv`
        around the configured value (cv = 0.15 means a 15% spread). A lognormal
        is used so every value is positive and the average stays the same.
        g_cah and g_leak are left equal across cells.

        Returns (g_cal, g_cah, g_kca, g_h), each either one number or one per cell."""
        p = self._p
        base = (p.g_cal, p.g_cah, p.g_kca, p.g_h)
        if not cv or cv <= 0.0:
            return base
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        mu = -0.5 * sigma * sigma                      # chosen so the average multiplier is exactly 1
        rng = np.random.default_rng(seed)
        draw = lambda b: b * rng.lognormal(mu, sigma, size=self.n)
        return draw(p.g_cal), p.g_cah, draw(p.g_kca), draw(p.g_h)

    def set_gap_junctions(self, g_gap):
        """Set the gap-junction matrix (None = no coupling), after checking it
        is symmetric with zeros on the diagonal."""
        if g_gap is None:
            self.g_gap = None
            self._gap_row_sum = 0.0
            return
        g_gap = np.asarray(g_gap, dtype=float)
        if g_gap.shape != (self.n, self.n):
            raise ValueError(f"g_gap must be ({self.n}, {self.n}), got {g_gap.shape}")
        if not np.allclose(g_gap, g_gap.T):
            raise ValueError("g_gap must be symmetric: a gap junction conducts equally both ways")
        if np.any(np.diag(g_gap) != 0.0):
            raise ValueError("g_gap must have a zero diagonal: a cell is not coupled to itself")
        self.g_gap = g_gap
        self._gap_row_sum = g_gap.sum(axis=1)                        # each cell's total gap conductance

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, params):
        self.set_params(params)

    def set_params(self, params):
        """Replace the parameters mid-run, and recompute the factors that
        depend on them. Not used by run.py."""
        self._p = params
        self._refresh_derived()

    def _refresh_derived(self):
        """Precompute the per-sub-step factors that depend only on parameters."""
        p, dt = self._p, self.sub_dt_ms
        self._noise_decay = float(np.exp(-dt / p.noise_tau_ms))                      # noise: how much carries over
        self._noise_kick = float(p.noise_sigma * np.sqrt(1.0 - self._noise_decay ** 2))  # noise: size of each new kick
        self._k_step = float(1.0 - np.exp(-dt / p.cal_tau_k_ms))                     # gate k has a fixed tau
        self._ca_decay_factor = float(np.exp(-dt * p.ca_decay))                      # calcium pumped out per sub-step

    # --- stepping ---------------------------------------------------------

    def step(self, dt_ms, g_gaba=0.0):
        """Advance every cell by one main step (dt_ms), given the DCN inhibition
        each one is receiving. Returns True/False per cell: did it fire a CF event?"""
        g_gaba = np.broadcast_to(np.asarray(g_gaba, dtype=float), (self.n,))
        self._g_gaba = g_gaba                                        # remembered for currents()
        noise_block = self.rng.standard_normal((self.n_sub, self.n))  # random numbers for all sub-steps at once
        fired = np.zeros(self.n, dtype=bool)
        for i in range(self.n_sub):
            fired |= self.substep(g_gaba, noise_block[i])            # fired in any sub-step -> fired this step
        self._update_display_rate(dt_ms, fired)
        return fired

    def substep(self, g_gaba, xi):
        """Advance every cell by one sub-step (0.1 ms). Returns which cells
        crossed the spike threshold. `xi` is this sub-step's random numbers."""
        p = self._p
        dt = self.sub_dt_ms
        v = self.V
        vg = np.minimum(np.maximum(v, _V_GATING_MIN), _V_GATING_MAX)   # V limited to a safe range, for the gates only
        exp = np.exp

        self.noise = self.noise * self._noise_decay + self._noise_kick * xi   # update the noise current (zero if off)

        # --- 1. each channel's conductance right now: density x gates ---
        g_cal = self._g_cal * self.k * self.k * self.k * self.l       # k^3 * l
        g_cah = self._g_cah * self.r * self.r                         # r^2
        g_kca = self._g_kca * self.s
        g_h = self._g_h * self.q

        # --- 2. update V ---
        # Every current has the form g * (V - E). V moves toward the weighted average of the
        # reversal potentials, v_inf = sum(g*E) / sum(g), at a speed set by sum(g).
        # Gap junctions add sum_j g_ij * (V_j - V_i): the V_i part goes into g_tot and the V_j
        # part into i_in.
        g_tot = g_cal + g_cah + g_kca + g_h + p.g_leak + g_gaba + self._gap_row_sum
        i_in = (g_cal * p.e_ca + g_cah * p.e_ca + g_kca * p.e_k + g_h * p.e_h
                + p.g_leak * p.e_leak + g_gaba * p.e_gaba
                + p.i_app + self.noise)                                # sum(g*E) + injected current + noise
        if self.g_gap is not None:
            i_in = i_in + self.g_gap @ v                               # pull from neighbours' voltages
        v_inf = i_in / g_tot                                          # where V is heading
        v_new = v_inf + (v - v_inf) * exp(-dt * g_tot / p.c_m)

        # --- 3. update calcium: I_CaH brings it in, a pump removes it ---
        i_cah = g_cah * (v - p.e_ca)                                   # negative = Ca2+ flowing in
        ca_inf = -p.ca_influx * i_cah / p.ca_decay                     # level calcium is heading toward
        self.Ca = ca_inf + (self.Ca - ca_inf) * self._ca_decay_factor
        np.maximum(self.Ca, 0.0, out=self.Ca)                          # can't go below zero

        # --- 4. move every gate toward its target (using V from the start of this sub-step) ---
        self.k += (cal_k_inf(vg, exp) - self.k) * self._k_step
        self.l += (cal_l_inf(vg, exp) - self.l) * (1.0 - exp(-dt / cal_tau_l(vg, exp)))
        r_alpha, r_beta = cah_r_rates(vg, exp)
        r_sum = r_alpha + r_beta
        self.r += (r_alpha / r_sum - self.r) * (1.0 - exp(-dt * r_sum))   # target a/(a+b), time constant 1/(a+b)
        self.q += (h_q_inf(vg, exp) - self.q) * (1.0 - exp(-dt / h_tau_q(vg, exp)))
        s_alpha = np.minimum(p.kca_ca_scale * self.Ca, p.kca_alpha_max)   # I_KCa gate follows calcium
        s_sum = s_alpha + p.kca_beta
        self.s += (s_alpha / s_sum - self.s) * (1.0 - exp(-dt * s_sum))

        # --- 5. CF event = V rising through v_spike_mv, unless one was just counted ---
        np.maximum(self._refractory_ms - dt, 0.0, out=self._refractory_ms)
        fired = (v < p.v_spike_mv) & (v_new >= p.v_spike_mv) & (self._refractory_ms <= 0.0)
        self._refractory_ms[fired] = p.spike_refractory_ms

        self.V = v_new
        return fired

    def currents(self):
        """Each channel's current right now (uA/cm^2), as a dict of arrays.
        For inspection only; not used by the simulation."""
        p = self._p
        out = {
            "I_CaL": self._g_cal * self.k ** 3 * self.l * (self.V - p.e_ca),
            "I_CaH": self._g_cah * self.r ** 2 * (self.V - p.e_ca),
            "I_KCa": self._g_kca * self.s * (self.V - p.e_k),
            "I_h": self._g_h * self.q * (self.V - p.e_h),
            "I_leak": p.g_leak * (self.V - p.e_leak),
            "I_GABA": self._g_gaba * (self.V - p.e_gaba),
        }
        if self.g_gap is not None:
            out["I_gap"] = self._gap_row_sum * self.V - self.g_gap @ self.V   # sum over partners of g * (V_self - V_partner)
        return out

    def _update_display_rate(self, dt_ms, fired):
        """Update the smoothed rate readout. Not used by the model."""
        instantaneous_hz = np.where(fired, 1000.0 / dt_ms, 0.0)
        self.display_rate_hz += (dt_ms / self._p.display_tau_ms) * (instantaneous_hz - self.display_rate_hz)


# ===========================================================================
#  3. One-cell wrapper (not used by run.py)
# ===========================================================================

class IOConductanceNeuron:
    """A single olive cell, for experimenting with one cell on its own.
    Internally it is an IOPopulation of size 1, so the equations are the same."""

    def __init__(self, params, dt_ms, rng=None):
        self.pop = IOPopulation(params, 1, dt_ms, rng=rng)

    # Each property returns the single cell's value as a plain number.
    V = property(lambda self: self.pop.V[0])
    Ca = property(lambda self: self.pop.Ca[0])
    k = property(lambda self: self.pop.k[0])
    l = property(lambda self: self.pop.l[0])
    r = property(lambda self: self.pop.r[0])
    s = property(lambda self: self.pop.s[0])
    q = property(lambda self: self.pop.q[0])
    noise = property(lambda self: self.pop.noise[0])
    display_rate_hz = property(lambda self: float(self.pop.display_rate_hz[0]))
    rng = property(lambda self: self.pop.rng)
    n_sub = property(lambda self: self.pop.n_sub)
    sub_dt_ms = property(lambda self: self.pop.sub_dt_ms)

    @property
    def p(self):
        return self.pop.p

    @p.setter
    def p(self, params):
        self.pop.set_params(params)

    def step(self, dt_ms, g_gaba=0.0):
        """Advance one main step under a given inhibition (mS/cm^2). Returns
        True if the cell fired a CF event."""
        return bool(self.pop.step(dt_ms, g_gaba)[0])

    def currents(self):
        return {name: float(value[0]) for name, value in self.pop.currents().items()}
