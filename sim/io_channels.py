"""Conductance-based inferior-olive neuron.

The olive is the one cell type here that is not an integrate-and-fire
abstraction: climbing-fiber output has to be *generated* by the cell's own
currents, because H1 is a claim about a self-paced oscillator being held at a
rate by inhibition, and an integrator with an imposed drive would assume the
answer. The currents:

  I_CaL   low-threshold ("T-type") Ca2+ -- k^3*l gating. Inactivated at rest,
          de-inactivates when hyperpolarized, then produces the regenerative
          low-threshold Ca2+ spike that carries the CF burst. This is the
          current that makes IO a rebound oscillator rather than a leaky
          integrator.
  I_CaH   high-threshold (dendritic, L-type-like) Ca2+ -- r^2 gating. Only
          opens on the Ca2+ spike, and is the Ca2+ source that loads [Ca]i.
  I_KCa   Ca2+-activated K+ -- gated by [Ca]i, not by voltage. Repolarizes
          the Ca2+ spike and produces the long AHP that both paces the
          subthreshold oscillation and de-inactivates I_CaL for the next
          cycle. I_CaL and I_KCa together are the oscillator.
  I_h     hyperpolarization-activated cation ("sag") current -- q gating.
          Depolarizes the trough of the AHP and sets oscillation frequency.
  I_L     leak.
  I_GABA  DCN inhibition, as a real Cl- conductance (shunting + hyperpolarizing)
          rather than a subtracted scalar.
  I_gap   IO-IO electrical (gap-junction) coupling -- see sim/io_coupling.py.

Gating kinetics (the voltage/[Ca] dependence -- i.e. the physiology) follow the
standard olivary model lineage: Schweighofer, Doya & Kawato (1999) J
Neurophysiol 82:804 and De Gruijl et al. (2012) PLoS Comput Biol 8:e1002814.
Maximal conductances and the leak reversal are re-fit here (see
config.IOChannelParams, which documents each departure) because those papers
are 3-compartment models and this is a single-compartment reduction --
kinetics transfer between the two, conductance densities do not.

Not modeled: the fast Na+/delayed-rectifier K+ spikes that ride on top of the
Ca2+ spike. The downstream circuit consumes only CF *event times*, and
resolving Na+ spikes would force a ~40x smaller timestep for no change to
anything this simulation measures. The CF event is therefore detected as the
low-threshold Ca2+ spike itself, which is the event that actually triggers the
somatic burst in the real cell.

The cells are integrated as a *population* (IOPopulation) rather than one at a
time, because gap junctions couple them within a sub-timestep: cell i's V
update needs every other cell's V at the same instant, so there is no way to
advance one cell to completion and then the next. IOConductanceNeuron is a
thin single-cell adapter over the same code, kept so the characterization
protocols and unit tests can drive one cell in isolation.
"""
import math
import numpy as np
from .native import advance_io

_EXP_CLAMP = 60.0   # |x| bound for exp(), avoids overflow at extreme V during transients


def _exp(x):
    """exp() with the argument clamped, so a transient excursion can't produce
    an inf mid-run. exp(+-60) is already 1e+-26, far past saturation of every
    sigmoid below. Works on scalars and arrays alike."""
    return np.exp(np.clip(x, -_EXP_CLAMP, _EXP_CLAMP))


def _unwrap(a):
    """Return a 0-d result as a numpy scalar, leaving real arrays alone. Lets
    every gating function below be called either with one voltage (tests,
    analysis) or with the population's whole V array."""
    a = np.asarray(a)
    return a[()] if a.ndim == 0 else a


# --- I_CaL: low-threshold (T-type) Ca2+ ------------------------------------
# k = activation (fast, 3rd power), l = inactivation (slow).
#
# Every function below takes the exponential to use as `_e`. The default is the
# clamped `_exp`, which is what callers passing an arbitrary voltage want.
# IOPopulation.substep clamps V once up front and then passes bare `np.exp`,
# because the clamp costs five times as much as the exponential it is guarding
# and it was being paid eighteen times per sub-timestep. Same equations either
# way -- the parameter only chooses how the overflow guard is applied.

def cal_k_inf(v_mv, _e=_exp):
    """Steady-state I_CaL activation. Half-activation -61 mV: 'low-threshold'
    means it opens just above rest, unlike the high-threshold Ca2+ current."""
    return _unwrap(1.0 / (1.0 + _e(-(np.asarray(v_mv, dtype=float) + 61.0) / 4.2)))


def cal_l_inf(v_mv, _e=_exp):
    """Steady-state I_CaL inactivation. Half-inactivation -85.5 mV: at a
    resting -60 mV this sits near 0, so the current is largely inactivated
    until the cell is hyperpolarized -- the de-inactivation that makes rebound
    bursting possible."""
    return _unwrap(1.0 / (1.0 + _e((np.asarray(v_mv, dtype=float) + 85.5) / 8.5)))


def cal_tau_l(v_mv, _e=_exp):
    """I_CaL inactivation time constant, ms. Rises steeply with
    hyperpolarization (~35 ms depolarized, ~165 ms at -85 mV), which is what
    sets the seconds-scale memory of the rebound."""
    v = np.asarray(v_mv, dtype=float)
    return _unwrap(20.0 * _e((v + 160.0) / 30.0) / (1.0 + _e((v + 84.0) / 7.3)) + 35.0)


# --- I_CaH: high-threshold Ca2+ (the [Ca]i source) -------------------------

def cah_r_rates(v_mv, _e=_exp):
    """(alpha, beta) for I_CaH activation. Both derived quantities need both
    rates, so they are computed together: r_inf = a/(a+b) and tau_r = 1/(a+b).

    beta = 0.02*(V+8.5)/(exp((V+8.5)/5)-1) is 0/0 at V = -8.5 mV; the limit there
    is 0.02*5 = 0.1, and np.where keeps the singular denominator out of the
    divide entirely rather than dividing and patching afterwards."""
    v = np.asarray(v_mv, dtype=float)
    alpha = 1.7 / (1.0 + _e(-(v - 5.0) / 13.9))
    x = v + 8.5
    singular = np.abs(x) < 1e-6
    denom = np.where(singular, 1.0, _e(x / 5.0) - 1.0)
    beta = np.where(singular, 0.1, 0.02 * x / denom)
    return alpha, beta


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


# --- I_KCa: Ca2+-activated K+ (gated by [Ca]i, voltage-independent) --------

# These take their parameters as arguments because they are the only
# [Ca]-dependent gates. The defaults mirror config.IOChannelParams so that
# calling them directly (tests, analysis) reproduces the simulated cell; the
# population always passes the configured values explicitly.
_KCA_CA_SCALE, _KCA_ALPHA_MAX, _KCA_BETA = 0.02, 0.01, 0.015


def kca_s_alpha(ca_um, ca_scale=_KCA_CA_SCALE, alpha_max=_KCA_ALPHA_MAX):
    """Opening rate rises linearly with [Ca]i then saturates -- the saturation
    is what keeps a big Ca2+ spike from producing an unboundedly long AHP."""
    return _unwrap(np.minimum(ca_scale * np.asarray(ca_um, dtype=float), alpha_max))


def kca_s_inf(ca_um, ca_scale=_KCA_CA_SCALE, alpha_max=_KCA_ALPHA_MAX, beta=_KCA_BETA):
    a = kca_s_alpha(ca_um, ca_scale, alpha_max)
    return _unwrap(a / (a + beta))


def kca_tau_s(ca_um, ca_scale=_KCA_CA_SCALE, alpha_max=_KCA_ALPHA_MAX, beta=_KCA_BETA):
    a = kca_s_alpha(ca_um, ca_scale, alpha_max)
    return _unwrap(1.0 / (a + beta))


# --- I_h: hyperpolarization-activated cation current -----------------------

def h_q_inf(v_mv, _e=_exp):
    """Activates on hyperpolarization (note the sign: q -> 1 as V -> -inf)."""
    return _unwrap(1.0 / (1.0 + _e((np.asarray(v_mv, dtype=float) + 80.0) / 4.0)))


def h_tau_q(v_mv, _e=_exp):
    """Very slow (hundreds of ms), which is why I_h sets the *frequency* of the
    subthreshold oscillation rather than participating in the spike."""
    v = np.asarray(v_mv, dtype=float)
    return _unwrap(1.0 / (_e(-0.086 * v - 14.6) + _e(0.070 * v - 1.87)))


# Voltage range the gating functions are evaluated over inside substep. Clamping V to this
# once keeps every exponent above within +-45, which float64 handles without overflow, so the
# inner loop can use a bare np.exp. The cell's V never approaches these bounds in normal
# operation (tests/test_io_channels.py asserts it stays inside +-150 mV); they exist only so a
# numerical excursion cannot produce inf, exactly as the per-argument clamp did before.
_V_GATING_MIN, _V_GATING_MAX = -200.0, 100.0

class IOPopulation:
    """N single-compartment conductance-based IO cells, integrated in lockstep.

    Integrated with exponential Euler on a sub-timestep: every gating variable
    and V is advanced as x <- x_inf + (x - x_inf)*exp(-dt/tau), which is
    unconditionally stable and stays accurate at dt values where forward Euler
    would blow up on the fast I_CaL activation. The outer simulation still runs
    at cfg.dt_ms (1 ms); this class silently substeps inside each call.

    Spontaneous firing is deterministic by default. The ionic machinery is a
    self-sustaining oscillator: with `i_app = 0` and `noise_sigma = 0` -- both
    the shipped values -- the cell free-runs at 3.00 Hz on I_CaL / I_KCa / I_h
    alone, and blocking I_CaL leaves it silent at -62.6 mV. Nothing pushes it.

    An Ornstein-Uhlenbeck noise current is available (`noise_sigma > 0`,
    standing in for synaptic bombardment and channel noise); it decides which
    subthreshold-oscillation peaks reach Ca2+ spike threshold, which smooths how
    steeply CF rate falls off with inhibition. It is off because the closed loop
    does not need it: the inhibition arrives as a handful of discrete DCN events
    rather than as a held conductance, and that granularity supplies the same
    smoothing (see config.dcn_io_gaba_gain). Cell-to-cell variability is a
    separate knob again -- `heterogeneity_cv` below, which diversifies the
    population without adding noise inside any one cell.

    `g_gap` is an (N, N) symmetric matrix of gap-junction conductances
    (mS/cm^2, zero diagonal) or None for uncoupled cells. It enters the current
    balance as I_gap(i) = sum_j g_gap[i,j] * (V_j - V_i), which splits exactly
    into a conductance term (the row sum, added to g_tot) and a current term
    (g_gap @ V, added to the numerator) -- so coupling folds into the same
    exponential-Euler V update as every other conductance, with no separate
    explicit term and no stability penalty. See sim/io_coupling.py.
    """

    def __init__(self, params, n_cells, dt_ms, rng=None, g_gap=None,
                 heterogeneity_cv=0.0, heterogeneity_seed=0):
        self.n = int(n_cells)
        self._p = params
        self.sub_dt_ms = min(params.sub_dt_ms, dt_ms)               # never substep coarser than the outer step
        self.n_sub = max(1, int(round(dt_ms / self.sub_dt_ms)))      # substeps per outer step
        self.sub_dt_ms = dt_ms / self.n_sub                          # exact divisor, so substeps tile the outer step
        self.rng = rng if rng is not None else np.random.default_rng()

        p = params
        self.V = np.full(self.n, p.v_init_mv, dtype=float)           # membrane potential, mV
        self.Ca = np.full(self.n, p.ca_init_um, dtype=float)         # intracellular [Ca2+], uM
        self.k = np.broadcast_to(cal_k_inf(self.V), (self.n,)).astype(float)   # I_CaL activation
        self.l = np.broadcast_to(cal_l_inf(self.V), (self.n,)).astype(float)   # I_CaL inactivation
        self.r = np.broadcast_to(cah_r_inf(self.V), (self.n,)).astype(float)   # I_CaH activation
        self.s = np.broadcast_to(                                              # I_KCa activation
            kca_s_inf(self.Ca, p.kca_ca_scale, p.kca_alpha_max, p.kca_beta), (self.n,)).astype(float)
        self.q = np.broadcast_to(h_q_inf(self.V), (self.n,)).astype(float)     # I_h activation
        self.noise = np.zeros(self.n)                                # OU current state, uA/cm^2

        # Per-cell maximal conductances. With cv = 0 these stay the scalars from `params`,
        # so the homogeneous fast path is untouched; above 0 they become (n,) arrays that
        # broadcast through the same arithmetic. The draw happens ONCE here, from a
        # dedicated seed, so the population is fixed for the life of the object and the
        # cells remain deterministic -- this is cell-to-cell variability, not noise.
        self._g_cal, self._g_cah, self._g_kca, self._g_h = self._draw_conductances(
            heterogeneity_cv, heterogeneity_seed)
        self.heterogeneity_cv = float(heterogeneity_cv)

        self._refractory_ms = np.zeros(self.n)                       # blocks re-detecting one Ca2+ spike as many CF events
        self._refresh_derived()                                      # OU coefficients, from noise_tau/noise_sigma

        self.display_rate_hz = np.zeros(self.n)                      # passive smoothed-spike-train readout, for logging
        self._g_gaba = np.zeros(self.n)                              # last GABA conductance seen, for currents()
        self.set_gap_junctions(g_gap)

    # --- configuration ----------------------------------------------------

    def _draw_conductances(self, cv, seed):
        """Lognormal per-cell spread on the three conductances that shape the rhythm:
        g_cal (the oscillation's driver), g_h (its frequency) and g_kca (the AHP that
        paces it). g_cah and g_leak are left alone -- varying g_leak would change each
        cell's input resistance, which would in turn change what a gap junction of a
        given conductance does, confounding heterogeneity with coupling strength.

        The lognormal is parameterised so the population mean is exactly the configured
        value and every draw is positive; cv is the coefficient of variation."""
        p = self._p
        base = (p.g_cal, p.g_cah, p.g_kca, p.g_h)
        if not cv or cv <= 0.0:
            return base
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        mu = -0.5 * sigma * sigma                      # makes E[exp(X)] == 1, so the mean is preserved
        rng = np.random.default_rng(seed)
        draw = lambda b: b * rng.lognormal(mu, sigma, size=self.n)
        return draw(p.g_cal), p.g_cah, draw(p.g_kca), draw(p.g_h)

    def set_gap_junctions(self, g_gap):
        """Install (or clear, with None) the IO-IO coupling matrix."""
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
        self._gap_row_sum = g_gap.sum(axis=1)                        # total coupling conductance seen by each cell

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, params):
        self.set_params(params)

    def set_params(self, params):
        """Swap the parameter set mid-run (the characterization protocols use
        this to inject an applied-current waveform). Re-derives the cached OU
        coefficients, which depend on noise_sigma / noise_tau_ms."""
        self._p = params
        self._refresh_derived()

    def _refresh_derived(self):
        p, dt = self._p, self.sub_dt_ms
        self._noise_decay = float(np.exp(-dt / p.noise_tau_ms))                      # OU exact-update coefficients,
        self._noise_kick = float(p.noise_sigma * np.sqrt(1.0 - self._noise_decay ** 2))  # recomputed on any param swap
        # These two exponential-Euler factors depend only on constants, so they are computed once
        # here rather than 10,000 times a second inside substep.
        self._k_step = float(1.0 - np.exp(-dt / p.cal_tau_k_ms))                     # I_CaL activation, constant tau
        self._ca_decay_factor = float(np.exp(-dt * p.ca_decay))                      # [Ca]i first-order extrusion

    # --- main integration -------------------------------------------------

    def step(self, dt_ms, g_gaba=0.0):
        """Advance every cell one outer timestep under its DCN GABA_A
        conductance (mS/cm^2, scalar or per-cell array). Returns a boolean
        array: True where that cell emitted a CF event (Ca2+ spike)."""
        g_gaba = np.broadcast_to(np.asarray(g_gaba, dtype=float), (self.n,))
        self._g_gaba = g_gaba                                        # stash for currents()
        noise_block = self.rng.standard_normal((self.n_sub, self.n))  # one RNG call per outer step, not per substep
        if advance_io is not None:
            fired = advance_io(self, g_gaba, noise_block, self.n_sub)
        else:
            fired = np.zeros(self.n, dtype=bool)
            for i in range(self.n_sub):
                fired |= self.substep(g_gaba, noise_block[i])            # a burst is one CF event regardless of substep count
        self._update_display_rate(dt_ms, fired)
        return fired

    def substep(self, g_gaba, xi):
        p = self._p
        dt = self.sub_dt_ms
        v = self.V
        # One clamp here replaces a clamp inside each of the eighteen exponentials below; see
        # _V_GATING_MIN. This bounds only the argument the gating functions are evaluated at --
        # self.V itself is never modified.
        vg = np.minimum(np.maximum(v, _V_GATING_MIN), _V_GATING_MAX)
        exp = np.exp

        self.noise = self.noise * self._noise_decay + self._noise_kick * xi   # OU membrane-current noise

        # --- conductances at the current V / [Ca] ---
        g_cal = self._g_cal * self.k * self.k * self.k * self.l       # k^3 * l
        g_cah = self._g_cah * self.r * self.r                         # r^2
        g_kca = self._g_kca * self.s
        g_h = self._g_h * self.q

        # --- V update: exponential Euler on dV/dt = -(sum g_i (V - E_i) - I)/C ---
        g_tot = g_cal + g_cah + g_kca + g_h + p.g_leak + g_gaba + self._gap_row_sum
        i_in = (g_cal * p.e_ca + g_cah * p.e_ca + g_kca * p.e_k + g_h * p.e_h
                + p.g_leak * p.e_leak + g_gaba * p.e_gaba
                + p.i_app + self.noise)                                # sum(g*E) + applied + noise
        if self.g_gap is not None:
            i_in = i_in + self.g_gap @ v                               # sum_j g_ij*V_j; the -g_ij*V_i half is in g_tot
        v_inf = i_in / g_tot                                          # steady state V the cell is currently pulled toward
        # tau_v = c_m/g_tot, so -dt/tau_v = -dt*g_tot/c_m: always negative, hence no clamp needed.
        v_new = v_inf + (v - v_inf) * exp(-dt * g_tot / p.c_m)

        # --- [Ca]i: driven by I_CaH influx, removed by first-order extrusion ---
        i_cah = g_cah * (v - p.e_ca)                                   # negative = inward = Ca2+ entering
        ca_inf = -p.ca_influx * i_cah / p.ca_decay                     # steady state of dCa/dt = -beta*I_CaH - k*Ca
        self.Ca = ca_inf + (self.Ca - ca_inf) * self._ca_decay_factor
        np.maximum(self.Ca, 0.0, out=self.Ca)                          # concentrations can't go negative

        # --- gating variables, all exponential Euler at the pre-update V ---
        self.k += (cal_k_inf(vg, exp) - self.k) * self._k_step         # tau_k is constant, so the factor is too
        self.l += (cal_l_inf(vg, exp) - self.l) * (1.0 - exp(-dt / cal_tau_l(vg, exp)))
        r_alpha, r_beta = cah_r_rates(vg, exp)                         # both derived gates need both rates
        r_sum = r_alpha + r_beta
        self.r += (r_alpha / r_sum - self.r) * (1.0 - exp(-dt * r_sum))   # tau_r = 1/(a+b), so dt/tau_r = dt*(a+b)
        self.q += (h_q_inf(vg, exp) - self.q) * (1.0 - exp(-dt / h_tau_q(vg, exp)))
        s_alpha = np.minimum(p.kca_ca_scale * self.Ca, p.kca_alpha_max)
        s_sum = s_alpha + p.kca_beta
        self.s += (s_alpha / s_sum - self.s) * (1.0 - exp(-dt * s_sum))    # tau_s = 1/(alpha+beta), likewise

        # --- CF event detection: upward crossing of the Ca2+ spike threshold ---
        np.maximum(self._refractory_ms - dt, 0.0, out=self._refractory_ms)
        fired = (v < p.v_spike_mv) & (v_new >= p.v_spike_mv) & (self._refractory_ms <= 0.0)
        self._refractory_ms[fired] = p.spike_refractory_ms             # one Ca2+ spike = one CF event

        self.V = v_new
        return fired

    def currents(self):
        """Per-channel current (uA/cm^2) at the present state, one array per
        channel. Diagnostics only -- recomputed on demand rather than cached
        per substep, which would cost more than the integration itself."""
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
            out["I_gap"] = self._gap_row_sum * self.V - self.g_gap @ self.V   # sum_j g_ij*(V_i - V_j)
        return out

    def _update_display_rate(self, dt_ms, fired):
        instantaneous_hz = np.where(fired, 1000.0 / dt_ms, 0.0)        # this bin's instantaneous "rate"
        self.display_rate_hz += (dt_ms / self._p.display_tau_ms) * (instantaneous_hz - self.display_rate_hz)


class IOConductanceNeuron:
    """Single-cell view of IOPopulation, for driving one olivary cell in
    isolation (the characterization protocols in
    experiments/run_io_characterization.py and the unit tests). Holds no
    physiology of its own -- it forwards to a population of size 1, so there is
    exactly one copy of the model equations."""

    def __init__(self, params, dt_ms, rng=None):
        self.pop = IOPopulation(params, 1, dt_ms, rng=rng)

    # State is exposed as scalars so callers can write `neuron.V` and get a number.
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
        """Advance one outer timestep under a DCN GABA_A conductance
        (mS/cm^2). Returns True if a CF event (Ca2+ spike) occurred."""
        return bool(self.pop.step(dt_ms, g_gaba)[0])

    def currents(self):
        return {name: float(value[0]) for name, value in self.pop.currents().items()}
