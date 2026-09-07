"""Unit tests for the conductance-based IO neuron (sim/io_channels.py).

Three groups: the gating functions in isolation, the integrator's numerical
behaviour, and the emergent cell-level properties the closed loop depends on
(spontaneous rhythm, T-type rebound, monotone response to inhibition).
"""
import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import IOChannelParams, SimConfig
from sim.io_channels import (
    IOConductanceNeuron,
    IOPopulation,
    _exp,
    cah_r_beta,
    cah_r_inf,
    cah_tau_r,
    cal_k_inf,
    cal_l_inf,
    cal_tau_l,
    h_q_inf,
    h_tau_q,
    kca_s_inf,
)
from sim.io_coupling import build_gap_junction_matrix, synchrony_index
from sim.simulate import Simulation

V_RANGE = np.linspace(-120.0, 60.0, 361)      # a wider sweep than the cell ever visits

# config.IOChannelParams defaults to noise_sigma = 0, so the cell's firing comes purely from its
# own ionic currents. Tests that are ABOUT the noise -- seed-dependence, cells decorrelating from
# one another, coupling pulling them back together -- have to ask for it explicitly.
def noisy(sigma=2.0, **overrides):
    return dataclasses.replace(IOChannelParams(), noise_sigma=sigma, **overrides)


def run(params, duration_ms, g_gaba=0.0, seed=0, settle_ms=2000.0, i_app_schedule=None):
    """Drive one cell and return (spike_times_ms, V trace sampled at 1 ms)."""
    neuron = IOConductanceNeuron(params, 1.0, rng=np.random.default_rng(seed))
    for _ in range(int(settle_ms)):
        neuron.step(1.0, g_gaba)
    base = params.i_app
    spikes, v = [], np.empty(int(duration_ms))
    for i in range(int(duration_ms)):
        if i_app_schedule is not None:
            neuron.p = dataclasses.replace(params, i_app=base + i_app_schedule(float(i)))
        if neuron.step(1.0, g_gaba):
            spikes.append(float(i))
        v[i] = neuron.V
    return np.asarray(spikes), v


class TestGatingFunctions(unittest.TestCase):
    def test_all_gates_are_probabilities(self):
        for name, fn in [("k", cal_k_inf), ("l", cal_l_inf), ("r", cah_r_inf), ("q", h_q_inf)]:
            values = np.array([fn(v) for v in V_RANGE])
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)), f"{name}_inf left [0, 1]")
            self.assertTrue(np.all(np.isfinite(values)), f"{name}_inf produced a non-finite value")

    def test_time_constants_are_positive_and_finite(self):
        for name, fn in [("tau_l", cal_tau_l), ("tau_r", cah_tau_r), ("tau_q", h_tau_q)]:
            values = np.array([fn(v) for v in V_RANGE])
            self.assertTrue(np.all(values > 0.0), f"{name} went non-positive")
            self.assertTrue(np.all(np.isfinite(values)), f"{name} went non-finite")

    def test_cal_activation_rises_with_depolarization(self):
        k = [cal_k_inf(v) for v in V_RANGE]
        self.assertTrue(np.all(np.diff(k) >= 0.0))
        self.assertLess(cal_k_inf(-80.0), 0.05)                  # shut at rest
        self.assertGreater(cal_k_inf(-40.0), 0.95)               # open by the time the spike is underway

    def test_cal_inactivation_is_relieved_by_hyperpolarization(self):
        """The T-type signature: at rest the current is largely inactivated, and
        hyperpolarizing is what makes it available again."""
        l = [cal_l_inf(v) for v in V_RANGE]
        self.assertTrue(np.all(np.diff(l) <= 0.0))               # monotone decreasing in V
        self.assertLess(cal_l_inf(-60.0), 0.05)                  # inactivated at rest
        self.assertGreater(cal_l_inf(-95.0), 5 * cal_l_inf(-60.0))  # de-inactivated when hyperpolarized

    def test_cal_inactivation_is_slower_when_hyperpolarized(self):
        self.assertGreater(cal_tau_l(-90.0), cal_tau_l(-40.0))

    def test_h_current_activates_on_hyperpolarization(self):
        q = [h_q_inf(v) for v in V_RANGE]
        self.assertTrue(np.all(np.diff(q) <= 0.0))               # note the inverted sign vs. the Ca2+ gates
        self.assertGreater(h_q_inf(-100.0), 0.95)
        self.assertLess(h_q_inf(-60.0), 0.05)

    def test_kca_activation_rises_with_calcium_then_saturates(self):
        s = [kca_s_inf(c) for c in np.linspace(0.0, 10.0, 200)]
        self.assertTrue(np.all(np.diff(s) >= 0.0))
        self.assertAlmostEqual(kca_s_inf(0.0), 0.0)              # closed with no Ca2+: it is Ca2+-gated, not voltage-gated
        self.assertAlmostEqual(kca_s_inf(5.0), kca_s_inf(50.0))  # alpha_s saturates, bounding the AHP

    def test_cah_beta_is_finite_through_its_removable_singularity(self):
        """beta_r = 0.02*(V+8.5)/(exp((V+8.5)/5)-1) is 0/0 at V = -8.5 mV."""
        self.assertAlmostEqual(cah_r_beta(-8.5), 0.1, places=6)
        near = [cah_r_beta(v) for v in np.linspace(-8.6, -8.4, 21)]
        self.assertTrue(np.all(np.isfinite(near)))
        self.assertLess(max(near) - min(near), 0.01)             # continuous across the removable point

    def test_exp_helper_does_not_overflow(self):
        for x in (-1e9, -800.0, 0.0, 800.0, 1e9):
            self.assertTrue(np.isfinite(_exp(x)))


class TestIntegration(unittest.TestCase):
    def test_long_run_stays_finite_and_physiological(self):
        spikes, v = run(IOChannelParams(), 10000.0)
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertGreater(v.min(), -150.0)
        self.assertLess(v.max(), 150.0)

    def test_calcium_never_goes_negative(self):
        p = IOChannelParams()
        neuron = IOConductanceNeuron(p, 1.0, rng=np.random.default_rng(0))
        for _ in range(8000):
            neuron.step(1.0, 0.0)
            self.assertGreaterEqual(neuron.Ca, 0.0)

    def test_substep_convergence(self):
        """The default sub_dt must give the same answer as a 4x finer one; if it
        does not, the exponential-Euler step is too coarse for these kinetics."""
        base = dataclasses.replace(IOChannelParams(), noise_sigma=0.0)
        isis = []
        for sub_dt in (base.sub_dt_ms, base.sub_dt_ms / 4.0):
            spikes, _ = run(dataclasses.replace(base, sub_dt_ms=sub_dt), 9000.0)
            self.assertGreater(len(spikes), 5)
            isis.append(np.diff(spikes).mean())
        self.assertLess(abs(isis[0] - isis[1]) / isis[1], 0.02)   # within 2%

    def test_one_calcium_spike_yields_one_cf_event(self):
        """The refractory period must stop a single broad Ca2+ spike being
        counted once per sub-threshold-crossing sample."""
        p = dataclasses.replace(IOChannelParams(), noise_sigma=0.0)
        spikes, _ = run(p, 10000.0)
        self.assertGreater(len(spikes), 5)
        self.assertGreaterEqual(np.diff(spikes).min(), p.spike_refractory_ms)

    def test_same_seed_reproduces_the_spike_train(self):
        a, _ = run(noisy(), 6000.0, seed=7)
        b, _ = run(noisy(), 6000.0, seed=7)
        np.testing.assert_array_equal(a, b)
        c, _ = run(noisy(), 6000.0, seed=8)
        self.assertFalse(len(a) == len(c) and np.array_equal(a, c))

    def test_without_noise_the_cell_is_deterministic_regardless_of_seed(self):
        """The flip side, and the point of the default: with noise_sigma = 0 the
        spike train is a property of the channels alone, not of the RNG."""
        a, _ = run(IOChannelParams(), 6000.0, seed=7)
        c, _ = run(IOChannelParams(), 6000.0, seed=8)
        np.testing.assert_array_equal(a, c)
        self.assertGreater(len(a), 5)


class TestCellBehaviour(unittest.TestCase):
    """Emergent properties, each one something the closed loop relies on."""

    def setUp(self):
        self.quiet = dataclasses.replace(IOChannelParams(), noise_sigma=0.0)  # noiseless: mechanism, not statistics

    def test_spontaneous_rhythm_without_any_applied_current(self):
        """The whole point of the exercise: firing with i_app = 0, driven only
        by the cell's own channels."""
        self.assertEqual(self.quiet.i_app, 0.0)
        spikes, v = run(self.quiet, 8000.0)
        rate = len(spikes) / 8.0
        self.assertGreater(rate, 1.0)
        self.assertLess(rate, 10.0)                               # an olivary rhythm, not a runaway
        self.assertGreater(v.max(), 0.0)                          # a real regenerative Ca2+ spike

    def test_blocking_t_type_calcium_abolishes_the_rhythm(self):
        spikes, _ = run(dataclasses.replace(self.quiet, g_cal=0.0), 8000.0)
        self.assertEqual(len(spikes), 0)

    def test_blocking_calcium_activated_potassium_abolishes_the_rhythm(self):
        """Without I_KCa nothing repolarizes the Ca2+ spike, so the cell latches
        into depolarization block instead of oscillating."""
        spikes, v = run(dataclasses.replace(self.quiet, g_kca=0.0), 8000.0)
        self.assertEqual(len(spikes), 0)
        self.assertGreater(v.min(), 0.0)                          # stuck depolarized, not silent at rest

    def test_h_current_speeds_the_rhythm(self):
        with_h, _ = run(self.quiet, 12000.0)
        without_h, _ = run(dataclasses.replace(self.quiet, g_h=0.0), 12000.0)
        self.assertGreater(len(with_h), len(without_h))

    def test_rebound_burst_after_hyperpolarization_requires_t_type(self):
        """Hyperpolarize for 800 ms, release, and a low-threshold Ca2+ spike
        should follow within a few ms -- far sooner than the free-running
        period -- and only when I_CaL is available."""
        step_end = 1300.0
        schedule = lambda ms: -4.0 if 500.0 <= ms < step_end else 0.0
        free, _ = run(self.quiet, 8000.0)
        period_ms = np.diff(free).mean()

        spikes, v = run(self.quiet, 3000.0, i_app_schedule=schedule)
        during = v[int(step_end) - 200]
        self.assertLess(during, -75.0)                            # the step really did hyperpolarize the cell
        rebound = spikes[(spikes >= step_end) & (spikes < step_end + 400.0)]
        self.assertGreater(len(rebound), 0)
        self.assertLess(rebound[0] - step_end, 0.5 * period_ms)   # a rebound, not just the next cycle

        blocked, _ = run(dataclasses.replace(self.quiet, g_cal=0.0), 3000.0, i_app_schedule=schedule)
        self.assertEqual(len(blocked[(blocked >= step_end) & (blocked < step_end + 400.0)]), 0)

    def test_cf_rate_decreases_monotonically_with_inhibition(self):
        """The loop's negative-feedback limb. If this is not monotone, H1's
        equilibrium argument does not hold.

        Measured on a population rather than one cell run for longer: eight
        uncoupled cells cost the same per step as one, so this buys the same
        forty cell-seconds per point in a quarter of the wall time, from eight
        independent noise streams instead of one."""
        p = noisy()

        def pooled_rate(g_gaba, n=8, secs=5.0):
            pop = IOPopulation(p, n, 1.0, rng=np.random.default_rng(0))
            for _ in range(2000):
                pop.step(1.0, g_gaba)                                 # settle
            fired = sum(int(pop.step(1.0, g_gaba).sum()) for _ in range(int(secs * 1000)))
            return fired / n / secs

        rates = [pooled_rate(g) for g in (0.0, 0.25, 0.5, 0.75)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])), f"not monotone: {rates}")
        self.assertGreater(rates[0], 1.0)
        self.assertLess(rates[-1], 0.2)                           # strong inhibition silences the cell

    def test_inhibition_hyperpolarizes_and_shunts(self):
        """GABA_A is a conductance here, not a subtracted drive, so it must both
        pull V toward E_Cl and lower the input resistance."""
        _, v_free = run(self.quiet, 4000.0, g_gaba=0.0)
        _, v_inh = run(self.quiet, 4000.0, g_gaba=1.5)
        self.assertLess(v_inh.mean(), v_free[v_free < -40.0].mean())


class TestPopulationEquivalence(unittest.TestCase):
    """IOConductanceNeuron is a view over a size-1 IOPopulation, so the two must
    be the same cell -- there is only one copy of the model equations, and this
    pins that the adapter does not quietly diverge from it."""

    def test_single_cell_matches_a_population_of_one(self):
        p = IOChannelParams()
        neuron = IOConductanceNeuron(p, 1.0, rng=np.random.default_rng(11))
        pop = IOPopulation(p, 1, 1.0, rng=np.random.default_rng(11))
        for _ in range(4000):
            a = neuron.step(1.0, 0.2)
            b = bool(pop.step(1.0, 0.2)[0])
            self.assertEqual(a, b)
        self.assertAlmostEqual(float(neuron.V), float(pop.V[0]), places=10)
        self.assertAlmostEqual(float(neuron.Ca), float(pop.Ca[0]), places=10)

    def test_uncoupled_cells_are_independent(self):
        """With g_gap = 0 an eight-cell population must reproduce eight cells run
        separately -- otherwise the population step is leaking state between them."""
        p = dataclasses.replace(IOChannelParams(), noise_sigma=0.0)   # noiseless: the RNG streams differ per layout
        pop = IOPopulation(p, 8, 1.0, rng=np.random.default_rng(0), g_gap=None)
        one = IOPopulation(p, 1, 1.0, rng=np.random.default_rng(0))
        for _ in range(3000):
            pop.step(1.0, 0.2); one.step(1.0, 0.2)
        self.assertTrue(np.allclose(pop.V, pop.V[0]))                 # identical cells, identical drive
        self.assertAlmostEqual(float(pop.V[0]), float(one.V[0]), places=8)


class TestGapJunctions(unittest.TestCase):
    def test_matrix_is_symmetric_with_a_zero_diagonal(self):
        for topology in ("all_to_all", "ring", "nearest_k"):
            g = build_gap_junction_matrix(8, 0.01, topology, n_neighbors=2)
            np.testing.assert_allclose(g, g.T)
            np.testing.assert_allclose(np.diag(g), 0.0)

    def test_topologies_have_the_expected_degree(self):
        self.assertTrue(np.all((build_gap_junction_matrix(8, 0.01, "all_to_all") > 0).sum(axis=1) == 7))
        self.assertTrue(np.all((build_gap_junction_matrix(8, 0.01, "ring") > 0).sum(axis=1) == 2))
        self.assertTrue(np.all((build_gap_junction_matrix(8, 0.01, "nearest_k", 2) > 0).sum(axis=1) == 4))

    def test_unknown_topology_is_rejected(self):
        with self.assertRaises(ValueError):
            build_gap_junction_matrix(4, 0.01, "fictional")

    def test_a_single_cell_has_no_junctions(self):
        np.testing.assert_allclose(build_gap_junction_matrix(1, 0.01, "all_to_all"), np.zeros((1, 1)))

    def test_asymmetric_matrix_is_rejected(self):
        """A gap junction is a resistor: it must conduct equally both ways."""
        bad = np.array([[0.0, 0.01], [0.02, 0.0]])
        with self.assertRaises(ValueError):
            IOPopulation(IOChannelParams(), 2, 1.0, rng=np.random.default_rng(0), g_gap=bad)

    def test_coupling_conserves_current_between_a_pair(self):
        """I_gap must sum to zero across a coupled pair -- charge leaving one
        cell through the junction is charge entering the other."""
        pop = IOPopulation(IOChannelParams(), 2, 1.0, rng=np.random.default_rng(0),
                            g_gap=np.array([[0.0, 0.02], [0.02, 0.0]]))
        pop.V[:] = [-70.0, -40.0]                                     # force a voltage difference
        i_gap = pop.currents()["I_gap"]
        self.assertAlmostEqual(float(i_gap.sum()), 0.0, places=12)
        self.assertLess(i_gap[0] * i_gap[1], 0.0)                     # equal and opposite

    def test_coupling_pulls_voltages_together(self):
        """The point of the exercise: coupled cells should track each other more
        closely than uncoupled ones. Needs noise: identical noiseless cells start
        and stay in lock-step, so there is nothing for coupling to pull together."""
        p = noisy()
        spreads = {}
        for g_gap in (0.0, 0.02):
            matrix = build_gap_junction_matrix(6, g_gap, "all_to_all")
            pop = IOPopulation(p, 6, 1.0, rng=np.random.default_rng(5), g_gap=matrix)
            for _ in range(2000):
                pop.step(1.0, 0.3)                                    # settle
            spread = []
            for _ in range(6000):
                pop.step(1.0, 0.3)
                spread.append(pop.V.std())
            spreads[g_gap] = float(np.mean(spread))
        self.assertLess(spreads[0.02], spreads[0.0],
                         f"coupling did not reduce voltage spread: {spreads}")

    def test_coupling_raises_cf_synchrony(self):
        """And the reason it matters: gap junctions should make the climbing-fiber
        output of neighbouring cells co-occur more than chance. Noisy, for the same
        reason as the test above."""
        p = noisy()
        sync = {}
        for g_gap in (0.0, 0.02):
            matrix = build_gap_junction_matrix(6, g_gap, "all_to_all")
            pop = IOPopulation(p, 6, 1.0, rng=np.random.default_rng(5), g_gap=matrix)
            for _ in range(2000):
                pop.step(1.0, 0.3)
            trains, n_steps = [[] for _ in range(6)], 22000
            for t in range(n_steps):
                for i in np.flatnonzero(pop.step(1.0, 0.3)):
                    trains[i].append(float(t))
            sync[g_gap] = synchrony_index(trains, 0.0, float(n_steps), bin_ms=20.0)
        self.assertGreater(sync[0.02], sync[0.0], f"coupling did not raise synchrony: {sync}")


class TestSimulationWiring(unittest.TestCase):
    def test_network_runs_and_every_population_spikes(self):
        cfg = SimConfig(duration_s=4.0, burn_in_s=1.0, seed=0, trace_window_s=1.0)
        log = Simulation(cfg).run()
        for name, trains in (("IO", log.io_spikes), ("PKJ", log.pkj_spikes), ("DCN", log.dcn_spikes)):
            self.assertTrue(any(len(t) for t in trains), f"no {name} cell ever fired")
            for t in trains:
                self.assertTrue(np.all(np.diff(t) > 0), f"{name} spike times not increasing")
                if len(t):
                    self.assertLessEqual(t.max(), 4000.0)
        self.assertTrue(np.all(np.isfinite(log.trace_io_v)))
        self.assertTrue(np.all(np.isfinite(log.trace_pkj_v)))
        self.assertTrue(np.all(np.isfinite(log.trace_dcn_v)))

    def test_each_group_is_driven_by_its_own_climbing_fiber(self):
        """cf_source_of_pkj must route each Purkinje cell to the one climbing fiber
        that contacts it; if it did not, plasticity across the microzone would
        resolve against the wrong CF."""
        cfg = SimConfig(duration_s=1.0, burn_in_s=0.0, seed=0)
        sim = Simulation(cfg)
        np.testing.assert_array_equal(sim.plasticity.cf_source_of_pkj, sim.io_of_pkj)
        self.assertEqual(sim.plasticity.n_cf_sources, cfg.n_io)

    def test_io_noise_does_not_consume_the_pf_random_stream(self):
        """The IO's membrane noise is drawn from its own generator, so the PF
        Poisson stream is unaffected by how the olive is configured -- otherwise
        a coupled and an uncoupled run could not be compared on a common seed."""
        states = []
        for g_gap in (0.0, 0.02):
            sim = Simulation(SimConfig(gap_g=g_gap, duration_s=2.0, burn_in_s=0.0, seed=3))
            sim.run()
            states.append(sim.rng.bit_generator.state)            # advanced only by PF draws
        self.assertEqual(states[0], states[1])


if __name__ == "__main__":
    unittest.main()
