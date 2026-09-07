"""Unit tests for the spiking PKJ/DCN cells (sim/neurons.py).

Three groups: the f-I inversion that turns a target rate into a tonic current,
the conductance-based membrane itself, and the CF pause.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LIFParams
from sim.neurons import ExpSynapse, LIFPopulation, PKJPopulation, tonic_drive_for_rate

P = LIFParams()


def make(cls=LIFPopulation, n=200, baseline=50.0, dt=1.0, noise=0.0, seed=0, **kw):
    return cls(n, dt, P.tau_m_ms, P.v_th_mv, P.v_reset_mv, P.e_leak_mv, P.t_ref_ms, baseline,
                5.0, 10.0, 0.0, -75.0, noise_sigma_mv=noise, rng=np.random.default_rng(seed), **kw)


def measure(pop, secs, dt=1.0):
    total = sum(int(pop.step().sum()) for _ in range(int(secs * 1000 / dt)))
    return total / pop.n_units / secs


class TestTonicDriveInversion(unittest.TestCase):
    def test_zero_and_negative_targets_give_no_drive(self):
        self.assertEqual(tonic_drive_for_rate(0.0, 20.0, -50.0, -65.0, -70.0, 2.0), 0.0)

    def test_target_faster_than_the_refractory_period_is_rejected(self):
        with self.assertRaises(ValueError):
            tonic_drive_for_rate(600.0, 20.0, -50.0, -65.0, -70.0, 2.0)   # 1.67 ms ISI vs 2 ms refractory

    def test_drive_increases_with_the_target_rate(self):
        drives = [tonic_drive_for_rate(r, 20.0, -50.0, -65.0, -70.0, 2.0) for r in (10, 30, 50, 80)]
        self.assertTrue(all(a < b for a, b in zip(drives, drives[1:])))

    def test_an_unsynapsed_cell_fires_at_the_requested_rate(self):
        """Exact wherever the ISI is a whole number of timesteps. Where it is not
        (45 Hz -> 22.22 ms), a cell stepped on a 1 ms grid can only produce
        integer-ms intervals, so one step of quantization is the floor."""
        for target, tol in ((10.0, 0.01), (20.0, 0.01), (50.0, 0.01), (45.0, 0.6), (60.0, 1.3)):
            got = measure(make(baseline=target), 10.0)
            self.assertAlmostEqual(got, target, delta=tol, msg=f"target {target} Hz -> {got:.2f} Hz")

    def test_the_grid_correction_matters(self):
        """Without the half-step correction the crossing lands exactly on a step
        boundary and rounds late, costing a whole dt of ISI."""
        uncorrected = tonic_drive_for_rate(50.0, 20.0, -50.0, -65.0, -70.0, 2.0, dt_ms=0.0)
        corrected = tonic_drive_for_rate(50.0, 20.0, -50.0, -65.0, -70.0, 2.0, dt_ms=1.0)
        self.assertGreater(corrected, uncorrected)
        pop = make(baseline=50.0)
        pop.i_tonic = uncorrected
        self.assertLess(measure(pop, 10.0), 49.0)                       # ~47.6 Hz: one step late every ISI


class TestMembrane(unittest.TestCase):
    def test_voltage_stays_between_reset_and_threshold(self):
        pop = make(noise=3.0, seed=1)
        for _ in range(5000):
            pop.step()
            self.assertTrue(np.all(pop.V >= P.v_reset_mv - 20.0))       # noise can push below reset, but not far
            self.assertTrue(np.all(pop.V <= P.v_th_mv))                 # anything at/over threshold spikes and resets

    def test_excitation_speeds_the_cell_up_and_inhibition_slows_it_down(self):
        base = measure(make(baseline=40.0), 8.0)
        exc = make(baseline=40.0)
        for _ in range(8000):
            exc.exc.add(0.004); exc.step()
        inh = make(baseline=40.0)
        for _ in range(8000):
            inh.inh.add(0.004); inh.step()
        self.assertGreater(measure(exc, 4.0), base)
        self.assertLess(measure(inh, 4.0), base)

    def test_strong_inhibition_silences_the_cell(self):
        pop = make(baseline=50.0)
        fired = 0
        for _ in range(6000):
            pop.inh.add(0.5)
            fired += int(pop.step().sum())
        self.assertEqual(fired, 0)

    def test_refractory_period_bounds_the_interspike_interval(self):
        pop = make(n=1, baseline=200.0)                                  # drive it far above rheobase
        times = [t for t in range(20000) if pop.step()[0]]
        self.assertGreater(len(times), 10)
        self.assertGreaterEqual(np.diff(times).min(), P.t_ref_ms)

    def test_noise_does_not_shift_the_baseline_rate(self):
        quiet = measure(make(baseline=45.0, noise=0.0, seed=2), 15.0)
        noisy = measure(make(baseline=45.0, noise=3.0, seed=2), 15.0)
        self.assertAlmostEqual(quiet, noisy, delta=2.0)

    def test_noise_smooths_the_response_to_inhibition(self):
        """The reason the noise is there: a noiseless LIF has a hard rheobase, so
        its rate falls off a cliff over a narrow band of inhibitory conductance.
        The loop's feedback needs that transition to be graded."""
        def curve(noise):
            out = []
            for g in np.linspace(0.0, 0.03, 7):
                pop = make(baseline=60.0, noise=noise, seed=3)
                total = 0
                for _ in range(6000):
                    pop.inh.add(g); total += int(pop.step().sum())
                out.append(total / pop.n_units / 6.0)
            return np.asarray(out)
        steepest_quiet = np.abs(np.diff(curve(0.0))).max()
        steepest_noisy = np.abs(np.diff(curve(4.0))).max()
        self.assertLess(steepest_noisy, steepest_quiet)


class TestExpSynapse(unittest.TestCase):
    def test_conductance_decays_with_its_time_constant(self):
        syn = ExpSynapse(1, 10.0, 1.0, 0.0)
        syn.add(1.0)
        for _ in range(10):
            syn.decay()
        self.assertAlmostEqual(float(syn.g[0]), np.exp(-1.0), places=6)   # one tau -> 1/e

    def test_steady_state_matches_rate_times_tau_times_increment(self):
        """The identity every gain in config.py is derived from."""
        tau, inc, per_ms = 50.0, 0.02, 0.03
        syn = ExpSynapse(1, tau, 1.0, 0.0)
        rng = np.random.default_rng(0)
        tail = []
        for i in range(200000):
            syn.add(inc * rng.poisson(per_ms))
            if i > 5000:
                tail.append(float(syn.g[0]))
            syn.decay()
        self.assertAlmostEqual(np.mean(tail), per_ms * tau * inc, delta=0.05 * per_ms * tau * inc)


class TestCFPause(unittest.TestCase):
    def _pkj(self, pause_ms=10.0, **kw):
        return make(PKJPopulation, n=8, baseline=60.0, pause_g=1.0, pause_ms=pause_ms, e_pause_mv=-75.0, **kw)

    def test_pause_suppresses_firing_for_its_duration(self):
        pop = self._pkj()
        for _ in range(500):
            pop.step()
        pop.trigger_cf_pause()
        fired = [int(pop.step().sum()) for _ in range(10)]
        self.assertEqual(sum(fired), 0)
        self.assertGreater(sum(int(pop.step().sum()) for _ in range(60)), 0)   # and recovers afterwards

    def test_pause_is_per_cell(self):
        """One population holds every group's Purkinje cells, and each group is
        paused by its own climbing fiber -- so a mask must pause only those cells."""
        # A 30 ms pause, against a 60 Hz baseline (16.7 ms ISI): every unpaused cell is then
        # guaranteed a spike inside the window whatever phase it happens to be in.
        pop = self._pkj(pause_ms=30.0)
        for _ in range(500):
            pop.step()
        mask = np.zeros(8, dtype=bool); mask[:4] = True
        pop.trigger_cf_pause(mask)
        counts = np.zeros(8, dtype=int)
        for _ in range(30):
            counts += pop.step()
        self.assertEqual(int(counts[:4].sum()), 0)
        self.assertTrue(np.all(counts[4:] > 0), f"unpaused cells did not all fire: {counts}")

    def test_pause_shunts_incoming_excitation(self):
        """Modeled as a conductance, not a subtracted current, so it should hold
        the cell down even under excitation that would otherwise drive it hard."""
        pop = self._pkj()
        for _ in range(500):
            pop.step()
        pop.trigger_cf_pause()
        fired = 0
        for _ in range(10):
            pop.exc.add(0.02)                                            # strong drive during the pause
            fired += int(pop.step().sum())
        self.assertEqual(fired, 0)


if __name__ == "__main__":
    unittest.main()
