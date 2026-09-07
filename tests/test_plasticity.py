"""Unit tests for sim/plasticity.py window boundaries (LTD: delay 1..ltd_steps,
null: delay ltd_steps+1..ltd_steps+null_steps, LTP: any larger delay or no CF)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.plasticity import Plasticity


def no_spikes(n_pkj=1, n_pf=1):
    return np.zeros((n_pkj, n_pf), dtype=bool)


def run_single_spike_then_cf(plasticity, weights, cf_delay, n_extra_steps=12):
    """Fire one spike on synapse (0, 0) at step 0, then step forward, placing
    a CF event at exactly `cf_delay` steps after the spike (cf_delay=None
    means no CF at all)."""
    spike = np.zeros_like(weights, dtype=bool)
    spike[0, 0] = True
    plasticity.step(spike, cf_event=(cf_delay == 0), weights=weights)
    for i in range(1, n_extra_steps):
        cf_now = (cf_delay is not None and i == cf_delay)
        plasticity.step(no_spikes(*weights.shape), cf_event=cf_now, weights=weights)


class TestTwoWindowPlasticity(unittest.TestCase):
    """dt_ms=1, ltd_window_ms=5 -> ltd_steps=5 (LTD window = delay 1..5)."""

    def setUp(self):
        self.w_init = 0.5
        self.delta_plus = 0.01
        self.delta_minus = 0.05

    def _new(self, n_pkj=1, n_pf=2):
        plasticity = Plasticity(n_pkj, n_pf, dt_ms=1.0, ltd_window_ms=5.0,
                                 delta_plus=self.delta_plus, delta_minus=self.delta_minus)
        weights = np.full((n_pkj, n_pf), self.w_init)
        return plasticity, weights

    def test_ltd_applied_when_cf_lands_within_window(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=3)  # well inside [1,5]
        self.assertAlmostEqual(weights[0, 0], self.w_init - self.delta_minus, places=9)
        self.assertAlmostEqual(weights[0, 1], self.w_init, places=9)  # never spiked

    def test_ltd_applied_at_inclusive_window_boundary(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=5)  # exactly ltd_steps
        self.assertAlmostEqual(weights[0, 0], self.w_init - self.delta_minus, places=9)

    def test_ltp_applied_once_window_has_closed(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=6)  # one step past the window
        self.assertAlmostEqual(weights[0, 0], self.w_init + self.delta_plus, places=9)

    def test_cf_in_the_same_timestep_as_the_spike_potentiates(self):
        """Delay 0 is BEFORE the window, not inside it: the rule is about a CF that
        FOLLOWS a PF spike. `step` folds this step's CF into the cumulative counters
        before recording them, so a simultaneous CF reads as already-past."""
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=0)
        self.assertAlmostEqual(weights[0, 0], self.w_init + self.delta_plus, places=9)

    def test_ltp_applied_when_no_cf_at_all(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=None)
        self.assertAlmostEqual(weights[0, 0], self.w_init + self.delta_plus, places=9)

    def test_weights_clip_to_bounds(self):
        w_min, w_max = 0.0, 1.0
        plasticity = Plasticity(1, 1, dt_ms=1.0, ltd_window_ms=5.0,
                                 delta_plus=0.01, delta_minus=0.5, w_min=w_min, w_max=w_max)
        weights = np.full((1, 1), 0.2)
        run_single_spike_then_cf(plasticity, weights, cf_delay=3)  # big LTD would go negative
        self.assertEqual(weights[0, 0], w_min)

    def test_independent_pf_units_resolved_independently(self):
        # synapse 0 spikes and gets caught by a CF; synapse 1 spikes later and doesn't.
        plasticity, weights = self._new()
        spike_a = no_spikes(1, 2)
        spike_a[0, 0] = True
        plasticity.step(spike_a, cf_event=False, weights=weights)          # step 0
        plasticity.step(no_spikes(1, 2), cf_event=False, weights=weights)  # step 1
        plasticity.step(no_spikes(1, 2), cf_event=False, weights=weights)  # step 2
        plasticity.step(no_spikes(1, 2), cf_event=True, weights=weights)   # step 3: CF (delay 3 for synapse 0)

        spike_b = no_spikes(1, 2)
        spike_b[0, 1] = True
        plasticity.step(spike_b, cf_event=False, weights=weights)          # step 4: synapse 1 spikes
        for _ in range(6):                                                 # no further CF -> LTP for synapse 1
            plasticity.step(no_spikes(1, 2), cf_event=False, weights=weights)

        self.assertAlmostEqual(weights[0, 0], self.w_init - self.delta_minus, places=9)
        self.assertAlmostEqual(weights[0, 1], self.w_init + self.delta_plus, places=9)


class TestThreeWindowPlasticity(unittest.TestCase):
    """dt_ms=1, ltd_window_ms=5, null_window_ms=3 -> LTD: delay 1-5,
    null: delay 6-8, LTP: delay >= 9 or no CF."""

    def setUp(self):
        self.w_init = 0.5
        self.delta_plus = 0.01
        self.delta_minus = 0.05

    def _new(self):
        plasticity = Plasticity(1, 1, dt_ms=1.0, ltd_window_ms=5.0, null_window_ms=3.0,
                                 delta_plus=self.delta_plus, delta_minus=self.delta_minus)
        weights = np.full((1, 1), self.w_init)
        return plasticity, weights

    def test_cf_in_ltd_window_still_causes_ltd(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=3, n_extra_steps=14)
        self.assertAlmostEqual(weights[0, 0], self.w_init - self.delta_minus, places=9)

    def test_cf_in_null_window_causes_no_change(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=7, n_extra_steps=14)
        self.assertAlmostEqual(weights[0, 0], self.w_init, places=9)

    def test_cf_after_null_window_causes_ltp(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=9, n_extra_steps=14)
        self.assertAlmostEqual(weights[0, 0], self.w_init + self.delta_plus, places=9)

    def test_no_cf_at_all_causes_ltp(self):
        plasticity, weights = self._new()
        run_single_spike_then_cf(plasticity, weights, cf_delay=None, n_extra_steps=14)
        self.assertAlmostEqual(weights[0, 0], self.w_init + self.delta_plus, places=9)


if __name__ == "__main__":
    unittest.main()
