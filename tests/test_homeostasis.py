"""Homeostatic feedback direction, ratios, boundaries and checkpoint state."""
import pickle
import unittest
import numpy as np
from config import SimConfig
from sim.homeostasis import HomeostaticScaling
from sim.simulate import Simulation
from experiments.pipeline.common import exact_equal


class TestHomeostaticScaling(unittest.TestCase):
    def test_low_target_high_activity_scale_per_cell(self):
        h=HomeostaticScaling(3, 10, target_hz=50, rate_tau_s=.1, scaling_tau_s=100)
        w=np.tile([.1,.2,.3], (3,1));before=w.copy()
        for step in range(100):
            h.step(np.array([False, step%2==0, True]),w)
        self.assertTrue(np.all(w[0]>before[0]))
        np.testing.assert_array_equal(w[1],before[1])
        self.assertTrue(np.all(w[2]<before[2]))
        np.testing.assert_allclose(w/w[:,:1],before/before[:,:1],rtol=1e-15)
        np.testing.assert_allclose(h.cumulative_weight_change,(w-before).sum(axis=1),atol=1e-16)

    def test_sustained_activity_is_averaged_and_four_hour_change_is_slow(self):
        h=HomeostaticScaling(1,1)
        w=np.array([[.2,.4]])
        for _ in range(1000):h.step(np.array([False]),w)
        self.assertGreater(h.rate_hz[0],59)
        self.assertGreater(w[0,0],.2)
        self.assertLess(w[0,0]-.2,1e-5)

    def test_burn_in_senses_without_scaling_and_partial_bin_counts_only_active_time(self):
        h=HomeostaticScaling(1,10,rate_tau_s=.1,scaling_tau_s=100)
        w=np.array([[.2,.4]]);before=w.copy()
        for _ in range(150):h.step(np.array([False]),w,False)
        np.testing.assert_array_equal(w,before)
        self.assertEqual(h.n_updates,0)
        self.assertLess(h.rate_hz[0],1)
        for _ in range(50):h.step(np.array([False]),w,True)
        self.assertEqual(h.n_updates,1)
        self.assertAlmostEqual(float(np.log(w[0,0]/.2)),.5/100*(1-h.rate_hz[0]/60))

    def test_clipping_is_recorded_and_zero_stays_zero(self):
        h=HomeostaticScaling(1,10,rate_tau_s=.01,scaling_tau_s=1)
        w=np.array([[0.,.8,1.]])
        for _ in range(100):h.step(np.array([False]),w)
        np.testing.assert_array_equal(w,[[0.,1.,1.]])
        self.assertEqual(h.clipped_synapse_updates[0],2)
        self.assertAlmostEqual(h.cumulative_weight_change[0],.2)

    def test_resume_inside_rate_bin_is_bitwise_identical(self):
        h=HomeostaticScaling(2,10);w=np.array([[.2,.4],[.3,.6]])
        for i in range(137):h.step(np.array([i%2==0,i%3==0]),w)
        restored,rw=pickle.loads(pickle.dumps((h,w)))
        for i in range(137,402):
            spikes=np.array([i%2==0,i%3==0]);h.step(spikes,w);restored.step(spikes,rw)
        exact_equal(h,restored);np.testing.assert_array_equal(w,rw)

    def test_feedback_reduces_error_in_independent_rate_plant(self):
        # Accelerated unit-test time constant only; experiment uses four hours.
        h=HomeostaticScaling(2,10,target_hz=50,rate_tau_s=1,scaling_tau_s=20)
        w=np.array([[.1,.3],[.6,.9]]);initial=100*w.mean(axis=1)-50
        phase=np.zeros(2)
        for _ in range(30000):
            phase+=100*w.mean(axis=1)*.01
            spikes=phase>=1;phase-=spikes
            h.step(spikes,w)
        final=100*w.mean(axis=1)-50
        self.assertTrue(np.all(abs(final)<abs(initial)/10))

    def test_invalid_parameters_fail(self):
        for kw in [dict(target_hz=0),dict(rate_tau_s=-1),dict(scaling_tau_s=float('nan')),
                   dict(update_s=.0015),dict(w_min=-1)]:
            with self.assertRaises(ValueError):HomeostaticScaling(1,1,**kw)

    def test_network_learning_switch_and_baseline_settings(self):
        cfg=SimConfig(homeostatic_scaling=True,n_io=4,n_dcn=8,n_pf_per_pkj=20,seed=0)
        sim=Simulation(cfg);sim._plasticity_on=False;w=sim.weights.copy()
        for _ in range(1000):sim._step()
        np.testing.assert_array_equal(sim.weights,w)
        self.assertEqual(sim.homeostasis.n_updates,0)
        sim._plasticity_on=True
        for _ in range(1000):sim._step()
        self.assertEqual(sim.homeostasis.n_updates,1)
        self.assertEqual(cfg.io_heterogeneity_cv,0)
        self.assertEqual(cfg.pf_pkj_gain,.000137)
        self.assertEqual(cfg.weight_dependence,0)
        self.assertIsNone(Simulation(SimConfig(n_io=4,n_dcn=8)).homeostasis)


if __name__=='__main__':unittest.main()
