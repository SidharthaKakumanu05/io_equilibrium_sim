"""Cell-to-cell heterogeneity and circuit ablation.

These are the two knobs the overnight sweeps vary, so they need to do exactly
what they claim: heterogeneity must make cells DIFFER without making them
NOISY, and ablation must remove the feedback limb and nothing else.
"""
import unittest
import numpy as np

from config import SimConfig, IOChannelParams
from sim.io_channels import IOPopulation
from sim.simulate import Simulation


def small(**kw):
    cfg = SimConfig(**kw)
    cfg.n_io, cfg.n_dcn = 4, 8
    cfg.duration_s = 2.0
    cfg.burn_in_s = 0.5
    return cfg


class TestHeterogeneity(unittest.TestCase):
    def test_zero_cv_leaves_scalars_untouched(self):
        """The homogeneous path must stay exactly as it was -- same scalars, no arrays."""
        p = IOChannelParams()
        pop = IOPopulation(p, 40, 1.0, heterogeneity_cv=0.0)
        self.assertEqual(pop._g_cal, p.g_cal)
        self.assertEqual(pop._g_kca, p.g_kca)
        self.assertEqual(pop._g_h, p.g_h)

    def test_spread_scales_with_cv_and_preserves_the_mean(self):
        p = IOChannelParams()
        last = 0.0
        for cv in (0.05, 0.10, 0.20):
            g = np.atleast_1d(IOPopulation(p, 400, 1.0, heterogeneity_cv=cv)._g_cal)
            realized = float(g.std() / g.mean())
            self.assertGreater(realized, last)                 # more cv, more spread
            self.assertAlmostEqual(float(g.mean()), p.g_cal, delta=0.05 * p.g_cal)
            self.assertTrue(np.all(g > 0.0), "conductances must stay positive")
            last = realized

    def test_draw_is_deterministic_given_the_seed(self):
        """Heterogeneity is variability BETWEEN cells, not noise WITHIN a run: the same
        seed must give the same population every time, so runs stay reproducible."""
        p = IOChannelParams()
        a = np.atleast_1d(IOPopulation(p, 40, 1.0, heterogeneity_cv=0.1, heterogeneity_seed=7)._g_cal)
        b = np.atleast_1d(IOPopulation(p, 40, 1.0, heterogeneity_cv=0.1, heterogeneity_seed=7)._g_cal)
        c = np.atleast_1d(IOPopulation(p, 40, 1.0, heterogeneity_cv=0.1, heterogeneity_seed=8)._g_cal)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c), "a different seed must give a different population")

    def test_heterogeneity_does_not_add_drive_or_noise(self):
        """The population mean conductance is preserved and noise_sigma is untouched, so
        the olive is not being pushed -- only diversified."""
        cfg = small()
        cfg.io_heterogeneity_cv = 0.2
        sim = Simulation(cfg)
        self.assertEqual(cfg.io_channels.noise_sigma, 0.0)
        self.assertEqual(cfg.io_channels.i_app, 0.0)
        self.assertAlmostEqual(float(np.mean(sim.io._g_cal)), cfg.io_channels.g_cal,
                               delta=0.15 * cfg.io_channels.g_cal)

    def test_identical_cells_stay_identical_and_diverse_cells_diverge(self):
        """The degeneracy check, at the level that matters: with a complete DCN->IO
        projection and no heterogeneity the cells integrate to the same trace."""
        cfg = small()
        cfg.n_dcn_per_io = cfg.n_dcn                      # complete projection: identical input
        cfg.gap_g = 0.0
        homo = Simulation(cfg).run().trace_io_v
        spread = np.abs(homo - homo[:, :1]).max()
        self.assertLess(spread, 1e-9, "identical cells with identical input must not differ")

        cfg2 = small()
        cfg2.n_dcn_per_io = cfg2.n_dcn
        cfg2.gap_g = 0.0
        cfg2.io_heterogeneity_cv = 0.15
        hetero = Simulation(cfg2).run().trace_io_v
        self.assertGreater(np.abs(hetero - hetero[:, :1]).max(), 1e-3,
                           "heterogeneous cells must produce different trajectories")


class TestDcnIoAblation(unittest.TestCase):
    def test_ablation_zeroes_the_drive(self):
        self.assertEqual(Simulation(small(ablate_dcn_io=True)).dcn_io_drive, 0.0)
        self.assertGreater(Simulation(small(ablate_dcn_io=False)).dcn_io_drive, 0.0)

    def test_ablation_removes_inhibition_but_leaves_the_rest_wired(self):
        """Only the feedback limb goes. The nucleus still spikes, the wiring tables are
        unchanged, and the climbing fibers still teach -- otherwise the experiment would
        confound 'no feedback' with 'no circuit'."""
        cfg = small(ablate_dcn_io=True)
        sim = Simulation(cfg)
        log = sim.run()
        self.assertTrue(np.all(log.trace_io_gaba == 0.0), "no GABA may reach the olive")
        self.assertGreater(sum(len(t) for t in log.dcn_spikes), 0, "DCN must still fire")
        self.assertTrue(np.all(sim.m_dcn_to_io.sum(axis=1) > 0), "wiring table stays intact")
        self.assertEqual(sim.conn.meta["dcn_to_io_complete"],
                         cfg.n_dcn_per_io >= cfg.n_dcn)

    def test_opening_the_loop_raises_cf_rate_and_drives_weights_down(self):
        """The H1 falsification. Without inhibition the olive runs at its intrinsic rate,
        which is far above the plasticity's balance point, so LTD dominates and the
        weights are driven toward the floor instead of settling mid-range."""
        closed = small(); closed.duration_s = 20.0
        opened = small(ablate_dcn_io=True); opened.duration_s = 20.0
        lc, lo = Simulation(closed).run(), Simulation(opened).run()
        rate = lambda lg: np.mean([len(t) for t in lg.io_spikes]) / (lg.duration_ms / 1000.0)
        self.assertGreater(rate(lo), 2.0 * rate(lc),
                           "cutting the feedback limb must let CF rate run free")
        self.assertLess(float(np.mean(lo.final_weights)), float(np.mean(lc.final_weights)),
                        "a faster CF rate must push weights down relative to the closed loop")


if __name__ == "__main__":
    unittest.main()
