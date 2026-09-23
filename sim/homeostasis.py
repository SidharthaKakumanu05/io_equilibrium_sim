"""Slow, postsynaptic firing-rate-dependent multiplicative PF weight scaling.

This is a phenomenological homeostatic rule, not a fitted Purkinje-cell
molecular model. Every incoming PF weight on a cell receives the same factor:
    d log(w_ij) / dt = (1 - r_i / target_hz) / scaling_tau_s.
The existing weight bounds also apply; clipping can break ratio preservation.
Rate sensing uses an exponential average of one-second spike-count bins.
No randomness, presynaptic activity test, or climbing-fiber event is required.
"""
import math
import numpy as np


class HomeostaticScaling:
    def __init__(self, n_cells, dt_ms, target_hz=60.0, rate_tau_s=60.0,
                 scaling_tau_s=14400.0, update_s=1.0, w_min=0.0, w_max=1.0):
        for name, value in [('dt_ms', dt_ms), ('target_hz', target_hz),
                            ('rate_tau_s', rate_tau_s), ('scaling_tau_s', scaling_tau_s),
                            ('update_s', update_s)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        if not (0 <= w_min < w_max and math.isfinite(w_max)):
            raise ValueError('Homeostasis requires finite nonnegative weight bounds')
        self.update_steps = round(update_s * 1000 / dt_ms)
        if self.update_steps < 1 or not math.isclose(
                self.update_steps * dt_ms / 1000, update_s, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Homeostatic update_s must be an integer number of timesteps')
        self.target_hz = float(target_hz)
        self.update_s = float(update_s)
        self.scaling_tau_s = float(scaling_tau_s)
        self.rate_alpha = -math.expm1(-update_s / rate_tau_s)
        # Starting at the target avoids mistaking initialization for prolonged silence.
        self.rate_hz = np.full(n_cells, target_hz, dtype=float)
        self.spike_counts = np.zeros(n_cells, dtype=np.int64)
        self.bin_steps = 0
        self.active_steps = 0
        self.w_min, self.w_max = w_min, w_max
        self.cumulative_log_factor = np.zeros(n_cells)
        self.cumulative_weight_change = np.zeros(n_cells)
        self.clipped_synapse_updates = np.zeros(n_cells, dtype=np.int64)
        self.n_updates = 0

    def step(self, spikes, weights, plasticity_on=True):
        """Sense activity during burn-in too, but scale only during active learning.

All counters, incomplete bins and sensor state live on this object and are
included in the pipeline's complete-state checkpoints.
"""
        self.spike_counts += spikes
        self.bin_steps += 1
        if plasticity_on:
            self.active_steps += 1
        if self.bin_steps < self.update_steps:
            return
        observed_hz = self.spike_counts / self.update_s
        self.rate_hz += self.rate_alpha * (observed_hz - self.rate_hz)
        if self.active_steps:
            elapsed_s = self.update_s * self.active_steps / self.update_steps
            log_factor = elapsed_s / self.scaling_tau_s * (1 - self.rate_hz / self.target_hz)
            before = weights.sum(axis=1)
            weights *= np.exp(log_factor)[:, None]
            self.clipped_synapse_updates += ((weights < self.w_min) | (weights > self.w_max)).sum(axis=1)
            np.clip(weights, self.w_min, self.w_max, out=weights)
            self.cumulative_weight_change += weights.sum(axis=1) - before
            self.cumulative_log_factor += log_factor
            self.n_updates += 1
        self.spike_counts.fill(0)
        self.bin_steps = self.active_steps = 0
