"""PF (parallel fiber) Poisson spike source."""
import numpy as np                                     # for rng.random() and array comparison


def generate_pf_spikes(rate_hz, dt_ms, rng, shape):
    p = np.clip(rate_hz * dt_ms / 1000.0, 0.0, 1.0)    # per-bin spike probability, clipped in case rate*dt exceeds 1
    return rng.random(shape) < p                        # one Bernoulli draw per PF unit per PKJ; shape = (n_pkj_per_io, n_pf_per_pkj)
