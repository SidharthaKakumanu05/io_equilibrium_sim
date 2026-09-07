"""PF (parallel fiber) Poisson spike source.

Each Purkinje cell owns a private pool of `n_pf_per_pkj` fibers, so the whole
input layer is one (n_pkj, n_pf_per_pkj) boolean array redrawn every timestep.
Private pools are the spec's own simplification of CbmSim's shared granule
population: they keep each Purkinje cell's coincidence detection independent of
every other cell's, which is the thing H2 measures.
"""
import numpy as np


def generate_pf_spikes(rate_hz, dt_ms, rng, shape):
    """One Bernoulli draw per fiber per timestep -- the discrete-time
    approximation of a Poisson process, exact in the limit rate*dt << 1 (at the
    default 20 Hz and 1 ms that is p = 0.02, so the two agree to well under a
    percent). `shape` is the weight matrix's shape, (n_pkj, n_pf_per_pkj)."""
    p = np.clip(rate_hz * dt_ms / 1000.0, 0.0, 1.0)   # per-bin spike probability; clipped in case rate*dt exceeds 1
    return rng.random(shape) < p
