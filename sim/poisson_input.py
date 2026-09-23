"""PF (parallel fiber) Poisson spike source.

Each Purkinje cell owns a private pool of `n_pf_per_pkj` fibers, so the whole
input layer is one (n_pkj, n_pf_per_pkj) boolean array redrawn every timestep.
Private pools are the spec's own simplification of CbmSim's shared granule
population: they keep each Purkinje cell's coincidence detection independent of
every other cell's, which is the thing H2 measures.
"""
import numpy as np
from .native import bernoulli


def generate_pf_spikes(rate_hz, dt_ms, rng, shape):
    """One Bernoulli draw per fiber per timestep -- the discrete-time
    approximation of a Poisson process, exact in the limit rate*dt << 1 (at the
    default 20 Hz and 1 ms that is p = 0.02, so the two agree to well under a
    percent). `shape` is the weight matrix's shape, (n_pkj, n_pf_per_pkj)."""
    p = np.clip(rate_hz * dt_ms / 1000.0, 0.0, 1.0)   # per-bin spike probability; clipped in case rate*dt exceeds 1
    if bernoulli is not None and shape is not None and isinstance(rng, np.random.Generator) and type(rng.bit_generator) is np.random.PCG64:
        with rng.bit_generator.lock:
            state = rng.bit_generator.state
            value, inc = state['state']['state'], state['state']['inc']
            spikes, high, low = bernoulli(shape, float(p), value >> 64, value & ((1 << 64) - 1), inc >> 64, inc & ((1 << 64) - 1))
            state['state']['state'] = (high << 64) | low
            rng.bit_generator.state = state
        return spikes[()] if spikes.ndim == 0 else spikes
    return rng.random(shape) < p
