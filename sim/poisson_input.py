"""Parallel-fibre (PF) input: random spikes at a fixed rate.

Each Purkinje cell has its own private set of n_pf_per_pkj fibres (500), so all
of the input is one True/False array of shape (n_pkj, n_pf_per_pkj), the same
shape as the weight matrix, drawn fresh every time step. True means that fibre
spiked this step.

(In the real cerebellum PFs are shared between Purkinje cells. Private fibres
are a simplification.)
"""
import numpy as np


def generate_pf_spikes(rate_hz, dt_ms, rng, shape):
    """Return a True/False array of `shape`: which fibres spike this step.

    Each fibre spikes independently with probability rate x dt (20 Hz x 1 ms
    = 0.02). That is the standard discrete-time version of Poisson firing."""
    p = np.clip(rate_hz * dt_ms / 1000.0, 0.0, 1.0)   # spike probability per step (rate in Hz, dt in ms)
    return rng.random(shape) < p
