"""Gap junctions between olive cells.

Neighbouring olive cells are joined by gap junctions, direct electrical
connections that let current flow from the higher-voltage cell to the lower one.
That pulls their voltages together and tends to make them fire in sync.

Each junction acts as a resistor between two cells. The current into cell i is

    I_gap(i) = sum over j of  g_gap[i, j] * (V_j - V_i)

so everything about the coupling is in one matrix, g_gap: entry [i, j] is the
conductance between cells i and j (0 if they aren't coupled). It is symmetric,
since current flows equally both ways, and zero on the diagonal.

This file only BUILDS that matrix. The current itself is computed inside the
olive cell's voltage update (sim/io_channels.py, IOPopulation.substep).

Also here: coupling_summary() and synchrony_index(). run.py calls neither.
"""
import numpy as np

TOPOLOGIES = ("all_to_all", "ring", "nearest_k", "small_world")   # the accepted values of `topology`


def build_gap_junction_matrix(n_io, g_gap, topology="all_to_all", n_neighbors=2,
                              rewire_p=0.15, seed=0):
    """Build the (n_io, n_io) gap-junction matrix, in mS/cm^2.

    g_gap is the conductance of ONE junction. A cell's total coupling is g_gap
    times its number of partners, and that total is what matters. Compare it
    with the cell's leak (0.06 mS/cm^2): the default is 4 x 0.0143 = 0.057.

    The cells are imagined sitting on a ring, numbered 0..n_io-1. Topologies:
      all_to_all   every cell coupled to every other cell
      ring         each cell coupled to its 2 immediate neighbours
      nearest_k    each cell coupled to the n_neighbors nearest cells on each
                   side (2 x n_neighbors partners). The default.
      small_world  nearest_k, then a fraction rewire_p of junctions moved to
                   random partners (seeded by `seed`), giving a few long-range
                   shortcuts
    """
    n = int(n_io)
    if n < 1:
        raise ValueError(f"n_io must be >= 1, got {n}")
    g = np.zeros((n, n), dtype=float)
    if n < 2 or g_gap == 0.0:
        return g                                              # coupling off: all zeros

    if topology == "all_to_all":
        g[:] = g_gap
        np.fill_diagonal(g, 0.0)
    elif topology in ("ring", "nearest_k"):
        k = 1 if topology == "ring" else int(n_neighbors)
        if k < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        k = min(k, n // 2)                                     # can't have more neighbours per side than half the ring
        # Couple each cell i to i+1, i+2, ..., i+k (wrapping around). Since the matrix is filled
        # symmetrically, that also couples i to i-1 ... i-k.
        for offset in range(1, k + 1):
            idx = np.arange(n)
            partner = (idx + offset) % n
            g[idx, partner] = g_gap                            # set both [i, j] and [j, i]
            g[partner, idx] = g_gap
        np.fill_diagonal(g, 0.0)
    elif topology == "small_world":
        k = max(1, min(int(n_neighbors), n // 2))
        rng = np.random.default_rng(seed)
        # Start from the nearest_k pairs, stored as (smaller index, larger index).
        edges = set()
        for offset in range(1, k + 1):
            for i in range(n):
                a, b = i, (i + offset) % n
                if a != b:
                    edges.add((min(a, b), max(a, b)))
        # With probability rewire_p, move one end of each junction to a random cell
        # (never onto itself, never duplicating an existing junction).
        for (a, b) in sorted(edges):
            if rng.random() >= rewire_p:
                continue
            for _ in range(50):
                c = int(rng.integers(n))
                new = (min(a, c), max(a, c))
                if c != a and new not in edges:
                    edges.discard((a, b))
                    edges.add(new)
                    break
        for (a, b) in edges:                                   # write the final pairs into the matrix
            g[a, b] = g_gap
            g[b, a] = g_gap
        np.fill_diagonal(g, 0.0)
    else:
        raise ValueError(f"unknown gap-junction topology {topology!r} (expected one of {TOPOLOGIES})")
    return g


def coupling_summary(g_gap_matrix, g_leak=None):
    """One-line text description of a gap-junction matrix. Not called by run.py."""
    g = np.asarray(g_gap_matrix, dtype=float)
    n = g.shape[0]
    n_edges = int((g > 0).sum() // 2)
    degree = (g > 0).sum(axis=1)
    total = g.sum(axis=1)
    parts = [f"{n} IO cells, {n_edges} gap junctions",
             f"degree {degree.min()}-{degree.max()}",
             f"total coupling conductance {total.min():.4f}-{total.max():.4f} mS/cm^2"]
    if g_leak:
        parts.append(f"({total.mean() / g_leak:.2f}x leak)")
    return ", ".join(parts)


def synchrony_index(spike_times_by_cell, t_start_ms, t_end_ms, bin_ms=20.0):
    """How synchronised the olive's CF events are. Not called by run.py.

    Counts each cell's spikes in bins of bin_ms, then returns the average
    correlation between every pair of cells: 0 = independent, 1 = identical.
    Cells that never fire in the window are left out."""
    edges = np.arange(t_start_ms, t_end_ms + bin_ms, bin_ms)
    if len(edges) < 3:
        return float("nan")
    counts = np.array([np.histogram(np.asarray(s, dtype=float), bins=edges)[0]
                       for s in spike_times_by_cell], dtype=float)
    active = counts.std(axis=1) > 0                            # drop cells with no variation (e.g. silent)
    counts = counts[active]
    if len(counts) < 2:
        return float("nan")
    c = np.corrcoef(counts)
    off_diagonal = ~np.eye(len(c), dtype=bool)                 # exclude each cell's correlation with itself
    return float(c[off_diagonal].mean())
