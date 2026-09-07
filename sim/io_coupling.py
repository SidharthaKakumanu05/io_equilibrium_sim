"""IO-IO gap-junction (electrical) coupling.

The olive is the most densely electrically coupled nucleus in the mammalian
brain: neighbouring IO neurons are joined by connexin-36 gap junctions inside
the glomeruli, which is what lets their subthreshold oscillations phase-lock
and what makes CF output arrive in synchronous ensembles rather than as
independent single-cell events.

Model: a gap junction is an ohmic resistor between two somata, so the current
into cell i is

    I_gap(i) = sum_j g_gap[i, j] * (V_j - V_i)

with g_gap symmetric (a resistor conducts equally both ways) and zero on the
diagonal. This is the same rule CbmSim applies to its own IO array --
`vCoupleIO[i] += coupleRiRjRatioIO * (vIO[j] - vIO[i])` summed over each cell's
partners in `MZone::updateIOOut`, with `connectIOtoIO` wiring every cell to
every other -- so `all_to_all` here is the direct analogue of CbmSim's default,
and `g_gap` plays the role of `coupleRiRjRatioIO`. Two differences, both
deliberate:

  * CbmSim adds the coupling as a voltage increment in arbitrary units, on top
    of a leaky-integrate-and-fire IO. Here it is a real conductance in mS/cm^2
    entering the conductance-based cell's current balance, so it shunts as well
    as pulls -- which is what a gap junction physically does.
  * CbmSim computes the coupling term in `updateIOOut`, i.e. from the *previous*
    timestep's voltages. Here it is evaluated inside the 0.1 ms sub-timestep,
    at the same instant as every other current. The Ca2+ spike upstroke moves V
    by tens of mV in a millisecond, so a 1 ms lag on the coupling term would
    materially mis-state the current flowing during exactly the event the
    coupling is supposed to synchronize.

The conductance splits exactly into a term for the cell's own voltage and a
term for its neighbours',

    I_gap(i) = (sum_j g_gap[i,j] * V_j)  -  (sum_j g_gap[i,j]) * V_i

so it folds into the exponential-Euler update in sim/io_channels.py with no
separate explicit term: the row sum joins g_tot, and g_gap @ V joins the
numerator. That is why the cells have to be advanced as a population --
sim/io_channels.py IOPopulation -- rather than one at a time.
"""
import numpy as np

TOPOLOGIES = ("all_to_all", "ring", "nearest_k", "small_world")


def build_gap_junction_matrix(n_io, g_gap, topology="all_to_all", n_neighbors=2,
                              rewire_p=0.15, seed=0):
    """Symmetric (n_io, n_io) coupling-conductance matrix, mS/cm^2, zero diagonal.

    g_gap is the conductance of a *single* junction, so a cell's total coupling
    conductance is g_gap times its number of partners: 2*g_gap on a ring,
    2k*g_gap under nearest_k, (n-1)*g_gap all-to-all. That TOTAL is the
    physiologically meaningful quantity, not g_gap -- compare it against
    IOChannelParams.g_leak (0.06 mS/cm^2). Comparable to leak is the
    physiological regime; far above it the cells are clamped into a single unit,
    which is why all-to-all does not survive the scale-up to 40 cells (39
    partners, ~12x leak) and the shipped topology is local instead.

    Topologies:
      all_to_all  every cell coupled to every other (CbmSim's connectIOtoIO).
      ring        cells on a closed ring, each coupled to its 2 neighbours --
                  local coupling, which is closer to the olive's anatomy and
                  produces patchy/traveling synchrony rather than global lock-step.
      nearest_k   ring generalized to the k nearest neighbours either side
                  (n_neighbors = k, so each cell has 2k partners).
      small_world Watts-Strogatz: start from nearest_k, then rewire a fraction
                  `rewire_p` of the junctions to random partners. Degree and
                  total conductance per cell are preserved on average, so this
                  buys synchrony with topology rather than with conductance.

    Why small_world exists: under every local topology, global synchrony is
    limited by path length rather than by conductance. Measured on the shipped
    40-cell olive at matched total conductance, directly coupled pairs reach
    Vm r ~ +0.90 while pairs on opposite sides plateau near +0.39 -- the local
    junctions are already close to saturation, so raising g_gap has almost
    nothing left to give there. Shortening the paths does what raising the
    conductance cannot: at n_io = 40 and k = 2 the graph diameters are 20 (ring),
    10 (nearest_k), 6-8 (small_world at 15% rewiring, depending on `seed`) and
    1 (all_to_all), and CF
    synchrony follows that order rather than following g_gap. See the README's
    topology table.
    """
    n = int(n_io)
    if n < 1:
        raise ValueError(f"n_io must be >= 1, got {n}")
    g = np.zeros((n, n), dtype=float)
    if n < 2 or g_gap == 0.0:
        return g                                              # a lone cell (or zero conductance) has nothing to couple to

    if topology == "all_to_all":
        g[:] = g_gap
        np.fill_diagonal(g, 0.0)
    elif topology in ("ring", "nearest_k"):
        k = 1 if topology == "ring" else int(n_neighbors)
        if k < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        k = min(k, n // 2)                                     # beyond n//2 the ring wraps onto itself: all_to_all
        for offset in range(1, k + 1):
            idx = np.arange(n)
            partner = (idx + offset) % n
            g[idx, partner] = g_gap                            # both assignments together keep the matrix symmetric,
            g[partner, idx] = g_gap                            # and handle the offset == n/2 case writing the same pair twice
        np.fill_diagonal(g, 0.0)
    elif topology == "small_world":
        k = max(1, min(int(n_neighbors), n // 2))
        rng = np.random.default_rng(seed)
        edges = set()
        for offset in range(1, k + 1):
            for i in range(n):
                a, b = i, (i + offset) % n
                if a != b:
                    edges.add((min(a, b), max(a, b)))
        for (a, b) in sorted(edges):
            if rng.random() >= rewire_p:
                continue
            # Move one end of this junction to a random cell, keeping the graph simple.
            for _ in range(50):
                c = int(rng.integers(n))
                new = (min(a, c), max(a, c))
                if c != a and new not in edges:
                    edges.discard((a, b))
                    edges.add(new)
                    break
        for (a, b) in edges:
            g[a, b] = g_gap
            g[b, a] = g_gap
        np.fill_diagonal(g, 0.0)
    else:
        raise ValueError(f"unknown gap-junction topology {topology!r} (expected one of {TOPOLOGIES})")
    return g


def coupling_summary(g_gap_matrix, g_leak=None):
    """Human-readable description of a coupling matrix, for run logs."""
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
    """Pairwise-correlation measure of how synchronized the CF output is.

    Spike trains are binned, then the mean off-diagonal Pearson correlation
    across cell pairs is returned. 0 = independent cells, 1 = identical trains.

    This is a SPIKE measure, and at ~1 Hz firing it is a blunt one: most bins
    are empty for every cell, so it moves far less than the coupling does. The
    subthreshold Vm correlation in experiments/run_gap_sweep.py is the sensitive
    read-out of what the junctions are doing; this is the downstream consequence
    that actually matters to the cerebellum, so both are reported."""
    edges = np.arange(t_start_ms, t_end_ms + bin_ms, bin_ms)
    if len(edges) < 3:
        return float("nan")
    counts = np.array([np.histogram(np.asarray(s, dtype=float), bins=edges)[0]
                       for s in spike_times_by_cell], dtype=float)
    active = counts.std(axis=1) > 0                            # a silent cell has undefined correlation with anything
    counts = counts[active]
    if len(counts) < 2:
        return float("nan")
    c = np.corrcoef(counts)
    off_diagonal = ~np.eye(len(c), dtype=bool)
    return float(c[off_diagonal].mean())
