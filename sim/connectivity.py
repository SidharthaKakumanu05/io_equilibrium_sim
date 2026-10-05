"""Builds the wiring: which cell connects to which.

Three projections are built here. (PF -> PKJ needs no table, since each Purkinje
cell simply has its own N_PF_PER_PKJ fibres.)

  IO  -> PKJ   climbing fibres. Olive cell i contacts Purkinje cells
               8i .. 8i+7, so each PKJ has exactly one climbing fibre.
  PKJ -> DCN   each DCN cell gets a fixed block of neighbouring PKJ (4 of them by
               default), then random extra PKJ until it has 12 inputs. Each
               PKJ ends up driving exactly 3 DCN.
  DCN -> IO    each olive cell gets a contiguous block of 8 DCN cells, and the
               block's start moves along the nucleus with the olive cell's
               index. Neighbouring olive cells share most of their inhibition;
               distant ones share none.

The numbers follow CbmSim, the reference cerebellum simulator, with the
population sizes scaled up 10x (4/8/32 -> 40/80/320) and the per-cell
connection counts kept the same.

Because the PKJ->DCN and DCN->IO projections overlap, a DCN cell receives Purkinje
cells from several climbing fibres. So the loop is not 40 separate small loops:
the territories are coupled through the nucleus.

The result is a Connectivity object holding the tables as lists of indices,
plus methods that turn them into the 0/1 matrices sim/simulate.py multiplies
spike vectors by.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Connectivity:
    """The connection tables. Each is a list with one entry per presynaptic
    cell, holding the indices of the cells it connects to:

      pkj_to_dcn[p]  the DCN cells Purkinje cell p inhibits
      dcn_to_io[d]   the olive cells DCN cell d inhibits
      io_to_pkj[i]   the Purkinje cells olive cell i's climbing fibre contacts
      cf_of_pkj[p]   the reverse of io_to_pkj: the one olive cell contacting PKJ p
    """
    n_io: int
    n_dcn: int
    n_pkj: int
    pkj_to_dcn: list                        # n_pkj lists of DCN indices
    dcn_to_io: list                         # n_dcn lists of IO indices
    io_to_pkj: list                         # n_io lists of PKJ indices
    cf_of_pkj: np.ndarray                   # (n_pkj,) which IO's CF owns each PKJ
    meta: dict = field(default_factory=dict)  # notes on how it was built, e.g. meta["loop"]

    def pkj_to_dcn_matrix(self):
        """(n_dcn, n_pkj) matrix with a 1 where PKJ p connects to DCN d.
        (matrix @ pkj_spikes) gives the number of input spikes each DCN received."""
        m = np.zeros((self.n_dcn, self.n_pkj))
        for p, targets in enumerate(self.pkj_to_dcn):
            m[list(targets), p] = 1.0
        return m

    def dcn_to_io_matrix(self):
        """(n_io, n_dcn) matrix with a 1 where DCN d connects to olive cell i."""
        m = np.zeros((self.n_io, self.n_dcn))
        for d, targets in enumerate(self.dcn_to_io):
            m[list(targets), d] = 1.0
        return m

    # --- counts per cell, as actually built (simulate.py uses pkj_per_dcn and dcn_per_io) ---

    @property
    def pkj_per_dcn(self):
        return self.pkj_to_dcn_matrix().sum(axis=1)

    @property
    def dcn_per_pkj(self):
        return np.array([len(t) for t in self.pkj_to_dcn], dtype=float)

    @property
    def io_per_dcn(self):
        return np.array([len(t) for t in self.dcn_to_io], dtype=float)

    @property
    def dcn_per_io(self):
        return self.dcn_to_io_matrix().sum(axis=1)

    @property
    def pkj_per_io(self):
        return np.array([len(t) for t in self.io_to_pkj], dtype=float)

    def describe(self):
        """One-line text summary of the wiring. Not called by run.py."""
        def rng(a):
            lo, hi = int(a.min()), int(a.max())
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
        loop = self.meta.get("loop", "closed")
        return (f"{self.n_io} IO, {self.n_dcn} DCN, {self.n_pkj} PKJ; "
                f"each CF -> {rng(self.pkj_per_io)} PKJ, each PKJ -> {rng(self.dcn_per_pkj)} DCN, "
                f"each DCN <- {rng(self.pkj_per_dcn)} PKJ and -> {rng(self.io_per_dcn)} IO, "
                f"each IO <- {rng(self.dcn_per_io)} DCN; DCN->IO {loop}")


def _connect_pkj_to_dcn(n_pkj, n_dcn, n_pkj_per_dcn, n_dcn_per_pkj, rng):
    """Wire PKJ -> DCN. Returns, for each PKJ, the list of DCN it inhibits.

    Two stages:
      1. Fixed part: DCN d takes the block of n_pkj/n_dcn neighbouring PKJ
         starting at d*block (4 PKJ with the defaults). Every PKJ gets one target.
      2. Random part: fill the remaining input slots of each DCN with random
         PKJ, never connecting the same pair twice and never giving a PKJ more
         than n_dcn_per_pkj targets.

    The random fill always serves the DCN that is furthest from its 12 inputs,
    and picks among the PKJ with the most free slots.

    Near the end it can get stuck: the last DCN needs an input, but the only PKJ
    with a free slot is already connected to it. _swap_to_fill() then fixes
    that by trading partners with another DCN (see there), so every DCN ends with
    exactly n_pkj_per_dcn inputs and every PKJ with exactly n_dcn_per_pkj targets.
    Seeds that never get stuck are wired exactly as if the swap did not exist.
    """
    per_dcn = [[] for _ in range(n_dcn)]
    targets_of_pkj = [[] for _ in range(n_pkj)]

    # --- stage 1: fixed block of neighbouring PKJ for each DCN ---
    block = n_pkj // n_dcn
    for d in range(n_dcn):
        for j in range(block):
            p = d * block + j
            per_dcn[d].append(p)
            targets_of_pkj[p].append(d)

    # --- stage 2: random extra inputs until every DCN has n_pkj_per_dcn ---
    while True:
        deficits = [n_pkj_per_dcn - len(per_dcn[d]) for d in range(n_dcn)]
        if max(deficits) <= 0:
            break
        d = int(np.argmax(deficits))                          # the DCN missing the most inputs
        eligible = [q for q in range(n_pkj)
                    if len(targets_of_pkj[q]) < n_dcn_per_pkj and d not in targets_of_pkj[q]]   # PKJ with a free slot, not already on d
        if not eligible:
            _swap_to_fill(d, per_dcn, targets_of_pkj, block, n_dcn_per_pkj, rng)   # stuck: trade partners
            continue
        room = max(n_dcn_per_pkj - len(targets_of_pkj[q]) for q in eligible)
        roomiest = [q for q in eligible if n_dcn_per_pkj - len(targets_of_pkj[q]) == room]
        p = int(rng.choice(roomiest))                          # random pick among the PKJ with most free slots
        per_dcn[d].append(p)
        targets_of_pkj[p].append(d)

    return targets_of_pkj


def _swap_to_fill(d, per_dcn, targets_of_pkj, block, n_dcn_per_pkj, rng):
    """Give DCN d one more input when no PKJ can be connected to it directly.

    The situation: some PKJ q still has a free slot, but q is already on d.
    The fix: pick another DCN d2 with a random-stage input p2, where p2 is not
    on d and q is not on d2. Then move p2 from d2 to d, and give d2 q instead:

        before:  d2 <- p2          d <- (one short)     q (one slot free)
        after:   d2 <- q           d <- p2              q (slot used)

    d gains one input, d2 keeps the same number, p2 keeps the same number of
    targets, and q uses up its free slot. The fixed (topographic) inputs are
    never moved. Lists are edited in place.
    """
    spare = [q for q in range(len(targets_of_pkj)) if len(targets_of_pkj[q]) < n_dcn_per_pkj]
    candidates = []                                            # every valid (q, d2, p2) trade
    for q in spare:
        for d2 in range(len(per_dcn)):
            if d2 == d or d2 in targets_of_pkj[q]:
                continue
            for p2 in per_dcn[d2]:
                if p2 // block != d2 and d not in targets_of_pkj[p2]:   # p2 is a random input, and not already on d
                    candidates.append((q, d2, p2))
    if not candidates:
        raise RuntimeError(f"PKJ->DCN wiring cannot be completed: no partner swap fills DCN {d}")
    q, d2, p2 = candidates[int(rng.integers(len(candidates)))]

    per_dcn[d2].remove(p2); targets_of_pkj[p2].remove(d2)     # unhook p2 from d2
    per_dcn[d2].append(q);  targets_of_pkj[q].append(d2)      # d2 takes q instead
    per_dcn[d].append(p2);  targets_of_pkj[p2].append(d)      # d takes p2


def _connect_dcn_to_io(n_dcn, n_io, n_dcn_per_io, enforce_closed_loop):
    """Wire DCN -> IO. Returns, for each DCN cell, the list of olive cells it inhibits.

    Olive cell i receives a contiguous block of n_dcn_per_io DCN cells, starting
    at i * n_dcn / n_io (DCN 0-7 for olive 0, DCN 2-9 for olive 1, and so on,
    wrapping around at the end).

    enforce_closed_loop=False shifts every block along by one block width, so
    each olive cell is inhibited by the wrong DCN cells (the "misrouted" variant).
    If n_dcn_per_io >= n_dcn, every DCN inhibits every olive cell."""
    targets_of_dcn = [[] for _ in range(n_dcn)]
    complete = n_dcn_per_io >= n_dcn
    for i in range(n_io):
        if complete:
            block = range(n_dcn)                      # every DCN inhibits every olive cell
        else:
            # Block start for olive cell i, shifted by one block width if the loop is misrouted.
            start = (i * n_dcn) // n_io + (0 if enforce_closed_loop else n_dcn_per_io)
            block = ((start + j) % n_dcn for j in range(n_dcn_per_io))   # % n_dcn wraps past the last DCN
        for d in block:
            targets_of_dcn[d].append(i)
    return targets_of_dcn


def _connect_io_to_pkj(n_io, n_pkj_per_io):
    """Wire IO -> PKJ (climbing fibres). Olive cell i contacts Purkinje cells
    i*n .. (i+1)*n - 1, so every PKJ has exactly one climbing fibre."""
    return [list(range(i * n_pkj_per_io, (i + 1) * n_pkj_per_io)) for i in range(n_io)]


def build_connectivity(n_io, n_dcn, n_pkj_per_io, n_pkj_per_dcn, n_dcn_per_pkj,
                       n_dcn_per_io, enforce_closed_loop=True, seed=0):
    """Check the counts are consistent, build all three projections, and return
    them as a Connectivity. `seed` only affects the random part of PKJ -> DCN."""
    n_pkj = n_io * n_pkj_per_io                               # one climbing fibre per Purkinje cell
    if n_pkj % n_dcn:
        raise ValueError(f"n_pkj ({n_pkj}) must be a multiple of n_dcn ({n_dcn}) for the topographic base")
    if n_dcn_per_io > n_dcn:
        raise ValueError(f"n_dcn_per_io ({n_dcn_per_io}) cannot exceed n_dcn ({n_dcn})")
    if n_pkj_per_dcn * n_dcn != n_dcn_per_pkj * n_pkj:
        raise ValueError(f"PKJ->DCN counts disagree: {n_dcn} DCN x {n_pkj_per_dcn} inputs "
                         f"!= {n_pkj} PKJ x {n_dcn_per_pkj} targets")

    rng = np.random.default_rng(seed)
    pkj_to_dcn = _connect_pkj_to_dcn(n_pkj, n_dcn, n_pkj_per_dcn, n_dcn_per_pkj, rng)
    dcn_to_io = _connect_dcn_to_io(n_dcn, n_io, n_dcn_per_io, enforce_closed_loop)
    io_to_pkj = _connect_io_to_pkj(n_io, n_pkj_per_io)
    cf_of_pkj = np.repeat(np.arange(n_io), n_pkj_per_io)     # [0,0,...,0, 1,1,...] -- 8 of each

    # Final check that PKJ -> DCN came out exactly as asked: right counts, no duplicate synapses.
    n_inputs = np.bincount([d for targets in pkj_to_dcn for d in targets], minlength=n_dcn)
    if (any(len(t) != n_dcn_per_pkj or len(set(t)) != len(t) for t in pkj_to_dcn)
            or (n_inputs != n_pkj_per_dcn).any()):
        raise RuntimeError("PKJ->DCN wiring came out with the wrong counts -- this is a bug in _connect_pkj_to_dcn")

    # A text label for how DCN -> IO was wired, stored in meta["loop"].
    complete = n_dcn_per_io >= n_dcn
    loop = "closed" if enforce_closed_loop else ("rotated (open)" if not complete else
                                                  "closed -- complete projection, nothing to rotate")
    return Connectivity(n_io=n_io, n_dcn=n_dcn, n_pkj=n_pkj,
                        pkj_to_dcn=pkj_to_dcn, dcn_to_io=dcn_to_io, io_to_pkj=io_to_pkj,
                        cf_of_pkj=cf_of_pkj,
                        meta={"loop": loop, "dcn_to_io_complete": complete, "seed": seed})
