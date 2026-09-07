"""Connectivity, ported from CbmSim's microzone.

Connection counts are CbmSim's own (`src/cbm_state/connectivityparams.cpp`) and
the wiring algorithms follow `MZoneConnectivityState` rather than being
re-derived. What is scaled is the three POPULATION sizes, 10x; what is not
scaled is per-cell convergence, which is the quantity that sets each cell's
synaptic load and therefore its operating point:

    CbmSim                          here          meaning
    num_pc            32            n_pkj  = 320  Purkinje cells          (x10)
    num_nc             8            n_dcn  =  80  deep nuclear cells      (x10)
    num_io             4            n_io   =  40  inferior olive          (x10)
    num_p_nc_from_pc_to_nc  12      n_pkj_per_dcn = 12   PKJ converging on one DCN
    num_p_pc_from_pc_to_nc   3      n_dcn_per_pkj =  3   DCN reached by one PKJ
    num_p_io_from_nc_to_io   8      n_dcn_per_io  =  8   DCN converging on one IO
    num_p_io_from_io_to_pc   8      n_pkj_per_io  =  8   PKJ contacted by one CF
    num_p_io_in_io_to_io     3      (gap matrix)  =  4   IO-IO coupling partners
    num_p_gr_to_pc       32768      n_pf_per_pkj  = 500  PF onto one PKJ -- SCALED DOWN

The only quantity not taken literally is the granule input: CbmSim gives each
Purkinje cell 32,768 of a million shared granule cells, and this MVP gives it a
private pool of `n_pf_per_pkj` (500) Poisson fibers instead. That is the spec's
own simplification -- private pools keep each Purkinje cell's coincidence
detection independent, which is what H2 is about -- and it is the one place this
module departs from the reference.

The counts stay mutually consistent at this scale (320 PKJ x 3 targets = 960 =
80 DCN x 12 inputs), and 40 < 80 < 320 satisfies the white paper's
N_IO << N_DCN << N_PKJ ordering without needing any assumption the paper does
not state.

**DCN->IO stops being complete, and that is the important consequence.** At
CbmSim's own scale, 8 nuclear cells converging on each olivary cell IS the whole
nucleus: `connectNCtoIO` is complete bipartite (`pIOfromNCtoIO[i][j] = j`), so
every olivary cell sees identical inhibition. Combined with identical parameters
and no noise, that made the four cells integrate to bit-identical traces -- one
cell computed four times, with the gap junctions acting on a difference that was
identically zero. Holding `n_dcn_per_io` at 8 while the nucleus grows to 80
makes the projection topographic instead (`_connect_dcn_to_io` below): each
olivary cell reads a contiguous block of the nucleus whose start advances with
its index, so neighbours share 6 of 8 inputs and distant cells share none. The
cells now diverge on their own; `tests/test_network_graph.py` asserts all n_io
inhibition patterns are distinct so the degeneracy cannot return unnoticed.

It also makes `enforce_closed_loop` meaningful. Rotating which block of the
nucleus an olivary cell reads misroutes the feedback without cutting DCN->IO
outright -- at CbmSim's complete projection there was nothing to rotate, and
build_connectivity says so in `meta["loop"]`.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Connectivity:
    """Explicit connection tables, in CbmSim's index-list style.

    `pkj_to_dcn[p]` lists the DCN that Purkinje cell p drives; `dcn_to_io[d]`
    lists the IO that nuclear cell d inhibits; `io_to_pkj[i]` lists the Purkinje
    cells olivary cell i's climbing fiber contacts. `cf_of_pkj[p]` is the inverse
    of the last one -- exactly one climbing fiber per Purkinje cell.
    """
    n_io: int
    n_dcn: int
    n_pkj: int
    pkj_to_dcn: list                        # n_pkj lists of DCN indices
    dcn_to_io: list                         # n_dcn lists of IO indices
    io_to_pkj: list                         # n_io lists of PKJ indices
    cf_of_pkj: np.ndarray                   # (n_pkj,) which IO's CF owns each PKJ
    meta: dict = field(default_factory=dict)

    def pkj_to_dcn_matrix(self):
        """(n_dcn, n_pkj) 0/1 -- row d picks out the Purkinje cells driving DCN d."""
        m = np.zeros((self.n_dcn, self.n_pkj))
        for p, targets in enumerate(self.pkj_to_dcn):
            m[list(targets), p] = 1.0
        return m

    def dcn_to_io_matrix(self):
        """(n_io, n_dcn) 0/1 -- row i picks out the nuclear cells inhibiting IO i."""
        m = np.zeros((self.n_io, self.n_dcn))
        for d, targets in enumerate(self.dcn_to_io):
            m[list(targets), d] = 1.0
        return m

    # --- convergence/divergence actually realized, for reporting and tests ---

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
        def rng(a):
            lo, hi = int(a.min()), int(a.max())
            return f"{lo}" if lo == hi else f"{lo}-{hi}"
        loop = self.meta.get("loop", "closed")
        return (f"{self.n_io} IO, {self.n_dcn} DCN, {self.n_pkj} PKJ; "
                f"each CF -> {rng(self.pkj_per_io)} PKJ, each PKJ -> {rng(self.dcn_per_pkj)} DCN, "
                f"each DCN <- {rng(self.pkj_per_dcn)} PKJ and -> {rng(self.io_per_dcn)} IO, "
                f"each IO <- {rng(self.dcn_per_io)} DCN; DCN->IO {loop}")


def _connect_pkj_to_dcn(n_pkj, n_dcn, n_pkj_per_dcn, n_dcn_per_pkj, rng):
    """PKJ -> DCN, after CbmSim's `MZoneConnectivityState::connectPCtoNC`.

    CbmSim's structure is a topographic backbone plus a randomized overlap: each
    nuclear cell first takes a contiguous block of `n_pkj / n_dcn` Purkinje cells
    (so every Purkinje cell has one guaranteed target and the projection stays
    ordered), and the remaining slots are filled at random from Purkinje cells
    that have not yet reached their `n_dcn_per_pkj` cap and are not already
    connected to that nuclear cell.

    Two departures, both because CbmSim's version does not close cleanly. Its
    retry loop gives up after ~100 draws and writes whatever candidate it last
    held, which can duplicate a synapse; and its mop-up dumps every remaining
    Purkinje cell onto the *last* nuclear cell without checking for duplicates,
    leaving that one cell lopsided. Here the random fill is deficit-driven --
    each pass serves whichever nuclear cell is furthest below its quota, drawing
    from the Purkinje cells with the most capacity left -- which is the same
    "random overlap on a topographic base" but closes on the configured counts
    exactly. That matters because the capacity is tight: n_pkj * n_dcn_per_pkj
    equals n_dcn * n_pkj_per_dcn with nothing to spare, so a greedy pass can
    strand a nuclear cell that a balanced one fills.
    """
    per_dcn = [[] for _ in range(n_dcn)]
    targets_of_pkj = [[] for _ in range(n_pkj)]

    block = n_pkj // n_dcn                                    # deterministic topographic base
    for d in range(n_dcn):
        for j in range(block):
            p = d * block + j
            per_dcn[d].append(p)
            targets_of_pkj[p].append(d)

    while True:
        deficits = [n_pkj_per_dcn - len(per_dcn[d]) for d in range(n_dcn)]
        if max(deficits) <= 0:
            break
        d = int(np.argmax(deficits))                          # serve the neediest nuclear cell first
        eligible = [q for q in range(n_pkj)
                    if len(targets_of_pkj[q]) < n_dcn_per_pkj and d not in targets_of_pkj[q]]
        if not eligible:
            break                                              # capacity exhausted; counts reported as realized
        room = max(n_dcn_per_pkj - len(targets_of_pkj[q]) for q in eligible)
        roomiest = [q for q in eligible if n_dcn_per_pkj - len(targets_of_pkj[q]) == room]
        p = int(rng.choice(roomiest))                          # random among the least-committed
        per_dcn[d].append(p)
        targets_of_pkj[p].append(d)

    return targets_of_pkj


def _connect_dcn_to_io(n_dcn, n_io, n_dcn_per_io, enforce_closed_loop):
    """DCN -> IO, generalizing CbmSim's `connectNCtoIO`.

    CbmSim's version is complete bipartite, which is what `n_dcn_per_io >= n_dcn`
    reproduces exactly. The shipped configuration has `n_dcn_per_io` (8) well
    below `n_dcn` (80), so each olivary cell instead reads a contiguous
    topographic block of the nucleus. `enforce_closed_loop=False` rotates which
    block, which is the only way this projection can be opened short of cutting
    it (see `SimConfig.ablate_dcn_io` for the outright cut). The module
    docstring explains why the incomplete case matters."""
    targets_of_dcn = [[] for _ in range(n_dcn)]
    complete = n_dcn_per_io >= n_dcn
    for i in range(n_io):
        if complete:
            block = range(n_dcn)                      # CbmSim's case: everyone inhibits everyone
        else:
            # Topographic: olivary cell i reads a contiguous block of nuclear cells whose start
            # advances with i, so neighbouring cells share most of their inhibition and distant
            # ones share none. Starts are spread over the whole nucleus (stride n_dcn/n_io), which
            # keeps every DCN projecting somewhere. Offsetting by a full block width misroutes the
            # feedback -- an olivary cell is then inhibited by nuclear cells its own CF does not
            # drive, which is what enforce_closed_loop=False is for.
            start = (i * n_dcn) // n_io + (0 if enforce_closed_loop else n_dcn_per_io)
            block = ((start + j) % n_dcn for j in range(n_dcn_per_io))
        for d in block:
            targets_of_dcn[d].append(i)
    return targets_of_dcn


def _connect_io_to_pkj(n_io, n_pkj_per_io):
    """CbmSim's `connectIOtoPC`: olivary cell i's climbing fiber takes the
    contiguous block of Purkinje cells [i*n, (i+1)*n). One CF per PKJ."""
    return [list(range(i * n_pkj_per_io, (i + 1) * n_pkj_per_io)) for i in range(n_io)]


def build_connectivity(n_io, n_dcn, n_pkj_per_io, n_pkj_per_dcn, n_dcn_per_pkj,
                       n_dcn_per_io, enforce_closed_loop=True, seed=0):
    n_pkj = n_io * n_pkj_per_io                               # one climbing fiber per Purkinje cell
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
    cf_of_pkj = np.repeat(np.arange(n_io), n_pkj_per_io)

    complete = n_dcn_per_io >= n_dcn
    loop = "closed" if enforce_closed_loop else ("rotated (open)" if not complete else
                                                  "closed -- complete projection, nothing to rotate")
    return Connectivity(n_io=n_io, n_dcn=n_dcn, n_pkj=n_pkj,
                        pkj_to_dcn=pkj_to_dcn, dcn_to_io=dcn_to_io, io_to_pkj=io_to_pkj,
                        cf_of_pkj=cf_of_pkj,
                        meta={"loop": loop, "dcn_to_io_complete": complete, "seed": seed})
