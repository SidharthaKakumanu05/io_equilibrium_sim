"""The wired network as an explicit graph: nodes with 3D positions, typed
edges, and the adjacency matrix they induce.

Nothing here participates in the simulation. It exists so the connectivity that
sim/simulate.py applies implicitly -- as array slices and index arrays -- can be
read out as one object and drawn (sim/network_viz.py). Because it is built from
the same Connectivity and gap-junction matrix the simulation runs on, the
picture is a picture of the actual network, not a redrawing of the intent.

Node ordering is PF, then PKJ, then DCN, then IO, so the adjacency matrix comes
out block-structured along the pathway and each block is a projection.

Only a subsample of the parallel fibers is included (`n_pf_per_pkj_shown`): the
full network has n_io * n_pkj_per_io * n_pf_per_pkj = 160,000 PF units at the
default scale, which is neither drawable nor a readable adjacency matrix. Every
other cell is present in full.

`group` on a node is its climbing-fiber territory for PF and PKJ, and the cell's
own index for DCN and IO -- this connectivity has no separable groups, since every
nuclear cell inhibits every olivary cell.
"""
from dataclasses import dataclass, field

import numpy as np

# Edge kinds, with the sign of their effect on the postsynaptic cell.
EDGE_KINDS = {
    "pf_pkj":  {"sign": +1, "label": "PF -> PKJ (excitatory, plastic)"},
    "io_pkj":  {"sign": +1, "label": "IO -> PKJ (climbing fiber / teaching)"},
    "pkj_dcn": {"sign": -1, "label": "PKJ -> DCN (inhibitory)"},
    "dcn_io":  {"sign": -1, "label": "DCN -> IO (inhibitory feedback)"},
    "io_io":   {"sign":  0, "label": "IO <-> IO (gap junction, electrical)"},
}


@dataclass
class NetworkGraph:
    """Nodes are indexed 0..n_nodes-1 in PF, PKJ, DCN, IO order."""
    kind: np.ndarray            # (N,) cell-type string per node
    group: np.ndarray           # (N,) which closed-loop group the node belongs to
    index: np.ndarray           # (N,) index within (kind, group)
    position: np.ndarray        # (N, 3) x/y/z coordinates for drawing
    edge_src: np.ndarray        # (E,) source node id
    edge_dst: np.ndarray        # (E,) destination node id
    edge_kind: np.ndarray       # (E,) key into EDGE_KINDS
    edge_weight: np.ndarray     # (E,) synaptic weight or gap conductance
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return len(self.kind)

    @property
    def n_edges(self):
        return len(self.edge_src)

    def node_ids(self, kind):
        return np.flatnonzero(self.kind == kind)

    def adjacency_matrix(self, signed=True, kinds=None, normalize_per_kind=False):
        """(N, N) matrix, A[i, j] = weight of the edge from node i to node j.

        `signed` multiplies each entry by its projection's sign, so excitation
        is positive and inhibition negative; gap junctions are symmetric and
        carry sign 0, so they are written as their raw conductance either way.
        `kinds` restricts the matrix to a subset of projections.

        `normalize_per_kind` divides each projection by its own largest
        magnitude, putting every entry in [-1, 1]. The projections are in
        genuinely different units -- a dimensionless synaptic weight, a
        conductance in 1/ms, a conductance in mS/cm^2 -- so a single scale
        across all of them is not a comparison of anything, and rendering it
        that way simply hides whichever projection has the smallest numbers.
        Use it whenever the matrix is being drawn to show structure.
        """
        a = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        selected = np.ones(self.n_edges, dtype=bool) if kinds is None else np.isin(self.edge_kind, list(kinds))
        scale = {}
        if normalize_per_kind:
            for kind in np.unique(self.edge_kind[selected]):
                m = np.abs(self.edge_weight[selected][self.edge_kind[selected] == kind]).max()
                scale[kind] = m if m > 0 else 1.0
        for src, dst, kind, w in zip(self.edge_src[selected], self.edge_dst[selected],
                                      self.edge_kind[selected], self.edge_weight[selected]):
            sign = EDGE_KINDS[kind]["sign"]
            value = w / scale.get(kind, 1.0) * (sign if (signed and sign != 0) else 1.0)
            a[src, dst] += value
            if sign == 0:
                # Sign 0 marks an undirected projection -- a gap junction is one resistor, stored
                # once per unordered pair in the edge list. Writing only [src, dst] would produce a
                # triangular block and misrepresent coupling that conducts equally both ways.
                a[dst, src] += value
        return a

    def kind_blocks(self):
        """[(kind, start, stop)] boundaries of each cell type in node order --
        what the adjacency-matrix plot draws its block dividers from."""
        blocks, start = [], 0
        for k in ("PF", "PKJ", "DCN", "IO"):
            n = int((self.kind == k).sum())
            if n:
                blocks.append((k, start, start + n))
                start += n
        return blocks

    def summary(self):
        counts = ", ".join(f"{int((self.kind == k).sum())} {k}" for k, _, _ in self.kind_blocks())
        per_kind = ", ".join(f"{int((self.edge_kind == k).sum())} {k}"
                             for k in EDGE_KINDS if (self.edge_kind == k).any())
        return f"{self.n_nodes} nodes ({counts}); {self.n_edges} edges ({per_kind})"


def _ring_positions(n, radius, z, phase=0.0):
    theta = phase + 2.0 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)], axis=1), theta


def build_network_graph(cfg, conn, gap_matrix=None, weights=None, n_pf_per_pkj_shown=8, seed=0):
    """Assemble the graph from the same connection tables the simulation runs on.

    `weights` is an optional (n_pkj, n_pf_per_pkj) array of live PF->PKJ weights;
    without it every PF edge is drawn at cfg.w_init. `gap_matrix` is the
    (n_io, n_io) coupling matrix from sim/io_coupling.py.

    Layout is anatomical in spirit: the olive sits at the bottom as a tight
    cluster (it is one nucleus, electrically coupled across the population), the
    nuclei above it, the Purkinje layer above that, and the parallel fibers on
    top. Purkinje cells fan out by climbing-fiber territory, so each olivary
    cell's eight targets sit together; nuclear cells sit on their own ring,
    because in this connectivity they are shared across the whole microzone
    rather than belonging to any one territory.
    """
    rng = np.random.default_rng(seed)
    n_io, n_dcn, n_pkj = conn.n_io, conn.n_dcn, conn.n_pkj
    n_pf_shown = min(n_pf_per_pkj_shown, cfg.n_pf_per_pkj)

    kinds, groups, indices, positions = [], [], [], []

    def add(kind, group, index, pos):
        kinds.append(kind); groups.append(group); indices.append(index); positions.append(pos)

    Z_IO, Z_DCN, Z_PKJ, Z_PF = 0.0, 3.0, 6.0, 8.2
    R_IO, R_DCN, R_PKJ = 1.6, 5.0, 7.6

    # Purkinje cells are laid out around a full circle, ordered by climbing fiber, so an
    # olivary cell's territory is a contiguous arc.
    pkj_theta = 2.0 * np.pi * np.arange(n_pkj) / max(n_pkj, 1)

    pf_node_of, pkj_node_of, dcn_node_of, io_node_of = {}, {}, {}, {}

    for pkj in range(n_pkj):                                   # PF cloud above each Purkinje cell
        th = pkj_theta[pkj]
        for s_i in range(n_pf_shown):
            r = R_PKJ + 0.35 * rng.normal()
            pos = np.array([r * np.cos(th), r * np.sin(th), Z_PF]) + rng.normal(0.0, 0.22, 3)
            pf_node_of[(pkj, s_i)] = len(kinds)
            add("PF", int(conn.cf_of_pkj[pkj]), pkj * n_pf_shown + s_i, pos)

    for pkj in range(n_pkj):
        th = pkj_theta[pkj]
        pkj_node_of[pkj] = len(kinds)
        add("PKJ", int(conn.cf_of_pkj[pkj]), pkj,
            np.array([R_PKJ * np.cos(th), R_PKJ * np.sin(th), Z_PKJ]))

    dcn_pos, _ = _ring_positions(n_dcn, R_DCN, Z_DCN)
    for d in range(n_dcn):
        dcn_node_of[d] = len(kinds)
        add("DCN", d, d, dcn_pos[d])

    io_pos, _ = _ring_positions(n_io, R_IO, Z_IO)
    for i in range(n_io):
        io_node_of[i] = len(kinds)
        add("IO", i, i, io_pos[i])

    src, dst, ekind, eweight = [], [], [], []

    def connect(a, b, kind, w):
        src.append(a); dst.append(b); ekind.append(kind); eweight.append(float(w))

    for pkj in range(n_pkj):                                   # PF -> PKJ (plastic; shown subsample)
        for s_i in range(n_pf_shown):
            w = cfg.w_init if weights is None else weights[pkj, s_i]
            connect(pf_node_of[(pkj, s_i)], pkj_node_of[pkj], "pf_pkj", w)

    for pkj, targets in enumerate(conn.pkj_to_dcn):            # PKJ -> DCN, from the table
        for d in targets:
            connect(pkj_node_of[pkj], dcn_node_of[d], "pkj_dcn", cfg.pkj_dcn_gain)

    for d, targets in enumerate(conn.dcn_to_io):               # DCN -> IO, from the table
        for i in targets:
            connect(dcn_node_of[d], io_node_of[i], "dcn_io", cfg.dcn_io_gaba_gain)

    for i, targets in enumerate(conn.io_to_pkj):               # IO -> PKJ, the climbing fiber
        for pkj in targets:
            connect(io_node_of[i], pkj_node_of[pkj], "io_pkj", cfg.cf_pause_g)

    if gap_matrix is not None:                                 # IO <-> IO, one edge per pair
        gm = np.asarray(gap_matrix, dtype=float)
        for a in range(n_io):
            for b in range(a + 1, n_io):
                if gm[a, b] != 0.0:
                    connect(io_node_of[a], io_node_of[b], "io_io", gm[a, b])

    return NetworkGraph(
        kind=np.array(kinds), group=np.array(groups), index=np.array(indices),
        position=np.array(positions, dtype=float),
        edge_src=np.array(src, dtype=int), edge_dst=np.array(dst, dtype=int),
        edge_kind=np.array(ekind), edge_weight=np.array(eweight, dtype=float),
        meta={"n_pf_per_pkj_shown": n_pf_shown, "n_pf_per_pkj_actual": cfg.n_pf_per_pkj,
              "gap_topology": cfg.gap_topology, "n_io": n_io, "n_dcn": n_dcn, "n_pkj": n_pkj},
    )
