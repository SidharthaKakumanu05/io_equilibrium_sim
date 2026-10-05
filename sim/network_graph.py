"""The network as a graph (nodes and edges), for drawing. Not used by run.py.

build_network_graph() takes the same Connectivity and gap-junction matrix the
simulation uses and turns them into:

  nodes  one per cell, in the order PF, PKJ, DCN, IO, each with a type,
         a group (climbing fibre for PF/PKJ, own index for DCN/IO) and an
         (x, y, z) position for drawing
  edges  one per connection, with its type (see EDGE_KINDS) and strength

sim/network_viz.py draws it. Only a few PFs per Purkinje cell are included
(n_pf_per_pkj_shown, default 8); all 160,000 could not be drawn.
"""
from dataclasses import dataclass, field

import numpy as np

# The five kinds of connection. sign: +1 excitatory, -1 inhibitory, 0 electrical (no direction).
EDGE_KINDS = {
    "pf_pkj":  {"sign": +1, "label": "PF -> PKJ (excitatory, plastic)"},
    "io_pkj":  {"sign": +1, "label": "IO -> PKJ (climbing fiber / teaching)"},
    "pkj_dcn": {"sign": -1, "label": "PKJ -> DCN (inhibitory)"},
    "dcn_io":  {"sign": -1, "label": "DCN -> IO (inhibitory feedback)"},
    "io_io":   {"sign":  0, "label": "IO <-> IO (gap junction, electrical)"},
}


@dataclass
class NetworkGraph:
    """The graph. Nodes are numbered 0..n_nodes-1 in PF, PKJ, DCN, IO order;
    each array below has one entry per node (N) or per edge (E)."""
    kind: np.ndarray            # (N,) cell-type string per node: "PF" | "PKJ" | "DCN" | "IO"
    group: np.ndarray           # (N,) climbing-fiber territory (PF/PKJ) or the cell's own index (DCN/IO)
    index: np.ndarray           # (N,) index within its cell type
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
        """Node numbers of every cell of one type, e.g. "IO"."""
        return np.flatnonzero(self.kind == kind)

    def adjacency_matrix(self, signed=True, kinds=None, normalize_per_kind=False):
        """(N, N) matrix with A[i, j] = strength of the connection from node i to node j.

        signed:             make inhibitory entries negative
        kinds:              only include these connection types (default: all)
        normalize_per_kind: divide each connection type by its own maximum, so
                            all of them show up on one colour scale (they are
                            in different units)
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
                # Gap junctions are stored once per pair; fill in both directions.
                a[dst, src] += value
        return a

    def kind_blocks(self):
        """[(type, first node, last node + 1)] for each cell type, in order."""
        blocks, start = [], 0
        for k in ("PF", "PKJ", "DCN", "IO"):
            n = int((self.kind == k).sum())
            if n:
                blocks.append((k, start, start + n))
                start += n
        return blocks

    def summary(self):
        """One-line count of nodes and edges by type."""
        counts = ", ".join(f"{int((self.kind == k).sum())} {k}" for k, _, _ in self.kind_blocks())
        per_kind = ", ".join(f"{int((self.edge_kind == k).sum())} {k}"
                             for k in EDGE_KINDS if (self.edge_kind == k).any())
        return f"{self.n_nodes} nodes ({counts}); {self.n_edges} edges ({per_kind})"


def _ring_positions(n, radius, z, phase=0.0):
    """n points evenly spaced on a circle of `radius` at height z."""
    theta = phase + 2.0 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)], axis=1), theta


def build_network_graph(cfg, conn, gap_matrix=None, weights=None, n_pf_per_pkj_shown=8, seed=0):
    """Build a NetworkGraph from a SimConfig and Connectivity.

    weights:    optional PF->PKJ weights to label edges with (default: w_init)
    gap_matrix: optional gap-junction matrix, to include IO <-> IO edges

    Positions are for drawing only. Each cell type is a ring at its own height:
    IO at the bottom, then DCN, then PKJ, then PFs on top. PKJ are ordered by
    climbing fibre, so each olive cell's 8 PKJ sit together.
    """
    rng = np.random.default_rng(seed)
    n_io, n_dcn, n_pkj = conn.n_io, conn.n_dcn, conn.n_pkj
    n_pf_shown = min(n_pf_per_pkj_shown, cfg.n_pf_per_pkj)

    kinds, groups, indices, positions = [], [], [], []

    def add(kind, group, index, pos):
        kinds.append(kind); groups.append(group); indices.append(index); positions.append(pos)

    Z_IO, Z_DCN, Z_PKJ, Z_PF = 0.0, 3.0, 6.0, 8.2         # height of each layer
    R_IO, R_DCN, R_PKJ = 1.6, 5.0, 7.6                    # radius of each ring

    # Angle of each PKJ around the ring.
    pkj_theta = 2.0 * np.pi * np.arange(n_pkj) / max(n_pkj, 1)

    pf_node_of, pkj_node_of, dcn_node_of, io_node_of = {}, {}, {}, {}   # cell index -> node number

    # --- nodes ---

    for pkj in range(n_pkj):                                   # a small cloud of PFs above each PKJ
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

    # --- edges ---
    src, dst, ekind, eweight = [], [], [], []

    def connect(a, b, kind, w):
        src.append(a); dst.append(b); ekind.append(kind); eweight.append(float(w))

    for pkj in range(n_pkj):                                   # PF -> PKJ (only the PFs included above)
        for s_i in range(n_pf_shown):
            w = cfg.w_init if weights is None else weights[pkj, s_i]
            connect(pf_node_of[(pkj, s_i)], pkj_node_of[pkj], "pf_pkj", w)

    for pkj, targets in enumerate(conn.pkj_to_dcn):            # PKJ -> DCN
        for d in targets:
            connect(pkj_node_of[pkj], dcn_node_of[d], "pkj_dcn", cfg.pkj_dcn_gain)

    for d, targets in enumerate(conn.dcn_to_io):               # DCN -> IO
        for i in targets:
            connect(dcn_node_of[d], io_node_of[i], "dcn_io", cfg.dcn_io_gaba_gain)

    for i, targets in enumerate(conn.io_to_pkj):               # IO -> PKJ (climbing fibres)
        for pkj in targets:
            connect(io_node_of[i], pkj_node_of[pkj], "io_pkj", cfg.cf_pause_g)

    if gap_matrix is not None:                                 # IO <-> IO gap junctions, one edge per pair
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
