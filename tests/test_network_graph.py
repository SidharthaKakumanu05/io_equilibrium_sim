"""Unit tests for sim/network_graph.py.

The graph is what the 3D snapshot and the connectivity matrix are drawn from, so
these check that it really describes the circuit sim/simulate.py wires up --
right projections, right directions, right routing -- rather than that a figure
was produced.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.connectivity import build_connectivity
from sim.io_coupling import build_gap_junction_matrix
from sim.network_graph import EDGE_KINDS, build_network_graph
from sim.simulate import Simulation


# A small, fully specified microzone for the mechanics tests: 2 IO x 4 PKJ = 8 PKJ,
# 4 DCN each reading 4 PKJ, each PKJ driving 2 DCN (8x2 = 16 = 4x4), every DCN reaching
# both IO. These tests are about the graph's mechanics, not about whatever the shipped
# ratios happen to be; the shipped ones are pinned by their own test below.
SMALL = dict(n_io=2, n_dcn=4, n_pkj_per_io=4, n_pkj_per_dcn=4, n_dcn_per_pkj=2, n_dcn_per_io=4)


def build(n_pf_shown=3, closed=True, gap_g=0.019, topology="all_to_all", **over):
    kw = dict(SMALL); kw.update(over)
    cfg = SimConfig(enforce_closed_loop=closed, gap_g=gap_g, gap_topology=topology, **kw)
    conn = build_connectivity(
        n_io=cfg.n_io, n_dcn=cfg.n_dcn, n_pkj_per_io=cfg.n_pkj_per_io,
        n_pkj_per_dcn=cfg.n_pkj_per_dcn, n_dcn_per_pkj=cfg.n_dcn_per_pkj,
        n_dcn_per_io=cfg.n_dcn_per_io, enforce_closed_loop=cfg.enforce_closed_loop,
        seed=cfg.connectivity_seed)
    gap = build_gap_junction_matrix(cfg.n_io, cfg.gap_g, cfg.gap_topology, cfg.gap_n_neighbors)
    return cfg, conn, build_network_graph(cfg, conn, gap, n_pf_per_pkj_shown=n_pf_shown)


class TestNodes(unittest.TestCase):
    def test_node_counts_match_the_configuration(self):
        cfg, conn, g = build(n_pf_shown=3)
        self.assertEqual(len(g.node_ids("IO")), cfg.n_io)
        self.assertEqual(len(g.node_ids("DCN")), cfg.n_dcn)
        self.assertEqual(len(g.node_ids("PKJ")), conn.n_pkj)
        self.assertEqual(len(g.node_ids("PF")), conn.n_pkj * 3)

    def test_nodes_are_ordered_pf_pkj_dcn_io(self):
        *_, g = build()
        self.assertEqual([k for k, _, _ in g.kind_blocks()], ["PF", "PKJ", "DCN", "IO"])
        for _, start, stop in g.kind_blocks():
            self.assertTrue(len(set(g.kind[start:stop])) == 1)          # each block is one cell type

    def test_every_node_has_a_finite_position(self):
        *_, g = build()
        self.assertEqual(g.position.shape, (g.n_nodes, 3))
        self.assertTrue(np.all(np.isfinite(g.position)))

    def test_layers_are_stacked_in_pathway_order(self):
        """IO at the base, then DCN, then PKJ, then PF: the picture should read
        bottom-to-top the way the loop returns to the olive."""
        *_, g = build()
        z = {k: g.position[g.node_ids(k), 2].mean() for k in ("PF", "PKJ", "DCN", "IO")}
        self.assertLess(z["IO"], z["DCN"])
        self.assertLess(z["DCN"], z["PKJ"])
        self.assertLess(z["PKJ"], z["PF"])

    def test_pf_subsampling_is_capped_at_the_real_pool_size(self):
        cfg, conn, _ = build()
        g = build_network_graph(cfg, conn, None, n_pf_per_pkj_shown=10 * cfg.n_pf_per_pkj)
        self.assertEqual(g.meta["n_pf_per_pkj_shown"], cfg.n_pf_per_pkj)


class TestEdges(unittest.TestCase):
    def test_every_edge_kind_is_known_and_correctly_typed(self):
        *_, g = build()
        for kind in np.unique(g.edge_kind):
            self.assertIn(kind, EDGE_KINDS)
        for kind, src_kind, dst_kind in (("pf_pkj", "PF", "PKJ"), ("pkj_dcn", "PKJ", "DCN"),
                                          ("dcn_io", "DCN", "IO"), ("io_pkj", "IO", "PKJ"),
                                          ("io_io", "IO", "IO")):
            sel = g.edge_kind == kind
            self.assertTrue(sel.any(), f"no {kind} edges")
            self.assertTrue(np.all(g.kind[g.edge_src[sel]] == src_kind))
            self.assertTrue(np.all(g.kind[g.edge_dst[sel]] == dst_kind))

    def test_edge_counts_match_the_wiring(self):
        cfg, conn, g = build(n_pf_shown=3)
        self.assertEqual(int((g.edge_kind == "pf_pkj").sum()), conn.n_pkj * 3)
        self.assertEqual(int((g.edge_kind == "io_pkj").sum()), conn.n_pkj)          # one CF per Purkinje cell
        self.assertEqual(int((g.edge_kind == "pkj_dcn").sum()), cfg.n_dcn * cfg.n_pkj_per_dcn)
        self.assertEqual(int((g.edge_kind == "pkj_dcn").sum()), conn.n_pkj * cfg.n_dcn_per_pkj)
        self.assertEqual(int((g.edge_kind == "dcn_io").sum()), cfg.n_io * cfg.n_dcn_per_io)
        self.assertEqual(int((g.edge_kind == "io_io").sum()),
                          cfg.n_io * (cfg.n_io - 1) // 2)                            # all-to-all, one per pair

    def test_climbing_fibers_stay_within_their_territory(self):
        *_, g = build()
        sel = g.edge_kind == "io_pkj"
        np.testing.assert_array_equal(g.group[g.edge_src[sel]], g.group[g.edge_dst[sel]])

    def test_every_purkinje_cell_influences_every_olive(self):
        """Spec section 2.2(2) -- every PKJ an IO's climbing fiber contacts must help
        regulate that same IO. Under CbmSim's wiring this holds for *all* pairs, not
        just a cell's own territory, because DCN->IO is complete: PKJ -> its 3 DCN ->
        every IO. That is also why 2.2(1) cannot hold; see sim/connectivity.py."""
        cfg, conn, _ = build()
        sim = Simulation(cfg)
        reach = (sim.m_dcn_to_io @ sim.m_pkj_to_dcn) > 0          # (n_io, n_pkj)
        self.assertTrue(np.all(reach), "some Purkinje cell does not reach some olivary cell")

    def test_open_loop_rotates_when_the_projection_is_partial(self):
        """enforce_closed_loop can only rotate DCN->IO when it is not already
        complete. With a partial projection the rotation must actually move which
        nuclear cells reach which olivary cell."""
        _, closed, _ = build(n_dcn_per_io=2, closed=True)
        _, opened, _ = build(n_dcn_per_io=2, closed=False)
        self.assertFalse(closed.meta["dcn_to_io_complete"])
        self.assertNotEqual([sorted(t) for t in closed.dcn_to_io],
                             [sorted(t) for t in opened.dcn_to_io])

    def test_complete_projection_reports_that_it_cannot_be_opened(self):
        """At CbmSim's ratios DCN->IO is complete, so asking for an open loop
        cannot do anything -- and the connectivity says so rather than pretending."""
        _, conn, _ = build(closed=False)                           # SMALL has n_dcn_per_io == n_dcn
        self.assertTrue(conn.meta["dcn_to_io_complete"])
        self.assertIn("nothing to rotate", conn.meta["loop"])

    def test_gap_junctions_appear_once_per_unordered_pair(self):
        *_, g = build(n_io=6, n_dcn=4, n_pkj_per_io=4, n_pkj_per_dcn=12,
                       n_dcn_per_pkj=2, n_dcn_per_io=4, topology="ring")
        sel = g.edge_kind == "io_io"
        pairs = {tuple(sorted(p)) for p in zip(g.edge_src[sel], g.edge_dst[sel])}
        self.assertEqual(len(pairs), int(sel.sum()))                  # no duplicated pair
        self.assertEqual(int(sel.sum()), 6)                            # a 6-cell ring has 6 junctions

    def test_zero_conductance_means_no_gap_edges(self):
        *_, g = build(gap_g=0.0)
        self.assertEqual(int((g.edge_kind == "io_io").sum()), 0)


class TestAdjacencyMatrix(unittest.TestCase):
    def test_shape_and_entry_count(self):
        """One entry per directed edge, and two per gap junction: the edge list
        stores each junction once per unordered pair, but the matrix has to show
        it conducting both ways."""
        *_, g = build()
        a = g.adjacency_matrix()
        self.assertEqual(a.shape, (g.n_nodes, g.n_nodes))
        n_undirected = int((g.edge_kind == "io_io").sum())
        self.assertEqual(int((a != 0).sum()), g.n_edges + n_undirected)

    def test_directed_projections_appear_only_in_their_own_direction(self):
        *_, g = build()
        a = g.adjacency_matrix(kinds=("dcn_io",))
        sel = g.edge_kind == "dcn_io"
        self.assertTrue(np.all(a[g.edge_dst[sel], g.edge_src[sel]] == 0.0))   # no DCN <- IO entries

    def test_signs_follow_the_projection(self):
        *_, g = build()
        a = g.adjacency_matrix(signed=True)
        for kind, expect_positive in (("pf_pkj", True), ("io_pkj", True),
                                       ("pkj_dcn", False), ("dcn_io", False)):
            sel = g.edge_kind == kind
            values = a[g.edge_src[sel], g.edge_dst[sel]]
            self.assertTrue(np.all(values > 0) if expect_positive else np.all(values < 0), kind)

    def test_kind_filter_selects_only_that_projection(self):
        *_, g = build()
        a = g.adjacency_matrix(kinds=("dcn_io",))
        self.assertEqual(int((a != 0).sum()), int((g.edge_kind == "dcn_io").sum()))
        io, dcn = g.node_ids("IO"), g.node_ids("DCN")
        self.assertEqual(int((a != 0).sum()), int((a[np.ix_(dcn, io)] != 0).sum()))  # nothing outside the block

    def test_gap_junction_block_is_symmetric(self):
        *_, g = build()
        a = g.adjacency_matrix(signed=False, kinds=("io_io",))
        io = g.node_ids("IO")
        block = a[np.ix_(io, io)]
        np.testing.assert_allclose(block, block.T)

    def test_per_kind_normalization_puts_every_projection_on_the_same_scale(self):
        """Without it the PKJ->DCN block (~0.008) is invisible next to the CF
        block (~1.0), which is exactly what the plot has to avoid."""
        *_, g = build()
        a = g.adjacency_matrix(signed=True, normalize_per_kind=True)
        self.assertLessEqual(np.abs(a).max(), 1.0 + 1e-12)
        for kind in ("pf_pkj", "pkj_dcn", "dcn_io", "io_pkj"):
            sel = g.edge_kind == kind
            self.assertAlmostEqual(np.abs(a[g.edge_src[sel], g.edge_dst[sel]]).max(), 1.0, places=9)


class TestMatchesTheSimulation(unittest.TestCase):
    """The graph is only worth drawing if it is the network that actually runs."""

    def test_routing_matches_the_simulation_matrices(self):
        cfg = SimConfig(enforce_closed_loop=True, **SMALL)
        sim = Simulation(cfg)
        g = build_network_graph(cfg, sim.conn, sim.gap_matrix, n_pf_per_pkj_shown=2)

        # PKJ -> DCN: the graph edge set must equal the nonzero entries of sim.m_pkj_to_dcn.
        sel = g.edge_kind == "pkj_dcn"
        pkj0, dcn0 = g.node_ids("PKJ")[0], g.node_ids("DCN")[0]
        from_graph = {(int(s - pkj0), int(d - dcn0)) for s, d in zip(g.edge_src[sel], g.edge_dst[sel])}
        from_sim = {(int(p), int(d)) for d, p in zip(*np.nonzero(sim.m_pkj_to_dcn))}
        self.assertEqual(from_graph, from_sim)

        # DCN -> IO likewise.
        sel = g.edge_kind == "dcn_io"
        io0 = g.node_ids("IO")[0]
        from_graph = {(int(s - dcn0), int(d - io0)) for s, d in zip(g.edge_src[sel], g.edge_dst[sel])}
        from_sim = {(int(d), int(i)) for i, d in zip(*np.nonzero(sim.m_dcn_to_io))}
        self.assertEqual(from_graph, from_sim)

    def test_gap_edges_match_the_simulated_coupling_matrix(self):
        cfg = SimConfig(gap_topology="ring", **SMALL)
        sim = Simulation(cfg)
        g = build_network_graph(cfg, sim.conn, sim.gap_matrix, n_pf_per_pkj_shown=1)
        io0 = g.node_ids("IO")[0]
        sel = g.edge_kind == "io_io"
        for s, d, w in zip(g.edge_src[sel], g.edge_dst[sel], g.edge_weight[sel]):
            self.assertAlmostEqual(w, sim.gap_matrix[int(s - io0), int(d - io0)])

    def test_live_weights_are_carried_into_the_graph(self):
        """`weights` is the flat (n_pkj, n_pf) matrix the simulation holds, so the
        drawn PF edges carry whatever the run ended on rather than w_init."""
        cfg, conn, _ = build()
        weights = np.full((conn.n_pkj, cfg.n_pf_per_pkj), 0.25)
        weights[cfg.n_pkj_per_io:] = 0.75                     # second IO's Purkinje cells
        g = build_network_graph(cfg, conn, None, weights=weights, n_pf_per_pkj_shown=2)
        sel = g.edge_kind == "pf_pkj"
        by_group = {gr: g.edge_weight[sel][g.group[g.edge_dst[sel]] == gr] for gr in (0, 1)}
        np.testing.assert_allclose(by_group[0], 0.25)
        np.testing.assert_allclose(by_group[1], 0.75)

    def test_shipped_defaults_match_cbmsim(self):
        """The connection numbers the network is actually shipped at, against
        CbmSim's own (src/cbm_state/connectivityparams.cpp). A change to any
        population default that breaks one of these should fail here rather than
        be discovered in a figure."""
        cfg = SimConfig()                                   # whatever config.py ships
        sim = Simulation(cfg)
        conn = sim.conn
        # Populations are CbmSim's scaled up 10x; the PER-CELL CONNECTION NUMBERS below
        # are CbmSim's own, unscaled. That is the whole point of this scaling: convergence
        # onto each cell is untouched, so every cell's operating point is preserved and no
        # synaptic gain needed refitting.
        self.assertEqual(cfg.n_io, 40, "CbmSim num_io (4) x10")
        self.assertEqual(cfg.n_dcn, 80, "CbmSim num_nc (8) x10")
        self.assertEqual(conn.n_pkj, 320, "CbmSim num_pc (32) x10")
        self.assertEqual(cfg.n_pkj_per_io, 8, "CbmSim num_p_io_from_io_to_pc")
        self.assertEqual(cfg.n_pkj_per_dcn, 12, "CbmSim num_p_nc_from_pc_to_nc")
        self.assertEqual(cfg.n_dcn_per_pkj, 3, "CbmSim num_p_pc_from_pc_to_nc")
        self.assertEqual(cfg.n_dcn_per_io, 8, "CbmSim num_p_io_from_nc_to_io")

        # The counts have to be mutually consistent, and the wiring has to realize them exactly.
        self.assertEqual(cfg.n_dcn * cfg.n_pkj_per_dcn, conn.n_pkj * cfg.n_dcn_per_pkj)
        self.assertTrue(np.all(conn.pkj_per_dcn == cfg.n_pkj_per_dcn))
        self.assertTrue(np.all(conn.dcn_per_pkj == cfg.n_dcn_per_pkj))
        self.assertTrue(np.all(conn.dcn_per_io == cfg.n_dcn_per_io))
        self.assertTrue(np.all(conn.pkj_per_io == cfg.n_pkj_per_io))

        # N_IO << N_DCN << N_PKJ, which CbmSim's numbers satisfy without extra assumptions.
        self.assertLess(cfg.n_io, cfg.n_dcn)
        self.assertLess(cfg.n_dcn, conn.n_pkj)

        # Exactly one climbing fiber per Purkinje cell.
        self.assertEqual(len(np.unique(conn.cf_of_pkj)), cfg.n_io)
        self.assertTrue(np.all(np.bincount(conn.cf_of_pkj) == cfg.n_pkj_per_io))

    def test_no_two_io_cells_receive_identical_inhibition(self):
        """The property that keeps the olive from collapsing into one repeated cell.

        CbmSim's connectNCtoIO is complete: every nuclear cell inhibits every
        olivary cell. At CbmSim's own scale (n_dcn_per_io == n_dcn == 8) that
        makes every olivary cell's input identical, and since the cells are
        deterministic (noise_sigma = 0) and identically parameterised, they
        integrate to bit-identical voltage traces -- measured, not inferred.
        Gap-junction synchrony then reads 1.000 at every strength including zero,
        because there is nothing left to synchronise.

        Holding n_dcn_per_io at CbmSim's 8 while the nucleus grew to 80 makes the
        projection topographic instead, and that is what breaks the tie."""
        cfg = SimConfig()
        sim = Simulation(cfg)
        self.assertFalse(sim.conn.meta["dcn_to_io_complete"])

        rows = {tuple(r) for r in sim.m_dcn_to_io.astype(int)}
        self.assertEqual(len(rows), cfg.n_io,
                         "two olivary cells share an inhibition pattern: they will be redundant")

        # Every nuclear cell still projects somewhere -- no dangling half of the nucleus.
        self.assertTrue(np.all(sim.m_dcn_to_io.sum(axis=0) > 0))
        # Convergence is exactly CbmSim's, and the load is balanced across the nucleus.
        self.assertTrue(np.all(sim.conn.dcn_per_io == cfg.n_dcn_per_io))
        self.assertTrue(np.all(sim.conn.io_per_dcn == sim.conn.io_per_dcn[0]))

    def test_every_cf_target_feeds_back_to_its_own_olivary_cell(self):
        """Spec 2.2(2): every Purkinje cell a climbing fiber contacts must help
        regulate that same olivary cell.

        With a complete DCN->IO projection this is trivially true (every PKJ
        reaches every IO). With the topographic projection it is a real
        constraint on how the two maps line up -- a Purkinje cell's own nuclear
        target has to land inside the block of the nucleus that its climbing
        fiber's olivary cell reads back from -- so it is worth pinning down."""
        cfg = SimConfig()
        sim = Simulation(cfg)
        reach = (sim.m_dcn_to_io @ sim.m_pkj_to_dcn) > 0     # (n_io, n_pkj)
        cf = sim.conn.cf_of_pkj
        feeds_back = reach[cf, np.arange(sim.conn.n_pkj)]
        self.assertTrue(np.all(feeds_back),
                        f"{int((~feeds_back).sum())} PKJ do not feed back to their own CF's olive")

    def test_each_olivary_cell_sees_only_part_of_the_microzone(self):
        """The converse of the test above, and the thing that actually changed.

        Spec 2.2(1) wants an olivary cell modulated only via its own CF targets.
        That is not satisfiable in this connectivity, but with DCN->IO complete
        EVERY Purkinje cell reached EVERY olivary cell, which is the maximal
        violation and also the cause of the synchrony degeneracy. Topographic
        DCN->IO cuts the reachable fraction well below the whole population."""
        cfg = SimConfig()
        sim = Simulation(cfg)
        reach = (sim.m_dcn_to_io @ sim.m_pkj_to_dcn) > 0
        frac = reach.sum(axis=1) / sim.conn.n_pkj
        self.assertTrue(np.all(frac < 0.5),
                        f"an olivary cell is reached by {frac.max():.0%} of the microzone")

    def test_pkj_to_dcn_overlap_is_topographic_plus_random(self):
        """The projection keeps CbmSim's structure: a contiguous topographic block
        per nuclear cell, plus a randomized overlap that gives each Purkinje cell
        its remaining targets. The block makes the first target deterministic."""
        cfg = SimConfig()
        conn = Simulation(cfg).conn
        block = conn.n_pkj // cfg.n_dcn
        for p in range(conn.n_pkj):
            self.assertIn(p // block, conn.pkj_to_dcn[p], "topographic base missing")
            self.assertEqual(len(set(conn.pkj_to_dcn[p])), cfg.n_dcn_per_pkj, "duplicate synapse")


if __name__ == "__main__":
    unittest.main()
