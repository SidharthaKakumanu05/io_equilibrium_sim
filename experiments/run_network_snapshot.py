#!/usr/bin/env python3
"""Snapshot of the wired network: every neuron in 3D, and the connectivity
matrix the wiring induces. Instant -- it runs no simulation.

It builds exactly the Connectivity and gap-junction matrix sim/simulate.py would
build from this config, turns them into a graph, and draws it. So it is a
picture of the network that would actually run, not a redrawing of the intent,
and it is the fastest way to see what a topology or population change does to
the wiring before paying for a simulation.

  results/network_3d.png            neurons in 3D + one path around the loop + the gap web
  results/connectivity_matrix.png   adjacency matrix, block-structured by cell type

Both are also written by run_network.py, from the network that just ran.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.connectivity import build_connectivity
from sim.io_coupling import build_gap_junction_matrix, coupling_summary
from sim.network_graph import build_network_graph
from sim.network_viz import plot_connectivity_matrix, plot_network_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-io", type=int, default=None,
                         help="override cfg.n_io; n_dcn follows as 2*n_io, which is what the "
                              "PKJ->DCN counts require (n_pkj*n_dcn_per_pkj == n_dcn*n_pkj_per_dcn)")
    parser.add_argument("--gap-topology", type=str, default=None,
                         choices=("all_to_all", "ring", "nearest_k", "small_world"),
                         help="override cfg.gap_topology; the panel-3 web is what changes "
                              "(default: nearest_k)")
    parser.add_argument("--gap-g", type=float, default=None,
                         help="override cfg.gap_g, the conductance of ONE junction (mS/cm^2). "
                              "0 draws no junctions at all (default: 0.0143)")
    parser.add_argument("--n-pf-shown", type=int, default=8,
                         help="PF units drawn per Purkinje cell. All 500 is neither drawable nor "
                              "a readable adjacency matrix; every other cell is shown in full "
                              "(default: 8)")
    parser.add_argument("--open-loop", action="store_true",
                         help="set enforce_closed_loop=False: rotate which block of the nucleus "
                              "each olivary cell reads, misrouting the feedback without cutting it")
    parser.add_argument("--out-dir", type=str, default="results",
                         help="directory the two PNGs are written to (default: results)")
    parser.add_argument("--seed", type=int, default=0,
                         help="seeds only the drawn PF cloud's jitter, not the wiring "
                              "(that is cfg.connectivity_seed) (default: 0)")
    args = parser.parse_args()

    cfg = SimConfig()
    if args.n_io is not None:
        cfg.n_io = args.n_io
        cfg.n_dcn = 2 * args.n_io      # forced by the connection counts; see the flag's help
    if args.gap_topology is not None:
        cfg.gap_topology = args.gap_topology
    if args.gap_g is not None:
        cfg.gap_g = args.gap_g
    cfg.enforce_closed_loop = not args.open_loop

    conn = build_connectivity(
        n_io=cfg.n_io, n_dcn=cfg.n_dcn, n_pkj_per_io=cfg.n_pkj_per_io,
        n_pkj_per_dcn=cfg.n_pkj_per_dcn, n_dcn_per_pkj=cfg.n_dcn_per_pkj,
        n_dcn_per_io=cfg.n_dcn_per_io, enforce_closed_loop=cfg.enforce_closed_loop,
        seed=cfg.connectivity_seed)
    gap = build_gap_junction_matrix(cfg.n_io, cfg.gap_g, cfg.gap_topology, cfg.gap_n_neighbors)
    graph = build_network_graph(cfg, conn, gap, n_pf_per_pkj_shown=args.n_pf_shown, seed=args.seed)

    print(f"[snapshot] {conn.describe()}")
    print(f"[snapshot] gap junctions: {coupling_summary(gap, cfg.io_channels.g_leak)}")
    print(f"[snapshot] graph: {graph.summary()}")
    print(f"[snapshot] full network has {cfg.n_io * cfg.n_pkj_per_io * cfg.n_pf_per_pkj} PF units; "
          f"{graph.meta['n_pf_per_pkj_shown']} per Purkinje cell are drawn")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_network_3d(graph, out_dir / "network_3d.png",
                     title=f"io_equilibrium_sim -- {cfg.n_io} IO / {cfg.n_dcn} DCN / {conn.n_pkj} PKJ, "
                           f"{cfg.gap_topology} IO coupling")
    plot_connectivity_matrix(graph, out_dir / "connectivity_matrix.png")
    print(f"[snapshot] wrote {out_dir}/network_3d.png and {out_dir}/connectivity_matrix.png")


if __name__ == "__main__":
    main()
