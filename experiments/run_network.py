#!/usr/bin/env python3
"""The scaled-up network run: 8 gap-junction-coupled IO neurons and the closed
loops behind them, recorded in full.

This is the experiment the recording layer was built for. It collects, and plots:

  results/network_rasters.png       PF / PKJ / DCN / IO spike rasters, one colour per group
  results/network_weights.png       mean PF->PKJ weight per group + tracked individual synapses
  results/network_voltages.png      membrane potentials for every cell type + the IO's GABA drive
  results/network_io_state.png      IO V and [Ca]i together -- the Ca2+ spike and its aftermath
  results/network_3d.png            the wired network in 3D
  results/connectivity_matrix.png   the adjacency matrix that wiring induces

Every spike of every cell is recorded exactly; rates below are counted from
those trains, not read off a smoothed trace.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import plot_io_state, plot_rasters, plot_voltage_traces, plot_weights, summarize
from sim.io_coupling import coupling_summary, synchrony_index
from sim.network_graph import build_network_graph
from sim.network_viz import plot_connectivity_matrix, plot_network_3d


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--burn-in-s", type=float, default=None)
    parser.add_argument("--n-io", type=int, default=None)
    parser.add_argument("--gap-topology", type=str, default=None,
                         choices=["all_to_all", "ring", "nearest_k"])
    parser.add_argument("--gap-g", type=float, default=None,
                         help="conductance of one gap junction, mS/cm^2 (0 = uncoupled)")
    parser.add_argument("--raster-window-s", type=float, default=10.0,
                         help="how much of the run the rasters show (all of it is a solid block)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    cfg = SimConfig(duration_s=args.duration_s, seed=args.seed)
    if args.burn_in_s is not None:
        cfg.burn_in_s = args.burn_in_s
    if args.n_io is not None:
        cfg.n_io = args.n_io
        cfg.n_dcn = 2 * args.n_io      # forced by the connection counts (n_pkj*n_dcn_per_pkj == n_dcn*n_pkj_per_dcn)
    if args.gap_topology is not None:
        cfg.gap_topology = args.gap_topology
    if args.gap_g is not None:
        cfg.gap_g = args.gap_g

    from sim.simulate import Simulation                                   # imported late so --help stays instant
    sim = Simulation(cfg)
    print(f"[network] {sim.conn.describe()}")
    print(f"[network] gap junctions: {coupling_summary(sim.gap_matrix, cfg.io_channels.g_leak)}")
    print(f"[network] running {cfg.duration_s:.0f} s (+{cfg.burn_in_s:.0f} s burn-in) ...")

    t0 = time.time()
    log = sim.run()
    print(f"[network] done in {time.time() - t0:.0f} s wall-clock")

    s = summarize(log)
    print(f"[network] PKJ {s['pkj_rate_hz']:6.2f} Hz | DCN {s['dcn_rate_hz']:6.2f} Hz | "
          f"IO {s['io_rate_hz']:5.2f} Hz (per cell: "
          f"{np.array2string(s['io_rate_per_cell'], precision=2, floatmode='fixed')})")
    print(f"[network] IO inter-CF interval CV {s['io_isi_cv']:.2f}  "
          f"(~0 is a clock, ~1 is Poisson; the loop should land in between)")
    print(f"[network] mean PF->PKJ weight {s['mean_weight']:.4f}, drift {s['weight_drift_slope']:+.5f} units/s, "
          f"saturated={s['weight_saturated']}, cross-synapse SD {s['cross_synapse_std']:.4f}")

    half = log.duration_ms / 2.0
    sync = synchrony_index(log.io_spikes, half, log.duration_ms)
    print(f"[network] CF synchrony index across the {cfg.n_io} IO cells: {sync:.3f} "
          f"(0 = independent, 1 = identical trains) at g_gap={cfg.gap_g} mS/cm^2, {cfg.gap_topology}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_rasters(log, out_dir / "network_rasters.png", window_s=args.raster_window_s,
                  title=f"{cfg.n_io}-group network")
    plot_weights(log, out_dir / "network_weights.png",
                  title=f"PF$\\to$PKJ weights ($\\delta_-/\\delta_+$ = "
                        f"{cfg.delta_minus / cfg.delta_plus:.0f}, LTD window {cfg.ltd_window_ms:.0f} ms)")
    plot_voltage_traces(log, out_dir / "network_voltages.png",
                         title="Membrane potentials, all cell types")
    plot_io_state(log, out_dir / "network_io_state.png",
                   title=f"IO membrane potential and [Ca$^{{2+}}$]$_i$ -- {cfg.n_io} coupled cells")

    graph = build_network_graph(cfg, sim.conn, sim.gap_matrix, weights=log.final_weights)
    plot_network_3d(graph, out_dir / "network_3d.png",
                     title=f"io_equilibrium_sim -- {cfg.n_io} closed-loop groups, {cfg.gap_topology} IO coupling")
    plot_connectivity_matrix(graph, out_dir / "connectivity_matrix.png")
    print(f"[network] plots written to {out_dir}/")


if __name__ == "__main__":
    main()
