#!/usr/bin/env python3
"""Experiment 1: the microzone held at equilibrium (H1).

Reports the numbers H1 turns on -- CF rate, mean PF->PKJ weight, drift -- plus
the raster, weight and membrane-potential plots, without the wiring figures and
synchrony analysis that experiments/run_network.py adds.

There is no smaller unit to run. CbmSim's connection counts only close for the
whole microzone (32 PKJ x 3 targets = 96 = 8 DCN x 12 inputs), so shrinking the
olive in isolation would leave the projection inconsistent; sim/connectivity.py
rejects that rather than silently rewiring. Use --gap-g 0 to see the loop with
the olivary cells electrically uncoupled.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import plot_rasters, plot_voltage_traces, plot_weights, summarize
from sim.simulate import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--gap-g", type=float, default=None,
                         help="conductance of one gap junction, mS/cm^2 (0 = uncoupled olive)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--n-tracked-synapses", type=int, default=15,
                         help="individual PF->PKJ synapses to plot alongside the mean (0 to disable)")
    args = parser.parse_args()

    cfg = SimConfig(duration_s=args.duration_s, seed=args.seed,
                     n_tracked_synapses=args.n_tracked_synapses)
    if args.gap_g is not None:
        cfg.gap_g = args.gap_g
    sim = Simulation(cfg)
    print(f"[baseline] {sim.conn.describe()}")
    log = sim.run()

    s = summarize(log)
    print(f"[baseline] PKJ {s['pkj_rate_hz']:.2f} Hz | DCN {s['dcn_rate_hz']:.2f} Hz | "
          f"IO {s['io_rate_hz']:.3f} Hz (target ~1 Hz)")
    print(f"[baseline] IO inter-CF interval CV: {s['io_isi_cv']:.2f}")
    print(f"[baseline] final mean PF->PKJ weight: {s['mean_weight']:.4f}")
    print(f"[baseline] weight drift slope: {s['weight_drift_slope']:+.5f} units/s")
    print(f"[baseline] weight saturated (near 0 or 1)? {s['weight_saturated']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_weights(log, out_dir / "baseline_weights.png",
                  title=f"Closed-loop baseline (H1) -- LTD window {cfg.ltd_window_ms:.0f} ms, "
                        f"$\\delta_-/\\delta_+$ = {cfg.delta_minus / cfg.delta_plus:.1f}")
    plot_rasters(log, out_dir / "baseline_rasters.png", window_s=10.0, title="Closed-loop baseline (H1)")
    plot_voltage_traces(log, out_dir / "baseline_voltages.png", title="Closed-loop baseline (H1)")
    print(f"[baseline] plots written to {out_dir}/")


if __name__ == "__main__":
    main()
