#!/usr/bin/env python3
"""H1 on the assembled microzone: does the closed loop settle at an equilibrium?

The quickest useful run in the project. Reports the three numbers H1 turns on --
CF rate, mean PF->PKJ weight, and the weight's drift -- plus raster, weight and
membrane-potential figures. experiments/run_network.py does the same thing and
adds the wiring figures and the synchrony analysis; this script is what you run
when you only want the numbers.

Caveats worth knowing before reading the output:

  * 60 s is NOT long enough to call an equilibrium. The weights need a few
    hundred seconds to reach their balance point from w_init, so the drift this
    prints on a default run is still measuring the transient. Use
    --duration-s 300 or more before quoting a drift, and see the README's
    settling result for what the transient looks like.
  * n=1. Every number here is one seed. The replicated versions of these results
    come from experiments/sweep.py.

There is no smaller network to run: CbmSim's connection counts have to close
(320 PKJ x 3 targets = 960 = 80 DCN x 12 inputs), so the populations cannot be
shrunk independently, and sim/connectivity.py raises rather than silently
rewiring. Use --gap-g 0 to run the same loop with the olive uncoupled.

Writes results/baseline_rasters.png, _weights.png and _voltages.png.
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
    parser.add_argument("--duration-s", type=float, default=60.0,
                         help="simulated seconds after burn-in; about 2.5x that in wall-clock at "
                              "the shipped scale. Use 300+ if you intend to read the drift "
                              "figure (default: 60)")
    parser.add_argument("--gap-g", type=float, default=None,
                         help="conductance of ONE gap junction, mS/cm^2; 0 uncouples the olive "
                              "(default: config.SimConfig.gap_g, 0.0143)")
    parser.add_argument("--cf-reversal-mv", type=float, default=None,
                         help="reversal potential of the CF conductance on PKJ: 0 = excitatory "
                              "climbing fiber (the shipped default), -75 = the original "
                              "post-complex-spike pause "
                              "(default: config.SimConfig.cf_pkj_reversal_mv)")
    parser.add_argument("--seed", type=int, default=0,
                         help="seeds the PF Poisson draws and the membrane noise (default: 0)")
    parser.add_argument("--out-dir", type=str, default="results",
                         help="directory the three PNGs are written to (default: results)")
    parser.add_argument("--n-tracked-synapses", type=int, default=15,
                         help="individual PF->PKJ synapses plotted alongside the mean; "
                              "0 disables the lower panel (default: 15)")
    args = parser.parse_args()

    cfg = SimConfig(duration_s=args.duration_s, seed=args.seed,
                     n_tracked_synapses=args.n_tracked_synapses)
    if args.gap_g is not None:
        cfg.gap_g = args.gap_g
    if args.cf_reversal_mv is not None:
        cfg.cf_pkj_reversal_mv = args.cf_reversal_mv
    sim = Simulation(cfg)
    print(f"[baseline] {sim.conn.describe()}")
    print(f"[baseline] CF -> PKJ reversal {cfg.cf_pkj_reversal_mv:+.1f} mV "
          f"({'excitatory' if cfg.cf_pkj_reversal_mv > cfg.pkj.v_th_mv else 'inhibitory pause'})")
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
