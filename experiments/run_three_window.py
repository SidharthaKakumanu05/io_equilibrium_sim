#!/usr/bin/env python3
"""H3: does an LTD -> null -> LTP rule slow the weights' random walk?

The two-window rule potentiates on every PF spike a climbing fiber does not
follow, so every synapse is doing a random walk between LTD and LTP events, and
the population spreads out even once its MEAN has settled. H3's claim is that
inserting a null window -- a stretch after the LTD window where a CF event
produces no change at all -- slows that walk, at the cost of blocking LTP the
loop needs.

This script fits each scheme fairly before comparing them: it finds each one's
own zero-drift delta-/delta+ ratio by bracketing (as run_window_sweep.py does),
then runs both at their own balance point and compares the spread they settle
at. Comparing at a shared ratio would just be comparing two loops sitting at
different rates.

Metric note: the comparison is on the FINAL CROSS-SYNAPSE weight spread -- the
variance across individual PF->PKJ synapses at run end -- not the variance of
the mean weight over time. The mean is an average over tens of thousands of
synapses, so the law of large numbers crushes its variance toward zero under
either scheme and it cannot distinguish them. The random walk is only visible
per synapse, which is why cfg.n_tracked_synapses exists.

**Read the spread off a settled run or not at all.** The weights start uniform
at `w_init` and spread out over time, so on a short `--eval-duration-s` the
comparison measures how far each arm has got through its transient rather than
where it settles, and it can come out backwards. The 90 s default is the
practical minimum; the sweep uses 120 s and ten seeds per condition.

The replicated version is `experiments/sweep.py h3_three_window`, which sweeps
five null-window widths at ten seeds each on the full network; prefer it for a
result to quote. This script is the cheap two-condition check, and it is n=1.

Writes results/three_window_2w_weights.png and _3w_weights.png.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import plot_individual_and_mean_weights, predicted_ltd_ltp_ratio, weight_drift_slope
from sim.simulate import Simulation


def microzone(n_io, **overrides):
    """A valid microzone scaled to `n_io` olivary cells.

    The connection counts have to close, and that fixes everything else:
    n_pkj = 8*n_io, and n_pkj * n_dcn_per_pkj == n_dcn * n_pkj_per_dcn forces
    n_dcn = 2*n_io. n_dcn_per_io is capped at n_dcn, which at small n_io makes
    DCN->IO complete -- CbmSim's own case, and harmless here: this experiment
    measures a property of the plasticity rule, not of how the olive's cells
    differ from each other."""
    cfg = SimConfig(**overrides)
    cfg.n_io = n_io
    cfg.n_dcn = 2 * n_io
    cfg.n_dcn_per_io = min(cfg.n_dcn_per_io, cfg.n_dcn)
    return cfg


def run_once(ltd_window_ms, null_window_ms, ratio, delta_plus, duration_s, seed, n_io, track_n=0):
    cfg = microzone(
        n_io,
        duration_s=duration_s, seed=seed,
        ltd_window_ms=ltd_window_ms, null_window_ms=null_window_ms,           # null_window_ms=0 -> 2-window mode
        delta_plus=delta_plus, delta_minus=delta_plus * ratio,
        n_tracked_synapses=track_n,
    )
    return Simulation(cfg).run()


def find_zero_drift_ratio(ltd_window_ms, null_window_ms, candidates, delta_plus, duration_s, seed, n_io):
    """The zero-drift delta-/delta+ ratio for one window scheme, bracketed from
    the candidates and interpolated. None if no sign change was found."""
    slopes = []
    for r in candidates:
        log = run_once(ltd_window_ms, null_window_ms, r, delta_plus, duration_s, seed, n_io)
        slopes.append(weight_drift_slope(log))
    for i in range(len(candidates) - 1):                                       # same bracket-and-interpolate as run_window_sweep.py
        s0, s1 = slopes[i], slopes[i + 1]
        if s0 == 0:
            return candidates[i]
        if (s0 > 0) != (s1 > 0):
            r0, r1 = candidates[i], candidates[i + 1]
            return r0 + (0 - s0) * (r1 - r0) / (s1 - s0)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ltd-window-ms", type=float, default=100.0,
                         help="LTD window, shared by both schemes (default: 100)")
    parser.add_argument("--null-window-ms", type=float, default=100.0,
                         help="null window appended after the LTD window in the 3-window arm. "
                              "Wider means steadier weights and a lower CF rate (default: 100)")
    parser.add_argument("--delta-plus", type=float, default=0.001,
                         help="LTP increment, held fixed while delta_minus is fitted (default: 0.001)")
    parser.add_argument("--search-duration-s", type=float, default=30.0,
                         help="simulated seconds per candidate ratio during the bracket search; "
                              "7 candidates x 2 schemes of these (default: 30)")
    parser.add_argument("--eval-duration-s", type=float, default=90.0,
                         help="simulated seconds for the two final runs the spread is measured on "
                              "(default: 90)")
    parser.add_argument("--n-io", type=int, default=4,
                         help="olivary cells; n_dcn follows as 2*n_io and the whole microzone "
                              "scales with it. 4 is CbmSim's own size (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                         help="seeds the PF Poisson draws and the membrane noise (default: 0)")
    parser.add_argument("--out-dir", type=str, default="results",
                         help="directory the two PNGs are written to (default: results)")
    parser.add_argument("--n-tracked-synapses", type=int, default=15,
                         help="individual synapses plotted against the mean; 0 skips the "
                              "figures entirely (default: 15)")
    args = parser.parse_args()

    predicted_2w = predicted_ltd_ltp_ratio(args.ltd_window_ms)                 # same candidate bracket for both schemes
    candidates = sorted(set(
        max(0.05, round(predicted_2w * f, 3)) for f in [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
    ))

    ratio_2w = find_zero_drift_ratio(args.ltd_window_ms, 0.0, candidates, args.delta_plus,
                                      args.search_duration_s, args.seed, args.n_io)   # null=0 -> 2-window
    ratio_3w = find_zero_drift_ratio(args.ltd_window_ms, args.null_window_ms, candidates, args.delta_plus,
                                      args.search_duration_s, args.seed, args.n_io)   # 3-window

    print(f"[three_window] 2-window zero-drift ratio: {ratio_2w}")
    print(f"[three_window] 3-window zero-drift ratio (null={args.null_window_ms:.0f} ms): {ratio_3w}")

    if ratio_2w is None or ratio_3w is None:
        print("[three_window] could not bracket a zero-drift ratio for one of the schemes; "
              "widen --search-duration-s or the candidate ratios in this script.")
        return

    log_2w = run_once(args.ltd_window_ms, 0.0, ratio_2w, args.delta_plus,
                       args.eval_duration_s, args.seed, args.n_io, track_n=args.n_tracked_synapses)
    log_3w = run_once(args.ltd_window_ms, args.null_window_ms, ratio_3w, args.delta_plus,
                       args.eval_duration_s, args.seed, args.n_io, track_n=args.n_tracked_synapses)

    var_2w = log_2w.final_weights.var()                                        # cross-synapse variance, 2-window, run end
    var_3w = log_3w.final_weights.var()                                        # cross-synapse variance, 3-window, run end

    print(f"[three_window] 2-window final cross-synapse weight variance: {var_2w:.6f} "
          f"(std {var_2w ** 0.5:.4f})")
    print(f"[three_window] 3-window final cross-synapse weight variance: {var_3w:.6f} "
          f"(std {var_3w ** 0.5:.4f})")
    print(f"[three_window] variance ratio (3-window / 2-window): {var_3w / var_2w:.3f} "
          "(H3 predicts < 1: a slower random walk. The cost is that the null window blocks "
          "LTP the loop needs, so the 3-window arm sits at a lower CF rate -- the full sweep "
          "measures that trade across five null widths)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.n_tracked_synapses > 0:
        plot_individual_and_mean_weights(
            log_2w, f"2-window: individual vs. mean weight (ratio={ratio_2w:.2f}, "
                       f"cross-synapse var={var_2w:.5f})",
            out_dir / "three_window_2w_weights.png",
        )
        plot_individual_and_mean_weights(
            log_3w, f"3-window: individual vs. mean weight (ratio={ratio_3w:.2f}, "
                       f"null={args.null_window_ms:.0f} ms, cross-synapse var={var_3w:.5f})",
            out_dir / "three_window_3w_weights.png",
        )
        print(f"[three_window] plots written to {out_dir}/three_window_2w_weights.png "
              f"and {out_dir}/three_window_3w_weights.png")


if __name__ == "__main__":
    main()
