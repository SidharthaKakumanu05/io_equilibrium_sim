#!/usr/bin/env python3
"""Experiment 4 (stretch): three-window LTD/null/LTP variant (H3).

NOT RE-VALIDATED since PKJ and DCN became spiking integrate-and-fire cells. It
runs, but whether the three-window scheme still lowers the equilibrium
cross-synapse variance has not been re-measured under the new neuron models and
re-fitted gains. Pinned to a single closed-loop group, like the H2 sweep.

Metric note: "forgetting" is compared via the FINAL CROSS-SYNAPSE weight
variance (spread across all individual PF->PKJ synapses at run end), not
the variance of the mean weight over time -- the mean, averaged across
~4000 synapses, is crushed toward zero variance by the law of large
numbers regardless of window scheme, so it can't distinguish the two.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import plot_individual_and_mean_weights, predicted_ltd_ltp_ratio, weight_drift_slope
from sim.simulate import Simulation


def run_once(ltd_window_ms, null_window_ms, ratio, delta_plus, duration_s, seed, track_n=0):
    cfg = SimConfig(
        n_io=1,                                                               # one group: this is a plasticity experiment
        duration_s=duration_s, seed=seed,
        ltd_window_ms=ltd_window_ms, null_window_ms=null_window_ms,           # null_window_ms=0 -> 2-window mode
        delta_plus=delta_plus, delta_minus=delta_plus * ratio,
        n_tracked_synapses=track_n,
    )
    return Simulation(cfg).run()


def drift_slope_for_ratio(ltd_window_ms, null_window_ms, ratio, delta_plus, duration_s, seed):
    log = run_once(ltd_window_ms, null_window_ms, ratio, delta_plus, duration_s, seed)
    return log, weight_drift_slope(log)


def find_zero_drift_ratio(ltd_window_ms, null_window_ms, candidates, delta_plus, duration_s, seed):
    slopes = []
    for r in candidates:
        _, slope = drift_slope_for_ratio(ltd_window_ms, null_window_ms, r, delta_plus, duration_s, seed)
        slopes.append(slope)
    for i in range(len(candidates) - 1):                                       # same bracket-and-interpolate as run_window_sweep.py
        s0, s1 = slopes[i], slopes[i + 1]
        if s0 == 0:
            return candidates[i]
        if (s0 > 0) != (s1 > 0):
            r0, r1 = candidates[i], candidates[i + 1]
            return r0 + (0 - s0) * (r1 - r0) / (s1 - s0)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltd-window-ms", type=float, default=100.0)
    parser.add_argument("--null-window-ms", type=float, default=100.0)
    parser.add_argument("--delta-plus", type=float, default=0.001)
    parser.add_argument("--search-duration-s", type=float, default=30.0,
                         help="sim duration used while searching for the zero-drift ratio")
    parser.add_argument("--eval-duration-s", type=float, default=90.0,
                         help="sim duration for the final equilibrium-variance comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--n-tracked-synapses", type=int, default=15)
    args = parser.parse_args()

    predicted_2w = predicted_ltd_ltp_ratio(args.ltd_window_ms)                 # same candidate bracket for both schemes
    candidates = sorted(set(
        max(0.05, round(predicted_2w * f, 3)) for f in [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
    ))

    ratio_2w = find_zero_drift_ratio(args.ltd_window_ms, 0.0, candidates, args.delta_plus,
                                      args.search_duration_s, args.seed)        # null_window_ms=0 -> 2-window
    ratio_3w = find_zero_drift_ratio(args.ltd_window_ms, args.null_window_ms, candidates, args.delta_plus,
                                      args.search_duration_s, args.seed)        # 3-window

    print(f"[three_window] 2-window zero-drift ratio: {ratio_2w}")
    print(f"[three_window] 3-window zero-drift ratio (null={args.null_window_ms:.0f} ms): {ratio_3w}")

    if ratio_2w is None or ratio_3w is None:
        print("[three_window] could not bracket a zero-drift ratio for one of the schemes; "
              "widen --search-duration-s or the candidate ratios in this script.")
        return

    log_2w = run_once(args.ltd_window_ms, 0.0, ratio_2w, args.delta_plus,
                       args.eval_duration_s, args.seed, track_n=args.n_tracked_synapses)
    log_3w = run_once(args.ltd_window_ms, args.null_window_ms, ratio_3w, args.delta_plus,
                       args.eval_duration_s, args.seed, track_n=args.n_tracked_synapses)

    var_2w = log_2w.final_weights.var()                                        # cross-synapse variance, 2-window, run end
    var_3w = log_3w.final_weights.var()                                        # cross-synapse variance, 3-window, run end

    print(f"[three_window] 2-window final cross-synapse weight variance: {var_2w:.6f} "
          f"(std {var_2w ** 0.5:.4f})")
    print(f"[three_window] 3-window final cross-synapse weight variance: {var_3w:.6f} "
          f"(std {var_3w ** 0.5:.4f})")
    print(f"[three_window] variance ratio (3-window / 2-window): {var_3w / var_2w:.3f} "
          "(H3 predicts < 1: slower random-walk 'forgetting', at the cost of "
          "ratio_3w being closer to 1 than ratio_2w and a potential extinction penalty "
          "not measured by this MVP script)")

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
