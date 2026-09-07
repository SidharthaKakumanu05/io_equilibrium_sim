#!/usr/bin/env python3
"""Experiment 3: delta+/delta- ratio sweep vs. LTD window length (H2).

NOT RE-VALIDATED since PKJ and DCN became spiking integrate-and-fire cells. It
runs, and the analytical prediction it is compared against is unchanged (it
depends only on the LTD window and the equilibrium CF interval, not on the
membrane models), but the measured curve has not been re-checked against it
under the new neuron models and re-fitted gains.

Pinned to a single closed-loop group: this experiment is about the plasticity
rule, not the network, and it runs one simulation per candidate ratio per
window -- dozens in all.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import plot_ratio_sweep, predicted_ltd_ltp_ratio, weight_drift_slope
from sim.simulate import Simulation


def drift_slope_for_ratio(ltd_window_ms, ratio, delta_plus, duration_s, seed):
    cfg = SimConfig(
        n_io=1,                                                              # one group: this is a plasticity experiment
        duration_s=duration_s, seed=seed,
        ltd_window_ms=ltd_window_ms,
        delta_plus=delta_plus, delta_minus=delta_plus * ratio,               # delta_minus derived from the candidate ratio
    )
    return weight_drift_slope(Simulation(cfg).run())


def find_zero_drift_ratio(ltd_window_ms, ratio_candidates, delta_plus, duration_s, seed):
    slopes = [drift_slope_for_ratio(ltd_window_ms, r, delta_plus, duration_s, seed)
              for r in ratio_candidates]                                       # one run per candidate ratio
    zero_ratio = None
    for i in range(len(ratio_candidates) - 1):
        s0, s1 = slopes[i], slopes[i + 1]
        if s0 == 0:
            zero_ratio = ratio_candidates[i]                                   # exact hit (rare)
            break
        if (s0 > 0) != (s1 > 0):                                               # sign change between consecutive candidates
            r0, r1 = ratio_candidates[i], ratio_candidates[i + 1]
            zero_ratio = r0 + (0 - s0) * (r1 - r0) / (s1 - s0)                 # linear interpolation to slope==0
            break
    return zero_ratio, slopes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=float, nargs="+", default=[50.0, 100.0, 200.0],
                         help="LTD window lengths (ms) to sweep")
    parser.add_argument("--n-ratios", type=int, default=6, help="ratio points per window")
    parser.add_argument("--duration-s", type=float, default=30.0, help="sim duration per run (s)")
    parser.add_argument("--delta-plus", type=float, default=0.001, help="fixed LTP magnitude")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    empirical, analytical = [], []                                             # one entry per window, for the summary curve
    first_window_slopes = None                                                 # kept for the diagnostic per-ratio plot
    first_window_ratios = None

    print(f"{'window_ms':>10} {'analytical_ratio':>17} {'empirical_ratio':>16}")
    for w in args.windows:
        predicted = predicted_ltd_ltp_ratio(w)                                 # analytical target for this window
        fractions = np.linspace(0.2, 2.2, args.n_ratios)                       # bracket the prediction on both sides
        candidates = sorted(set(max(0.05, round(predicted * f, 3)) for f in fractions))
        zero_ratio, slopes = find_zero_drift_ratio(w, candidates, args.delta_plus,
                                                     args.duration_s, args.seed)
        empirical.append(zero_ratio)
        analytical.append(predicted)
        print(f"{w:>10.1f} {predicted:>17.3f} "
              f"{'n/a (no sign change)' if zero_ratio is None else f'{zero_ratio:>16.3f}'}")

        if first_window_slopes is None:
            first_window_slopes, first_window_ratios = slopes, candidates      # only keep the first window's raw curve

    plot_path = out_dir / "window_sweep_ratio_curve.png"
    plot_ratio_sweep(first_window_ratios, first_window_slopes, predicted_ltd_ltp_ratio(args.windows[0]),
                      f"Drift slope vs. ratio @ LTD window={args.windows[0]:.0f} ms", plot_path)
    print(f"[window_sweep] slope-vs-ratio plot written to {plot_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(args.windows, analytical, "o--", label="analytical", color="tab:green")   # theory curve
    valid = [(w, e) for w, e in zip(args.windows, empirical) if e is not None]         # drop windows with no zero-crossing
    if valid:
        ax.plot([w for w, _ in valid], [e for _, e in valid], "o-", label="empirical", color="tab:blue")
    ax.set_xlabel("LTD window (ms)")
    ax.set_ylabel("zero-drift delta_minus/delta_plus ratio")
    ax.set_title("H2: LTD-window length vs. required delta-/delta+ ratio")
    ax.legend()
    fig.tight_layout()
    curve_path = out_dir / "window_sweep_ratio_vs_window.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[window_sweep] ratio-vs-window plot written to {curve_path}")


if __name__ == "__main__":
    main()
