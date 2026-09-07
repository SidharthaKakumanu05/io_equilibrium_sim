#!/usr/bin/env python3
"""H2, the direct form: which delta-/delta+ ratio makes the weights stop drifting?

H2 says the loop settles where LTD and LTP balance, at an inter-CF interval of
`window x (1 + delta-/delta+)`. Turned around, a given LTD window has exactly one
ratio at which the weights neither climb nor fall at 1 Hz, and it is
`(1000 - window) / window` -- `sim.analysis.predicted_ltd_ltp_ratio`. This script
measures that ratio instead of assuming it: for each LTD window it runs the loop
at several candidate ratios, brackets the sign change in the weight-drift slope,
and interpolates to zero.

That is one simulation per candidate ratio per window -- 18 by default, so
budget accordingly (see --n-io). The replicated, full-scale version of H2 is
`experiments/sweep.py h2_window`, which tests the same claim the other way
round: fix the ratio, predict the rate, and check eight settings against
prediction over ten seeds each. Prefer that one for a result to quote; this
script is the cheap, direct check of the same relationship.

Writes results/window_sweep_ratio_curve.png (drift vs. ratio at the first
window) and results/window_sweep_ratio_vs_window.png (fitted vs. analytical).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import plot_ratio_sweep, predicted_ltd_ltp_ratio, weight_drift_slope
from sim.simulate import Simulation


def microzone(n_io, **overrides):
    """A valid microzone scaled to `n_io` olivary cells.

    The connection counts have to close, and that fixes everything else:
    n_pkj = 8*n_io, and n_pkj * n_dcn_per_pkj == n_dcn * n_pkj_per_dcn forces
    n_dcn = 2*n_io. n_dcn_per_io is capped at n_dcn, which at small n_io makes
    DCN->IO complete -- CbmSim's own case, and harmless here: this experiment
    measures the plasticity rule's balance point, which is a property of the
    weights and the CF rate, not of how the olive's cells differ from each
    other. sim/connectivity.py raises on any combination that does not close,
    so a bad --n-io fails immediately rather than running something else."""
    cfg = SimConfig(**overrides)
    cfg.n_io = n_io
    cfg.n_dcn = 2 * n_io
    cfg.n_dcn_per_io = min(cfg.n_dcn_per_io, cfg.n_dcn)
    return cfg


def drift_slope_for_ratio(ltd_window_ms, ratio, delta_plus, duration_s, seed, n_io):
    cfg = microzone(
        n_io,
        duration_s=duration_s, seed=seed,
        ltd_window_ms=ltd_window_ms,
        delta_plus=delta_plus, delta_minus=delta_plus * ratio,               # delta_minus derived from the candidate ratio
    )
    return weight_drift_slope(Simulation(cfg).run())


def find_zero_drift_ratio(ltd_window_ms, ratio_candidates, delta_plus, duration_s, seed, n_io):
    """Bracket the ratio at which the drift slope crosses zero, then interpolate.
    Returns (ratio or None, slopes). None means no sign change was bracketed --
    every candidate drifted the same way, so widen the candidate range or run
    longer."""
    slopes = [drift_slope_for_ratio(ltd_window_ms, r, delta_plus, duration_s, seed, n_io)
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
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=float, nargs="+", default=[50.0, 100.0, 200.0],
                         help="LTD window lengths (ms) to sweep; one fit per window "
                              "(default: 50 100 200)")
    parser.add_argument("--n-ratios", type=int, default=6,
                         help="candidate ratios per window, spread from 0.2x to 2.2x the "
                              "analytical prediction. One simulation each (default: 6)")
    parser.add_argument("--duration-s", type=float, default=30.0,
                         help="simulated seconds per candidate run. Short on purpose -- the drift "
                              "SIGN is what gets bracketed, not the settled value (default: 30)")
    parser.add_argument("--delta-plus", type=float, default=0.001,
                         help="LTP increment, held fixed while delta_minus is varied (default: 0.001)")
    parser.add_argument("--n-io", type=int, default=4,
                         help="olivary cells; n_dcn follows as 2*n_io and the whole microzone "
                              "scales with it. 4 is CbmSim's own size and keeps the default "
                              "sweep to roughly half an hour; 40 is the shipped network and is "
                              "about 10x that (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                         help="seeds the PF Poisson draws and the membrane noise (default: 0)")
    parser.add_argument("--out-dir", type=str, default="results",
                         help="directory the two PNGs are written to (default: results)")
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
                                                     args.duration_s, args.seed, args.n_io)
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
