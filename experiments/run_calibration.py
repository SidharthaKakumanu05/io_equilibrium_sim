#!/usr/bin/env python3
"""Fit the three synaptic gains, one stage at a time.

The white paper fixes population ratios and the plasticity constants; it says
nothing about membrane models or coupling strengths. Those have to be fitted,
and this script is how the values in config.py were obtained -- so they can be
re-derived after any change to the neuron models rather than being magic numbers.

Each stage is measured in isolation, driven by a surrogate of its input, because
the stages are independent: what PKJ does to DCN does not depend on where the
PKJ spikes came from.

  1. pf_pkj_gain      PKJ alone under PF input at a fixed weight.
                      Target: 50 Hz at weight 0, 70 Hz at weight 1. The span is
                      what matters, not its endpoints: it has to leave the loop
                      room to move in both directions from w_init = 0.5, and it
                      is what makes a weight of 0.5 mean a mid-range Purkinje rate.
  2. pkj_dcn_gain     DCN alone under Poisson PKJ input, with as many converging
                      trains as the real circuit has (12 at CbmSim's ratios) and
                      the same convergence normalization.
                      Target: 15 Hz when those Purkinje cells fire at 60 Hz, i.e.
                      at weight 0.5.
  3. dcn_io_gaba_gain IO alone under Poisson DCN input through the real
                      exponential synapse.
                      Target: 1 Hz CF output at DCN = 15 Hz. Fitted rather than
                      taken from the constant-conductance transfer curve in
                      run_io_characterization.py, because a spike-driven
                      conductance *fluctuates*: the IO escapes during the troughs,
                      so it fires faster than the same mean conductance held steady.

Every stage reads the LIVE config, including the convergence normalization
sim/simulate.py applies, so re-running this after changing a population ratio,
the olive's noise level or a membrane parameter re-derives the right gain rather
than reporting a stale one. It prints the fitted value next to what config.py
currently holds, so a drift between the two is visible at a glance -- it does
not edit config.py.

The IO stage dominates the runtime (CF events are rare, so each probe has to run
long); --skip-io fits the two fast stages alone.

Writes results/calibration_curves.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.io_channels import IOPopulation
from sim.neurons import DCNPopulation, ExpSynapse, PKJPopulation
from sim.poisson_input import generate_pf_spikes


def pkj_rate(cfg, gain, weight, secs, n=64, seed=0):
    """PKJ population driven by PF input at a uniform weight, nothing else."""
    rng = np.random.default_rng(seed)
    pop = PKJPopulation(n, cfg.dt_ms, cfg.pkj.tau_m_ms, cfg.pkj.v_th_mv, cfg.pkj.v_reset_mv,
                         cfg.pkj.e_leak_mv, cfg.pkj.t_ref_ms, cfg.pkj_baseline_hz, cfg.tau_pf_pkj_ms,
                         cfg.tau_pkj_dcn_ms, cfg.e_exc_mv, cfg.e_inh_mv,
                         noise_sigma_mv=cfg.pkj.noise_sigma_mv, rng=np.random.default_rng(seed + 1),
                         pause_g=cfg.cf_pause_g, pause_ms=cfg.cf_pause_ms, e_pause_mv=cfg.e_inh_mv)
    w = np.full((n, cfg.n_pf_per_pkj), weight)
    total = 0
    for _ in range(int(secs * 1000)):
        pop.exc.add(gain * (w * generate_pf_spikes(cfg.pf_rate_hz, cfg.dt_ms, rng, w.shape)).sum(axis=1))
        total += int(pop.step().sum())
    return total / n / secs


def dcn_rate(cfg, gain, pkj_hz, secs, n=64, seed=0):
    """DCN population inhibited by as many independent Poisson PKJ trains as actually
    converge on a nuclear cell (cfg.n_pkj_per_dcn -- 12 at CbmSim's ratios), with the
    same convergence normalization sim/simulate.py applies. Both have to match the
    real circuit, or the fitted gain describes a different one."""
    rng = np.random.default_rng(seed)
    n_pre = cfg.n_pkj_per_dcn
    scale = cfg.pkj_dcn_fitted_at_n_pkj / n_pre
    pop = DCNPopulation(n, cfg.dt_ms, cfg.dcn.tau_m_ms, cfg.dcn.v_th_mv, cfg.dcn.v_reset_mv,
                         cfg.dcn.e_leak_mv, cfg.dcn.t_ref_ms, cfg.dcn_baseline_hz, cfg.tau_pf_pkj_ms,
                         cfg.tau_pkj_dcn_ms, cfg.e_exc_mv, cfg.e_inh_mv,
                         noise_sigma_mv=cfg.dcn.noise_sigma_mv, rng=np.random.default_rng(seed + 1))
    p = pkj_hz * cfg.dt_ms / 1000.0
    total = 0
    for _ in range(int(secs * 1000)):
        pop.inh.add(gain * scale * (rng.random((n, n_pre)) < p).sum(axis=1))
        total += int(pop.step().sum())
    return total / n / secs


def io_rate(cfg, gain, dcn_hz, secs, n=16, seed=0, settle_ms=3000):
    """IO population inhibited by n_dcn_per_io Poisson DCN trains through the
    real exponential GABA synapse. Returns (rate_hz, mean conductance).

    Driven by SPIKES, not by a held conductance, and that is the whole point:
    held constant at the operating point's mean the cell is silent, while the
    same mean delivered as discrete events leaves it firing at ~1 Hz. Fitting
    against the held-constant curve would produce a gain the closed loop never
    sees. See config.dcn_io_gaba_gain.

    Uncoupled: gap junctions change *when* cells fire relative to each other,
    not the population rate this stage is fitting."""
    rng = np.random.default_rng(seed)
    io = IOPopulation(cfg.io_channels, n, cfg.dt_ms, rng=np.random.default_rng(seed + 1), g_gap=None)
    syn = ExpSynapse(n, cfg.tau_dcn_io_ms, cfg.dt_ms, cfg.io_channels.e_gaba)
    p = dcn_hz * cfg.dt_ms / 1000.0
    scale = cfg.dcn_io_fitted_at_n_dcn / cfg.n_dcn_per_io      # as in sim/simulate.py

    def one_step():
        syn.add(gain * scale * (rng.random((n, cfg.n_dcn_per_io)) < p).sum(axis=1))
        fired = io.step(cfg.dt_ms, syn.g)
        g_now = float(syn.g.mean())
        syn.decay()
        return fired, g_now

    for _ in range(settle_ms):
        one_step()
    total, g_sum, steps = 0, 0.0, int(secs * 1000)
    for _ in range(steps):
        fired, g_now = one_step()
        total += int(fired.sum()); g_sum += g_now
    return total / n / secs, g_sum / steps


def bisect(f, target, lo, hi, decreasing=False, iters=16):
    """Bisect f(x) = target on [lo, hi]. `decreasing` for stages where a bigger
    gain means a lower rate (both inhibitory ones)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = (f(mid) > target) if decreasing else (f(mid) < target)
        lo, hi = (mid, hi) if below else (lo, mid)
    return 0.5 * (lo + hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-secs", type=float, default=8.0,
                         help="simulated seconds per bisection probe in stages 1 and 2; "
                              "16 bisection steps each (default: 8)")
    parser.add_argument("--curve-secs", type=float, default=15.0,
                         help="simulated seconds per plotted curve point (default: 15)")
    parser.add_argument("--io-fit-secs", type=float, default=25.0,
                         help="simulated seconds per stage-3 probe. CF events are rare (~1 Hz), so "
                              "this stage needs far longer runs to resolve a rate (default: 25)")
    parser.add_argument("--skip-io", action="store_true",
                         help="fit stages 1 and 2 only. Stage 3 dominates the runtime, and the "
                              "other two gains do not depend on it")
    parser.add_argument("--seed", type=int, default=0,
                         help="seeds the surrogate input trains and the membrane noise (default: 0)")
    parser.add_argument("--out-dir", type=str, default="results",
                         help="directory calibration_curves.png is written to (default: results)")
    args = parser.parse_args()

    cfg = SimConfig()
    print(f"[calib] targets: PKJ 50 Hz @ w=0 and 70 Hz @ w=1; DCN 15 Hz @ PKJ=60 Hz; IO 1 Hz @ DCN=15 Hz")

    # --- stage 1 ---
    g_pf = bisect(lambda g: pkj_rate(cfg, g, 1.0, args.fit_secs, seed=args.seed), 70.0, 0.0, 0.001)
    weights = np.linspace(0.0, 1.0, 5)
    pkj_curve = [pkj_rate(cfg, g_pf, float(w), args.curve_secs, seed=args.seed) for w in weights]
    print(f"[calib] pf_pkj_gain      = {g_pf:.6f}   (config has {cfg.pf_pkj_gain})")
    print("[calib]   PKJ vs weight: " + ", ".join(f"w={w:.2f}->{r:.1f} Hz" for w, r in zip(weights, pkj_curve)))

    # --- stage 2 ---
    g_pd = bisect(lambda g: dcn_rate(cfg, g, 60.0, args.fit_secs, seed=args.seed), 15.0, 0.0, 0.05, decreasing=True)
    pkj_rates = np.linspace(45.0, 75.0, 7)
    dcn_curve = [dcn_rate(cfg, g_pd, float(r), args.curve_secs, seed=args.seed) for r in pkj_rates]
    slope = np.polyfit(pkj_rates, dcn_curve, 1)[0]
    print(f"[calib] pkj_dcn_gain     = {g_pd:.6f}   (config has {cfg.pkj_dcn_gain})")
    print("[calib]   DCN vs PKJ:    " + ", ".join(f"{p:.0f}->{r:.1f} Hz" for p, r in zip(pkj_rates, dcn_curve))
          + f"   slope {slope:+.2f} Hz/Hz")

    io_curve, dcn_rates, g_di = None, None, cfg.dcn_io_gaba_gain
    if not args.skip_io:
        g_di = bisect(lambda g: io_rate(cfg, g, 15.0, args.io_fit_secs, seed=args.seed)[0],
                       1.0, 0.05, 1.5, decreasing=True, iters=10)
        dcn_rates = np.linspace(5.0, 30.0, 6)
        io_curve = [io_rate(cfg, g_di, float(r), args.io_fit_secs, seed=args.seed)[0] for r in dcn_rates]
        print(f"[calib] dcn_io_gaba_gain = {g_di:.6f}   (config has {cfg.dcn_io_gaba_gain})")
        print("[calib]   IO vs DCN:     " + ", ".join(f"{d:.0f}->{r:.2f} Hz" for d, r in zip(dcn_rates, io_curve)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].plot(weights, pkj_curve, "o-", color="tab:blue")
    axes[0].axhline(50, color="gray", ls="--", lw=1); axes[0].axhline(70, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("mean PF$\\to$PKJ weight"); axes[0].set_ylabel("PKJ rate (Hz)")
    axes[0].set_title(f"stage 1: pf_pkj_gain = {g_pf:.6f}")

    axes[1].plot(pkj_rates, dcn_curve, "o-", color="tab:orange")
    axes[1].axhline(15, color="gray", ls="--", lw=1); axes[1].axvline(60, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("PKJ rate (Hz)"); axes[1].set_ylabel("DCN rate (Hz)")
    axes[1].set_title(f"stage 2: pkj_dcn_gain = {g_pd:.5f}  (slope {slope:+.2f})")

    if io_curve is not None:
        axes[2].plot(dcn_rates, io_curve, "o-", color="tab:red")
        axes[2].axhline(1.0, color="gray", ls="--", lw=1); axes[2].axvline(15, color="gray", ls="--", lw=1)
        axes[2].set_title(f"stage 3: dcn_io_gaba_gain = {g_di:.4f}")
    else:
        axes[2].text(0.5, 0.5, "skipped (--skip-io)", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("stage 3: dcn_io_gaba_gain")
    axes[2].set_xlabel("DCN rate (Hz)"); axes[2].set_ylabel("IO CF rate (Hz)")

    fig.suptitle("Stage-by-stage gain calibration -- each stage driven by a surrogate of its input", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "calibration_curves.png", dpi=150)
    plt.close(fig)
    print(f"[calib] plot written to {out_dir}/calibration_curves.png")


if __name__ == "__main__":
    main()
