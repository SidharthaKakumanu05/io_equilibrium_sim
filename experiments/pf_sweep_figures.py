#!/usr/bin/env python3
"""The parallel-fiber pool-size result, and the cross-condition comparison.

Kept separate from make_long_figures.py because these read MANY bundles rather
than one, and the question they answer is different: make_long_figures asks
"what did this run do", these ask "what does the knob do".

The pool-size claim, stated as precisely as the sweep can state it
-------------------------------------------------------------------
A Purkinje cell's excitatory drive is `n_pf x pf_rate x w x gain`, and the only
term in it the loop can move is `w`. So the loop's setpoint is the PRODUCT
`n_pf x w`, not `n_pf`, and the number of parallel fibers is absorbed by a
compensating change in weight:

    equilibrium weight   is proportional to   1 / n_pf
    equilibrium CF rate  is independent of    n_pf

up to the point where the `w` the compensation calls for leaves [w_min, w_max].
Below that pool size the cell cannot be driven hard enough even with every
synapse at the ceiling; above it the weights approach the floor and the same
argument runs out from the other side. Only in those two pinned regimes does
pool size control the firing rate -- and it controls it by breaking the
regulator, not by setting its setpoint.

That distinction is the whole result: "how many PF" is a statement about where
the loop sits in weight space and how much headroom it has left, and the rate
it settles at is set by the plasticity window instead (`rate = 1000 / (window x
(1 + delta-/delta+))`, the README's H2). This sweep measures where the
invariance holds and where it fails, at 900 s x 3 seeds x 13 pool sizes -- 10x
the length of the 90 s sweep already in the README, over a grid that brackets
both failures instead of ending at them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.analysis import _mpl
from sim.recording import RunBundle


def bundle_stats(bundle, tail_frac=0.5):
    """The scalars a sweep point is summarized by, all measured on the last
    `tail_frac` of the run so the weight transient is excluded."""
    m = bundle.meta
    lo = m["duration_s"] * (1.0 - tail_frac)
    dur = m["duration_s"] - lo
    s = bundle.slow
    t_s, w = s["t_s"], s["mean_weight"].mean(axis=1).astype(np.float64)
    sel = t_s >= lo
    slope, _ = np.polyfit(t_s[sel], w[sel], 1)
    final = np.asarray(bundle.final["final_weights"])
    w_eq = float(w[sel].mean())
    return {
        "n_pf_per_pkj": m["n_pf_per_pkj"], "seed": m["seed"], "duration_s": m["duration_s"],
        "io_rate_hz": float(bundle.io.per_cell_counts(lo, m["duration_s"]).sum() / dur / m["n_io"]),
        "pkj_rate_hz": float(bundle.pkj.per_cell_counts(lo, m["duration_s"]).sum() / dur / m["n_pkj"]),
        "dcn_rate_hz": float(bundle.dcn.per_cell_counts(lo, m["duration_s"]).sum() / dur / m["n_dcn"]),
        "mean_weight": w_eq,
        "weight_drift_per_hour": float(slope * 3600),
        "cross_synapse_sd": float(final.std()),
        # Total excitatory drive one Purkinje cell receives, up to the fixed gain: n_pf fibers x
        # pf_rate spikes/s x mean weight. This is the quantity the loop holds constant by moving w,
        # and the reason the rate can be invariant while the weight is not.
        "pf_drive": float(m["n_pf_per_pkj"] * m["pf_rate_hz"] * w_eq),
        "saturated_high": bool(w_eq > 0.98), "saturated_low": bool(w_eq < 0.02),
        "io_isi_cv": float(np.nanmean(bundle.io.isi_cv(lo, m["duration_s"])[1])),
        "ltd_window_ms": m["ltd_window_ms"], "delta_ratio": m["delta_minus"] / m["delta_plus"],
    }


def pf_sweep(sweep_root, path, json_path=None):
    """Six panels: the invariance, the 1/n law behind it, the conserved product,
    the rest of the loop, the cost in weight spread, and whether each point
    actually settled. See the module docstring for what they add up to."""
    plt = _mpl()
    rows = [bundle_stats(RunBundle(d)) for d in sorted(Path(sweep_root).glob("pf*_s*"))
            if (d / "meta.json").exists()]
    if not rows:
        raise SystemExit(f"no finished bundles under {sweep_root}")

    ns = np.array(sorted({r["n_pf_per_pkj"] for r in rows}))

    def agg(key):
        mu = np.array([np.mean([r[key] for r in rows if r["n_pf_per_pkj"] == n]) for n in ns])
        sd = np.array([np.std([r[key] for r in rows if r["n_pf_per_pkj"] == n]) for n in ns])
        return mu, sd

    io_mu, io_sd = agg("io_rate_hz")
    w_mu, w_sd = agg("mean_weight")
    pkj_mu, pkj_sd = agg("pkj_rate_hz")
    dcn_mu, _ = agg("dcn_rate_hz")
    drive_mu, drive_sd = agg("pf_drive")
    sd_mu, _ = agg("cross_synapse_sd")
    drift_mu, drift_sd = agg("weight_drift_per_hour")
    sat_hi = np.array([any(r["saturated_high"] for r in rows if r["n_pf_per_pkj"] == n) for n in ns])
    sat_lo = np.array([any(r["saturated_low"] for r in rows if r["n_pf_per_pkj"] == n) for n in ns])

    r0 = rows[0]
    predicted = 1000.0 / (r0["ltd_window_ms"] * (1.0 + r0["delta_ratio"]))
    # The regulable band: pool sizes whose equilibrium weight is strictly inside the bounds, so
    # the loop still has a setpoint to move to. Everything outside is pinned.
    ok = ~(sat_hi | sat_lo) & (w_mu > 0.05) & (w_mu < 0.95)
    band = (ns[ok].min(), ns[ok].max()) if ok.any() else (np.nan, np.nan)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    def shade(ax):
        pinned = ~ok
        for i in np.flatnonzero(pinned):
            ax.axvspan(ns[i] / 1.06, ns[i] * 1.06, color="tab:red", alpha=0.13,
                       label="_pinned" if i else "weights pinned at a bound")
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=7, rotation=45)
        ax.set_xlabel("parallel fibers per Purkinje cell")
        ax.grid(alpha=0.25)

    # (a) the claim under test: does pool size move the rate?
    axes[0, 0].errorbar(ns, io_mu, yerr=io_sd, fmt="o-", color="tab:red", capsize=3)
    axes[0, 0].axhline(predicted, color="tab:green", linestyle="--",
                       label=f"H2 prediction {predicted:.2f} Hz")
    axes[0, 0].set_ylabel("equilibrium CF rate (Hz)")
    axes[0, 0].set_title("(a) RATE is flat across the regulable band\n"
                         f"{band[0]:.0f}-{band[1]:.0f} fibers: "
                         f"{io_mu[ok].mean():.3f} $\\pm$ {io_mu[ok].std():.3f} Hz")
    shade(axes[0, 0])
    axes[0, 0].legend(fontsize=8)

    # (b) what pool size does set
    axes[0, 1].errorbar(ns, w_mu, yerr=w_sd, fmt="o-", color="tab:blue", capsize=3, label="measured")
    ref_n = ns[ok][len(ns[ok]) // 2] if ok.any() else ns[len(ns) // 2]
    ref_w = w_mu[list(ns).index(ref_n)]
    axes[0, 1].plot(ns, np.clip(ref_w * ref_n / ns, 0, 1), "--", color="gray",
                    label="$\\propto 1/n_{PF}$, clipped")
    axes[0, 1].axhline(1.0, color="tab:red", linestyle=":", label="$w_{max}$ / $w_{min}$")
    axes[0, 1].axhline(0.0, color="tab:red", linestyle=":")
    axes[0, 1].set_ylabel("equilibrium mean weight")
    axes[0, 1].set_title("(b) WEIGHT is what pool size sets,\nand it follows $1/n_{PF}$")
    shade(axes[0, 1])
    axes[0, 1].legend(fontsize=8)

    # (c) the conserved quantity
    cv = drive_mu[ok].std() / drive_mu[ok].mean() * 100 if ok.any() else np.nan
    axes[0, 2].errorbar(ns, drive_mu, yerr=drive_sd, fmt="o-", color="tab:purple", capsize=3)
    if ok.any():
        axes[0, 2].axhline(drive_mu[ok].mean(), color="black", linestyle="--",
                           label="mean over the band")
    axes[0, 2].set_ylabel("$n_{PF}\\times$ rate $\\times\\ w$   (drive, a.u.)")
    axes[0, 2].set_title(f"(c) the PRODUCT is the controlled variable\n"
                         f"CV across the band: {cv:.1f}%")
    shade(axes[0, 2])
    axes[0, 2].legend(fontsize=8)

    # (d) the rest of the loop inherits the same invariance
    axes[1, 0].errorbar(ns, pkj_mu, yerr=pkj_sd, fmt="o-", color="tab:blue", capsize=3, label="PKJ")
    axes[1, 0].errorbar(ns, dcn_mu, fmt="s-", color="tab:orange", capsize=3, label="DCN")
    axes[1, 0].set_ylabel("firing rate (Hz)")
    axes[1, 0].set_title("(d) PKJ and DCN rates inherit it")
    shade(axes[1, 0])
    axes[1, 0].legend(fontsize=8)

    # (e) what a bigger pool costs
    axes[1, 1].plot(ns, sd_mu, "o-", color="tab:green")
    axes[1, 1].set_ylabel("cross-synapse SD of $w$")
    axes[1, 1].set_title("(e) the spread the weights walk over\nshrinks with the weight itself")
    shade(axes[1, 1])

    # (f) did each point settle at all?
    axes[1, 2].errorbar(ns, drift_mu, yerr=drift_sd, fmt="o-", color="tab:brown", capsize=3)
    axes[1, 2].axhline(0, color="gray", linewidth=1)
    axes[1, 2].set_ylabel("weight drift (units/hour, 2nd half)")
    axes[1, 2].set_title("(f) residual drift")
    shade(axes[1, 2])

    n_seeds = len(rows) // max(len(ns), 1)
    fig.suptitle("Parallel-fiber pool size sets the equilibrium WEIGHT, not the equilibrium RATE"
                 f"   --   {r0['duration_s']:.0f} s $\\times$ {n_seeds} seeds "
                 f"$\\times$ {len(ns)} pool sizes"
                 "\nred: the loop is pinned against a weight bound, so it has no setpoint left "
                 "and the rate finally does depend on pool size", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=150)
    plt.close(fig)

    out = {"rows": rows, "n_pf": ns.tolist(), "io_rate_mean": io_mu.tolist(),
           "io_rate_sd": io_sd.tolist(), "mean_weight": w_mu.tolist(),
           "mean_weight_sd": w_sd.tolist(), "pkj_rate": pkj_mu.tolist(),
           "dcn_rate": dcn_mu.tolist(), "drive": drive_mu.tolist(),
           "cross_synapse_sd": sd_mu.tolist(), "drift_per_hour": drift_mu.tolist(),
           "pinned": (~ok).tolist(),
           "regulable_band": [float(band[0]), float(band[1])],
           "rate_in_band_mean": float(io_mu[ok].mean()) if ok.any() else None,
           "rate_in_band_sd": float(io_mu[ok].std()) if ok.any() else None,
           "drive_cv_percent_in_band": float(cv) if ok.any() else None,
           "predicted_hz": predicted}
    if json_path:
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def pf_sweep_dynamics(sweep_root, path):
    """The weight trajectory and CF rate at every pool size, on two axes.

    This is the mechanism as a movie rather than as endpoints: every pool size
    starts from the same w_init = 0.5 and walks to its own equilibrium weight,
    while the CF rate on the right is already at 1 Hz throughout. The two arms
    that walk into a bound are the ones whose rate then leaves 1 Hz."""
    plt = _mpl()
    dirs = [d for d in Path(sweep_root).glob("pf*_s0") if (d / "meta.json").exists()]
    if not dirs:
        raise SystemExit(f"no finished bundles under {sweep_root}")
    dirs.sort(key=lambda d: RunBundle(d).meta["n_pf_per_pkj"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(dirs):
        b = RunBundle(d)
        s, n = b.slow, b.meta["n_pf_per_pkj"]
        colour = cmap(i / max(len(dirs) - 1, 1))
        axes[0].plot(s["t_s"], s["mean_weight"].mean(axis=1), color=colour, linewidth=1.2, label=f"{n}")
        t, r = b.io.rate_trace(30.0, b.meta["duration_s"])
        axes[1].plot(t, r, color=colour, linewidth=0.9)
    for y in (0.0, 1.0):
        axes[0].axhline(y, color="tab:red", linestyle=":", linewidth=1)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("mean PF$\\to$PKJ weight")
    axes[0].set_title("weight settles to a different place for each pool size")
    axes[0].legend(fontsize=7, ncol=2, title="fibers/PKJ", title_fontsize=7)
    axes[0].grid(alpha=0.25)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="1 Hz")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("CF rate (Hz, 30 s bins)")
    axes[1].set_title("...to the same rate, except where a bound was hit")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Pool-size sweep, seed 0: the compensation in progress", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_conditions(bundle_dirs, path, json_path=None):
    """The 5-hour conditions side by side.

    baseline / cf_inh / gap_off / open_loop each change one limb, so the
    comparison reads as what that limb contributes. open_loop is the one that
    should break: cut the nucleo-olivary projection and there is no negative
    feedback left to settle against."""
    plt = _mpl()
    bundles = [RunBundle(d) for d in bundle_dirs if (Path(d) / "meta.json").exists()]
    if not bundles:
        raise SystemExit("no finished bundles to compare")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    cmap = plt.get_cmap("tab10")
    stats = []
    for i, b in enumerate(bundles):
        m, s = b.meta, b.slow
        name, colour = m["label"].split("/")[0], cmap(i)
        w = s["mean_weight"].mean(axis=1)
        hw = len(w) // 2
        t, r = b.io.rate_trace(60.0, m["duration_s"])
        half = len(r) // 2
        final = np.asarray(b.final["final_weights"]).ravel()
        axes[0, 0].plot(s["t_s"] / 3600.0, w, color=colour, linewidth=1.0, label=name)
        axes[0, 1].plot(t / 3600.0, r, color=colour, linewidth=0.8, label=name)
        axes[1, 0].hist(final, bins=np.linspace(0, 1, 101), histtype="step", density=True,
                        color=colour, linewidth=1.4, label=name)
        axes[1, 1].plot(s["t_s"] / 3600.0, s["sample_weights"].std(axis=1), color=colour,
                        linewidth=1.0, label=name)
        stats.append({"condition": name, "duration_s": m["duration_s"],
                      "cf_rate_hz": float(r[half:].mean()),
                      "mean_weight": float(w[hw:].mean()),
                      "cross_synapse_sd": float(final.std()),
                      "weight_saturated": bool(w[hw:].mean() > 0.98 or w[hw:].mean() < 0.02),
                      "pkj_rate_hz": float(
                          b.pkj.per_cell_counts(m["duration_s"] / 2, m["duration_s"]).sum()
                          / (m["duration_s"] / 2) / m["n_pkj"])})
    axes[0, 0].set_ylabel("mean PF$\\to$PKJ weight")
    axes[0, 0].set_title("(a) weight settling")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("CF rate (Hz, 60 s bins)")
    axes[0, 1].set_title("(b) climbing-fiber rate")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_xlabel("time (hours)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("final PF$\\to$PKJ weight")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title("(c) final weight distribution, all 160,000 synapses")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_ylabel("cross-synapse SD")
    axes[1, 1].set_title("(d) spread of the tracked ensemble")
    fig.suptitle("Five simulated hours, one limb changed at a time", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if json_path:
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pf-sweep", default=None, help="directory holding the pf<N>_s<seed> bundles")
    p.add_argument("--compare", nargs="*", default=[], help="bundles for the cross-condition figure")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.pf_sweep:
        r = pf_sweep(args.pf_sweep, out / "pf_sweep.png", out / "pf_sweep.json")
        pf_sweep_dynamics(args.pf_sweep, out / "pf_sweep_dynamics.png")
        print(f"[pf] regulable band {r['regulable_band'][0]:.0f}-{r['regulable_band'][1]:.0f} fibers; "
              f"rate there {r['rate_in_band_mean']:.4f} +/- {r['rate_in_band_sd']:.4f} Hz "
              f"(predicted {r['predicted_hz']:.3f}); drive CV {r['drive_cv_percent_in_band']:.1f}%")
    if args.compare:
        st = compare_conditions(args.compare, out / "compare_conditions.png",
                                out / "compare_conditions.json")
        for s in st:
            print(f"[compare] {s['condition']:<10} CF {s['cf_rate_hz']:.4f} Hz | "
                  f"w {s['mean_weight']:.4f} | PKJ {s['pkj_rate_hz']:.2f} Hz | "
                  f"saturated {s['weight_saturated']}")


if __name__ == "__main__":
    main()
