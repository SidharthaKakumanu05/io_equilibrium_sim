#!/usr/bin/env python3
"""Every figure the long runs support, made from a saved RunBundle.

Split out from the simulation on purpose: the runs this reads take half a day,
and a plotting bug should not cost one. Re-run this as often as you like; it
touches nothing in the bundle.

Where a figure already exists in sim/analysis.py it is REUSED rather than
reimplemented -- `_as_simlog` rehydrates a window of a bundle into the SimLog
shape those functions already take, so a raster cut out of hour three of a
5-hour run is drawn by exactly the code that draws the 60 s baseline raster.
The functions below are the ones that only make sense at length:

  rates_full           population rates across the whole run, 10 s bins -- the
                       transient and the equilibrium in one axis
  raster_io_full       every climbing-fiber event of the whole run. The only
                       population sparse enough to draw whole (~700k events);
                       PKJ at 60 Hz would be 346 million
  weight_population    the 10,000-synapse ensemble as a density over time,
                       against the mean the loop actually regulates
  weight_diffusion     cross-synapse spread, mean squared displacement and
                       autocorrelation -- the random walk H3 is about, which a
                       60 s run cannot resolve and a 5-hour one can
  io_stats             per-cell CF rate and ISI structure over the full run
  io_synchrony         pairwise CF synchrony against gap-junction distance
  equilibrium          the headline panel: is it actually settled, fitted on
                       the second half only

Cross-run figures -- the pool-size sweep and the condition comparison -- live in
experiments/pf_sweep_figures.py instead, because they read many bundles and ask
a different question.

Usage:
    make_long_figures.py --bundle DIR [DIR ...] --out-dir DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.analysis import (GROUP_CMAP, _mpl, plot_io_state, plot_rasters, plot_voltage_traces,
                          plot_weights)
from sim.recording import RunBundle
from sim.simulate import SimLog


# --- rehydration: a bundle window in the shape sim/analysis.py already plots ---

def _as_simlog(bundle, t0_s, t1_s, trace_idx=None):
    """A SimLog holding one window of a bundle, so the existing plotting code
    can draw it. Spike trains are cut with two searchsorted calls per
    population, so this costs the window and not the run."""
    m, f = bundle.meta, bundle.final
    log = SimLog()
    log.io_spikes = bundle.io.trains(t0_s, t1_s)
    log.pkj_spikes = bundle.pkj.trains(t0_s, t1_s)
    log.dcn_spikes = bundle.dcn.trains(t0_s, t1_s)
    log.pf_spikes = bundle.pf.trains(t0_s, t1_s)
    log.pf_recorded = f["pf_recorded"]
    log.io_of_pkj = f["io_of_pkj"]
    log.group_of_pkj = f["io_of_pkj"]
    log.group_of_dcn = f["group_of_dcn"]
    log.final_weights = f["final_weights"]
    log.gap_matrix = f["gap_matrix"]
    log.duration_ms = t1_s * 1000.0
    log.meta = {"n_io": m["n_io"], "n_dcn": m["n_dcn"], "n_pkj": m["n_pkj"],
                "n_pf_per_pkj": m["n_pf_per_pkj"], "duration_s": m["duration_s"]}

    s = bundle.slow
    log.t_ms = s["t_s"] * 1000.0
    log.mean_weight = s["mean_weight"]
    log.pkj_mean_weight = s["pkj_mean_weight"]
    log.sample_weights = s["sample_weights"]
    log.tracked_synapses = s["tracked_synapses"]

    if trace_idx is not None:
        tr = bundle.trace(trace_idx)
        log.trace_t_ms = tr["t_ms"]
        log.trace_io_v, log.trace_io_ca, log.trace_io_gaba = tr["io_v"], tr["io_ca"], tr["io_gaba"]
        log.trace_pkj_v, log.trace_dcn_v = tr["pkj_v"], tr["dcn_v"]
    return log


def _title(bundle, extra=""):
    m = bundle.meta
    bits = [m["label"].split("/")[0]]
    if m["n_pf_per_pkj"] != 500:
        bits.append(f"{m['n_pf_per_pkj']} PF/PKJ")
    if m["gap_g"] == 0:
        bits.append("gap off")
    if m["ablate_dcn_io"]:
        bits.append("DCN$\\to$IO ablated")
    if m["cf_pkj_reversal_mv"] < 0:
        bits.append(f"CF reversal {m['cf_pkj_reversal_mv']:.0f} mV")
    head = ", ".join(bits) + f"  --  {m['duration_s'] / 3600:.2f} simulated hours"
    return head + (f"\n{extra}" if extra else "")


# --- figures that only make sense at length -------------------------------

def rates_full(bundle, path, bin_s=10.0):
    """Population rate of every cell type across the whole run, with the mean
    weight on a twin axis. This is the figure that separates the transient from
    the equilibrium: the weights fall from w_init for the first few hundred
    seconds while the CF rate is already near 1 Hz, and both then sit flat for
    hours."""
    plt = _mpl()
    m = bundle.meta
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, store, name, colour in ((axes[0], bundle.pkj, "PKJ", "tab:blue"),
                                    (axes[1], bundle.dcn, "DCN", "tab:orange"),
                                    (axes[2], bundle.io, "IO / CF", "tab:red")):
        t, r = store.rate_trace(bin_s, m["duration_s"])
        ax.plot(t / 3600.0, r, color=colour, linewidth=0.7)
        # The equilibrium value, measured on the second half only -- the first half
        # still contains the weight transient, and averaging over it would bias this low.
        half = len(r) // 2
        ax.axhline(r[half:].mean(), color="black", linestyle="--", linewidth=1.2,
                   label=f"2nd-half mean {r[half:].mean():.3f} Hz")
        ax.set_ylabel(f"{name}\n(Hz/cell)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

    s = bundle.slow
    w = s["mean_weight"].mean(axis=1)
    axes[3].plot(s["t_s"] / 3600.0, w, color="black", linewidth=1.0)
    axes[3].set_ylabel("mean PF$\\to$PKJ\nweight", fontsize=9)
    axes[3].set_xlabel("time (hours)")
    axes[3].set_ylim(0, max(1.0, w.max() * 1.1))

    axes[0].set_title(_title(bundle, f"population rates, {bin_s:.0f} s bins"))
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def raster_io_full(bundle, path):
    """Every climbing-fiber event in the run, all 40 cells, plus the
    instantaneous population CF rate underneath.

    The olive is the one population sparse enough to draw whole: at ~1 Hz a
    5-hour run is ~700,000 events, which renders. This is where a slow drift or
    a synchronous episode would be visible and a 60 s window could not show it."""
    plt = _mpl()
    m = bundle.meta
    t, c = bundle.io.window(0.0, m["duration_s"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    cmap = plt.get_cmap(GROUP_CMAP)
    axes[0].scatter(t / 3.6e6, c, s=0.35, marker="|",
                    c=[cmap(int(i) % 10) for i in c], linewidths=0.35, rasterized=True)
    axes[0].set_ylabel(f"IO cell ({m['n_io']} cells)")
    axes[0].set_ylim(-1, m["n_io"])
    axes[0].set_title(_title(bundle, f"every climbing-fiber event -- {len(t):,} total"))

    tb, r = bundle.io.rate_trace(30.0, m["duration_s"])
    axes[1].plot(tb / 3600.0, r, color="tab:red", linewidth=0.7)
    half = len(r) // 2
    axes[1].axhline(r[half:].mean(), color="black", linestyle="--", linewidth=1.2,
                    label=f"2nd-half mean {r[half:].mean():.3f} Hz")
    axes[1].set_ylabel("CF rate\n(Hz/cell, 30 s bins)")
    axes[1].set_xlabel("time (hours)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def weight_population(bundle, path, n_shown=400):
    """The tracked-synapse ensemble, four ways.

    The mean weight is a single number averaged over 160,000 synapses, and it
    hides everything interesting: individual synapses do not sit at the mean,
    they random-walk around it between the bounds. With 10,000 of them tracked
    at 1 s the distribution itself can be plotted rather than inferred, which is
    the point of sampling this many."""
    plt = _mpl()
    s = bundle.slow
    t_h, sw = s["t_s"] / 3600.0, s["sample_weights"]
    mean_w = s["mean_weight"].mean(axis=1)
    final = np.asarray(bundle.final["final_weights"]).ravel()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # (a) density of the ensemble over time -- a 2D histogram, because 10,000 overplotted
    # lines is a solid block and the thing worth seeing is where the mass is.
    edges = np.linspace(0, 1, 101)
    dens = np.stack([np.histogram(sw[i], bins=edges)[0] for i in range(len(t_h))], axis=1)
    # Log colour scale: every synapse starts at w_init, so the first column holds the whole
    # ensemble in a single bin and a linear scale renders the entire rest of the run as black.
    from matplotlib.colors import LogNorm
    im = axes[0, 0].imshow(np.maximum(dens, 0.5), origin="lower", aspect="auto", cmap="magma",
                           norm=LogNorm(vmin=1, vmax=max(dens.max(), 2)),
                           extent=(t_h[0], t_h[-1], 0, 1))
    axes[0, 0].plot(t_h, mean_w, color="cyan", linewidth=1.6, label="mean weight")
    axes[0, 0].set_ylabel("PF$\\to$PKJ weight")
    axes[0, 0].set_xlabel("time (hours)")
    axes[0, 0].set_title(f"(a) density of {sw.shape[1]:,} tracked synapses")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=axes[0, 0], label="synapses per bin")

    # (b) a readable subsample of the individual trajectories
    idx = np.linspace(0, sw.shape[1] - 1, min(n_shown, sw.shape[1])).astype(int)
    axes[0, 1].plot(t_h, sw[:, idx], color="tab:blue", alpha=0.08, linewidth=0.5)
    axes[0, 1].plot(t_h, mean_w, color="black", linewidth=2.0, label="mean weight")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylabel("PF$\\to$PKJ weight")
    axes[0, 1].set_xlabel("time (hours)")
    axes[0, 1].set_title(f"(b) {len(idx)} individual synapses")
    axes[0, 1].legend(loc="upper right", fontsize=8)

    # (c) the distribution at a few times, and the full 160,000 at the end
    for frac, colour in ((0.0, "tab:purple"), (0.02, "tab:blue"), (0.25, "tab:green"), (1.0, "tab:red")):
        i = min(len(t_h) - 1, int(frac * (len(t_h) - 1)))
        axes[1, 0].hist(sw[i], bins=edges, histtype="step", density=True, color=colour,
                        linewidth=1.4, label=f"t = {t_h[i]:.2f} h")
    axes[1, 0].hist(final, bins=edges, histtype="stepfilled", density=True, color="black",
                    alpha=0.18, label=f"final, all {final.size:,}")
    axes[1, 0].set_xlabel("PF$\\to$PKJ weight")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title("(c) weight distribution over time")
    axes[1, 0].legend(fontsize=8)

    # (d) the tracked sample against the full final matrix -- does 10,000 represent 160,000?
    axes[1, 1].hist(final, bins=edges, histtype="stepfilled", density=True, color="tab:gray",
                    alpha=0.5, label=f"all {final.size:,} synapses")
    axes[1, 1].hist(sw[-1], bins=edges, histtype="step", density=True, color="tab:red",
                    linewidth=1.6, label=f"the {sw.shape[1]:,} tracked")
    axes[1, 1].set_xlabel("PF$\\to$PKJ weight at end of run")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title(f"(d) sample vs population: mean {sw[-1].mean():.4f} vs "
                         f"{final.mean():.4f}, SD {sw[-1].std():.4f} vs {final.std():.4f}")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(_title(bundle, "individual PF$\\to$PKJ synapse weights"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def weight_diffusion(bundle, path):
    """Is the ensemble stationary, and how fast does a single synapse wander?

    Three questions a 60 s run cannot answer. The cross-synapse SD says whether
    the spread has reached a steady state or is still widening. The mean squared
    displacement against lag says whether individual synapses diffuse (MSD ~ t,
    a free random walk) or are confined (MSD saturating, a walk in a restoring
    potential) -- the loop regulates the MEAN, so individual synapses should be
    the second. The autocorrelation gives the timescale of that confinement."""
    plt = _mpl()
    s = bundle.slow
    t_s, sw = s["t_s"], s["sample_weights"].astype(np.float64)
    dt_s = float(np.median(np.diff(t_s)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    axes[0].plot(t_s / 3600.0, sw.std(axis=1), color="tab:purple", linewidth=1.0)
    axes[0].set_xlabel("time (hours)")
    axes[0].set_ylabel("cross-synapse SD")
    axes[0].set_title("(a) spread of the ensemble")
    axes[0].grid(alpha=0.3)

    # MSD over the second half only, so the transient does not masquerade as diffusion.
    half = len(t_s) // 2
    w2 = sw[half:]
    max_lag = min(len(w2) - 1, int(3600.0 / dt_s))               # up to 1 hour of lag
    lags = np.unique(np.round(np.logspace(0, np.log10(max(max_lag, 2)), 40)).astype(int))
    msd = np.array([np.mean((w2[l:] - w2[:-l]) ** 2) for l in lags])
    axes[1].loglog(lags * dt_s, msd, "o-", color="tab:blue", markersize=3, label="measured")
    ref = msd[0] * (lags / lags[0])
    axes[1].loglog(lags * dt_s, ref, "--", color="gray", linewidth=1, label="free diffusion ($\\propto t$)")
    axes[1].axhline(2 * w2.var(), color="tab:red", linestyle=":",
                    label="confined limit ($2\\sigma^2$)")
    axes[1].set_xlabel("lag (s)")
    axes[1].set_ylabel("MSD of one synapse")
    axes[1].set_title("(b) single-synapse displacement")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    # Autocorrelation of each synapse's deviation from the network mean, averaged.
    dev = w2 - w2.mean(axis=1, keepdims=True)
    dev -= dev.mean(axis=0, keepdims=True)
    n_ac = min(dev.shape[1], 2000)                               # 2000 synapses is plenty for a mean AC
    acs = []
    for j in range(n_ac):
        x = dev[:, j]
        denom = np.dot(x, x)
        if denom > 0:
            acs.append(np.correlate(x, x, mode="full")[len(x) - 1:len(x) - 1 + max_lag] / denom)
    ac = np.mean(acs, axis=0)
    axes[2].plot(np.arange(len(ac)) * dt_s, ac, color="tab:green", linewidth=1.0)
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].axhline(1 / np.e, color="gray", linestyle=":", linewidth=0.8, label="1/e")
    below = np.flatnonzero(ac < 1 / np.e)
    if len(below):
        axes[2].axvline(below[0] * dt_s, color="tab:red", linestyle="--",
                        label=f"$\\tau$ = {below[0] * dt_s:.0f} s")
    axes[2].set_xlabel("lag (s)")
    axes[2].set_ylabel("autocorrelation")
    axes[2].set_title(f"(c) deviation from mean, {n_ac} synapses")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(_title(bundle, "weight random walk, measured on the second half"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def io_stats(bundle, path):
    """Per-cell CF rate, the inter-CF interval distribution, and CV -- over the
    whole run rather than a window, which is the only way the ISI histogram has
    enough events per cell to show the olive's oscillation-gated structure."""
    plt = _mpl()
    m = bundle.meta
    half = m["duration_s"] / 2.0
    counts = bundle.io.per_cell_counts(half, m["duration_s"])
    rates = counts / half
    trains = bundle.io.trains(half, m["duration_s"])
    isis = np.concatenate([np.diff(t) for t in trains if len(t) > 1])
    _, cvs = bundle.io.isi_cv(half, m["duration_s"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].bar(np.arange(len(rates)), rates, color="tab:red", width=0.85)
    axes[0].axhline(rates.mean(), color="black", linestyle="--",
                    label=f"mean {rates.mean():.3f} Hz  (SD {rates.std():.3f})")
    axes[0].set_xlabel("IO cell")
    axes[0].set_ylabel("CF rate (Hz)")
    axes[0].set_title(f"(a) per-cell rate, {int(counts.sum()):,} events")
    axes[0].legend(fontsize=8)

    axes[1].hist(isis / 1000.0, bins=160, color="tab:red", alpha=0.85)
    axes[1].set_xlabel("inter-CF interval (s)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"(b) ISI distribution, n = {len(isis):,}\nmean {isis.mean() / 1000:.3f} s, "
                      f"median {np.median(isis) / 1000:.3f} s")

    axes[2].hist(cvs[np.isfinite(cvs)], bins=25, color="tab:purple", alpha=0.85)
    axes[2].axvline(np.nanmean(cvs), color="black", linestyle="--",
                    label=f"mean CV {np.nanmean(cvs):.3f}")
    axes[2].axvline(1.0, color="gray", linestyle=":", label="Poisson (CV = 1)")
    axes[2].set_xlabel("ISI coefficient of variation")
    axes[2].set_ylabel("cells")
    axes[2].set_title("(c) regularity")
    axes[2].legend(fontsize=8)

    fig.suptitle(_title(bundle, "climbing-fiber statistics, second half of the run"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def io_synchrony(bundle, path, bin_ms=20.0, max_lag_ms=500.0):
    """Do gap-junction-coupled olivary cells fire together?

    Pairwise zero-lag correlation of binned CF trains, plotted against distance
    along the coupling ring, plus the mean cross-correlogram for coupled and
    uncoupled pairs. The claim gap junctions are in the model to support is that
    coupled neighbours synchronize; the uncoupled-pair curve is the control that
    says any correlation seen is not just shared inhibition from the nucleus."""
    plt = _mpl()
    m, n_io = bundle.meta, bundle.meta["n_io"]
    gap = np.asarray(bundle.final["gap_matrix"])
    half = m["duration_s"] / 2.0
    n_bins = int((m["duration_s"] - half) * 1000.0 / bin_ms)
    t, c = bundle.io.window(half, m["duration_s"])
    b = ((t - half * 1000.0) / bin_ms).astype(np.int64)
    ok = (b >= 0) & (b < n_bins)
    binned = np.zeros((n_io, n_bins), dtype=np.float32)
    np.add.at(binned, (c[ok], b[ok]), 1.0)
    z = binned - binned.mean(axis=1, keepdims=True)
    sd = z.std(axis=1)
    corr = (z @ z.T) / n_bins / np.outer(sd, sd).clip(1e-12)

    iu = np.triu_indices(n_io, k=1)
    ring_d = np.minimum(np.abs(iu[0] - iu[1]), n_io - np.abs(iu[0] - iu[1]))
    coupled = gap[iu] > 0
    cc = corr[iu]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].imshow(corr, cmap="RdBu_r", vmin=-np.abs(cc).max(), vmax=np.abs(cc).max())
    axes[0].set_title(f"(a) pairwise CF correlation\n({bin_ms:.0f} ms bins)")
    axes[0].set_xlabel("IO cell"); axes[0].set_ylabel("IO cell")

    for d in np.unique(ring_d):
        sel = ring_d == d
        axes[1].errorbar(d, cc[sel].mean(), yerr=cc[sel].std(), fmt="o", markersize=4,
                         color="tab:red" if coupled[sel].any() else "tab:gray")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("distance along the coupling ring")
    axes[1].set_ylabel("zero-lag correlation")
    axes[1].set_title("(b) correlation vs distance\nred = gap-coupled pairs exist at that distance")
    axes[1].grid(alpha=0.3)

    lag_bins = int(max_lag_ms / bin_ms)
    def mean_ccg(mask):
        out = np.zeros(2 * lag_bins + 1)
        pairs = np.flatnonzero(mask)
        if not len(pairs):
            return out
        for p in pairs[:400]:                       # 400 pairs is enough for a mean CCG
            i, j = iu[0][p], iu[1][p]
            x, y = z[i], z[j]
            full = np.correlate(x, y, mode="full") / (n_bins * sd[i] * sd[j] + 1e-12)
            out += full[n_bins - 1 - lag_bins:n_bins + lag_bins]
        return out / min(len(pairs), 400)

    lags = np.arange(-lag_bins, lag_bins + 1) * bin_ms
    axes[2].plot(lags, mean_ccg(coupled), color="tab:red", label=f"gap-coupled ({coupled.sum()} pairs)")
    axes[2].plot(lags, mean_ccg(~coupled), color="tab:gray", label=f"uncoupled ({(~coupled).sum()} pairs)")
    axes[2].axvline(0, color="gray", linewidth=0.8)
    axes[2].set_xlabel("lag (ms)")
    axes[2].set_ylabel("cross-correlation")
    axes[2].set_title("(c) mean cross-correlogram")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(_title(bundle, f"olivary synchrony (gap_g = {m['gap_g']})"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def equilibrium(bundle, path, bin_s=60.0):
    """The headline question, answered on the half of the run that is settled.

    Drift is fitted over the SECOND HALF only and quoted per hour: the whole-run
    fit that sim/analysis.weight_drift_slope does is dominated by the transient
    on any run long enough to have one, and it would report a settled loop as
    drifting."""
    plt = _mpl()
    m = bundle.meta
    s = bundle.slow
    t_s, w = s["t_s"], s["mean_weight"].mean(axis=1).astype(np.float64)
    half = len(t_s) // 2
    slope, intercept = np.polyfit(t_s[half:], w[half:], 1)
    slope_all, _ = np.polyfit(t_s, w, 1)

    bin_s = min(bin_s, max(m["duration_s"] / 20.0, 1.0))   # >= 20 bins, so the fit below is defined
    tb, r = bundle.io.rate_trace(bin_s, m["duration_s"])
    hb = len(r) // 2
    r_slope, r_int = np.polyfit(tb[hb:], r[hb:], 1)
    predicted = 1000.0 / (m["ltd_window_ms"] * (1.0 + m["delta_minus"] / m["delta_plus"]))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(t_s / 3600.0, w, color="black", linewidth=0.9, label="mean PF$\\to$PKJ weight")
    axes[0].plot(t_s[half:] / 3600.0, slope * t_s[half:] + intercept, color="tab:red", linewidth=2,
                 label=f"2nd-half drift {slope * 3600:+.5f} /hour")
    axes[0].axvspan(0, t_s[half] / 3600.0, color="gray", alpha=0.12, label="1st half (transient)")
    axes[0].set_ylabel("mean weight")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].set_title(_title(bundle, "is it settled?"))

    axes[1].plot(tb / 3600.0, r, color="tab:red", linewidth=0.6, alpha=0.7,
                 label=f"CF rate ({bin_s:.0f} s bins)")
    axes[1].plot(tb[hb:] / 3600.0, r_slope * tb[hb:] + r_int, color="black", linewidth=2,
                 label=f"2nd-half drift {r_slope * 3600:+.5f} Hz/hour")
    axes[1].axhline(predicted, color="tab:green", linestyle="--", linewidth=1.4,
                    label=f"H2 prediction {predicted:.3f} Hz")
    axes[1].axhline(r[hb:].mean(), color="tab:blue", linestyle=":", linewidth=1.4,
                    label=f"2nd-half mean {r[hb:].mean():.4f} Hz")
    axes[1].axvspan(0, tb[hb] / 3600.0, color="gray", alpha=0.12)
    axes[1].set_ylabel("CF rate (Hz)")
    axes[1].set_xlabel("time (hours)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"weight_drift_per_hour_2nd_half": float(slope * 3600),
            "weight_drift_per_hour_whole": float(slope_all * 3600),
            "cf_rate_drift_hz_per_hour_2nd_half": float(r_slope * 3600),
            "cf_rate_2nd_half_hz": float(r[hb:].mean()),
            "cf_rate_predicted_hz": float(predicted),
            "mean_weight_2nd_half": float(w[half:].mean()),
            "cross_synapse_sd_final": float(np.asarray(bundle.final["final_weights"]).std())}


# --- the per-bundle driver -------------------------------------------------

def figures_for_bundle(bundle_dir, out_dir):
    bundle = RunBundle(bundle_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = bundle.meta
    print(f"[figures] {bundle_dir} -- {m['duration_s']:.0f} s, "
          f"{sum(m['n_spikes'].values()) / 1e6:.0f}M spikes")

    summary = equilibrium(bundle, out_dir / "equilibrium.png")
    rates_full(bundle, out_dir / "rates_full.png")
    raster_io_full(bundle, out_dir / "raster_io_full.png")
    weight_population(bundle, out_dir / "weight_population.png")
    weight_diffusion(bundle, out_dir / "weight_diffusion.png")
    io_stats(bundle, out_dir / "io_stats.png")
    # Drawn for the gap_off run too: an uncoupled olive is the control this figure is read against.
    io_synchrony(bundle, out_dir / "io_synchrony.png")

    # Rasters, voltages and the mean/individual weight figure, at each recorded window,
    # drawn by the same functions the short runs use.
    names = ["early", "mid", "late"]
    for k, start in enumerate(m["trace_windows_s"]):
        name = names[k] if k < len(names) else f"w{k}"
        t0, t1 = start, start + m["trace_len_s"]
        log = _as_simlog(bundle, t0, t1, trace_idx=k)
        plot_rasters(log, out_dir / f"rasters_{name}.png", window_s=t1 - t0,
                     title=f"{m['label']} -- {t0:.0f}-{t1:.0f} s ({name})")
        plot_voltage_traces(log, out_dir / f"voltages_{name}.png",
                            title=f"{m['label']} membrane potentials -- {t0:.0f}-{t1:.0f} s ({name})")
        plot_io_state(log, out_dir / f"io_state_{name}.png",
                      title=f"{m['label']} IO $V_m$ and [Ca$^{{2+}}$]$_i$ -- {t0:.0f}-{t1:.0f} s ({name})")
    plot_weights(_as_simlog(bundle, 0, min(10.0, m["duration_s"])), out_dir / "weights.png",
                 title=_title(bundle, "mean and individual PF$\\to$PKJ weights"))

    summary.update({k: m[k] for k in ("label", "duration_s", "n_pf_per_pkj", "gap_g",
                                      "cf_pkj_reversal_mv", "ablate_dcn_io", "seed", "wall_s")})
    summary["n_spikes"] = m["n_spikes"]
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[figures] -> {out_dir}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", nargs="*", default=[], help="run bundle directory/ies")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for b in args.bundle:
        figures_for_bundle(b, out / Path(b).name)


if __name__ == "__main__":
    main()
