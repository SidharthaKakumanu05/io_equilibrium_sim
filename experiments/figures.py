"""The four analysis figures, built from the sweep sidecars and the raw .npz
trajectories. Re-runs in seconds; it never re-simulates anything.

    python experiments/sweep.py all --workers 60    # produce the data (hours)
    python experiments/figures.py                   # produce the figures (seconds)

Writes results/fig1_h1_equilibrium.png (H1: the equilibrium and the ablation
that removes it), fig2_plasticity.png (H2 predicted vs measured, and H3 as a
trade-off), fig3_coupling.png (what gap junctions do and what limits them), and
fig4_robustness.png (settling time, PF-pool sensitivity, timestep convergence).

It takes no arguments -- every panel is tied to a specific experiment in
sweep.py, so there is nothing to select. Run the sweep first: without sidecars
this exits with a message rather than a traceback.

Styling follows one validated categorical palette (blue/orange/aqua/yellow,
checked with the dataviz validator: worst adjacent CVD dE 9.1, normal-vision
22.9). Aqua and yellow sit below 3:1 on this surface, so every series that uses
them carries a legend AND a direct label -- identity is never colour alone.

No chart here uses two y-scales. Where a panel would have needed one (rate and
weight, rate and spread), the measures are split into their own panels or recast
as a trade-off scatter, which is what the relationship actually is.
"""
import collections, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "results" / "sweeps"
OUT = ROOT / "results"

# Every panel below indexes a specific sidecar or .npz by name, so a missing sweep
# surfaces as a KeyError or FileNotFoundError deep inside a plotting call. Say what
# is actually wrong instead.
_NEEDED = ("open_loop", "initial_conditions", "heterogeneity_coupling",
           "h2_window", "h3_three_window", "settling", "dt_convergence",
           "topology", "pf_pool")
_missing = [e for e in _NEEDED if not (SWEEPS / e).is_dir()]
if _missing:
    sys.exit(f"no sweep data for: {', '.join(_missing)}.\n"
             f"Run `python experiments/sweep.py all --workers <n>` first "
             f"(it is resumable, so a partial run can be continued).")
OUT.mkdir(parents=True, exist_ok=True)

# --- validated palette -----------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "text.color": INK, "font.size": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 5.5,
})


def load(exp):
    """Every sidecar for one experiment."""
    return [json.loads(p.read_text()) for p in sorted((SWEEPS / exp).glob("*.json"))]


def raw(exp, name):
    """One run's saved trajectories. Only conditions marked `_save_raw` in
    sweep.py have these, which is why the panels that use them pin seed=0."""
    return np.load(SWEEPS / exp / f"{name}.npz", allow_pickle=True)


def agg(recs, key, metric):
    """(x, mean, sample SD) of one metric against one swept override."""
    g = collections.defaultdict(list)
    for r in recs:
        g[r["overrides"].get(key)].append(r["metrics"][metric])
    xs = sorted(g)
    return (np.array(xs, float),
            np.array([np.mean(g[x]) for x in xs]),
            np.array([np.std(g[x], ddof=1) if len(g[x]) > 1 else 0.0 for x in xs]))


def tidy(ax, title=None, xl=None, yl=None):
    if title: ax.set_title(title, fontsize=10.5, pad=8)
    if xl: ax.set_xlabel(xl)
    if yl: ax.set_ylabel(yl)
    ax.grid(alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)


# ===========================================================================
# FIGURE 1 -- H1: the equilibrium, and what happens when you remove it
# ===========================================================================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.6))
fig.suptitle("H1 — the CF rate equilibrium is produced by the DCN→IO feedback limb",
             fontsize=13.5, y=0.975)

# 1a: convergence from five initial weights
a = ax[0, 0]
for w0 in (0.1, 0.3, 0.5, 0.7, 0.9):
    d = raw("initial_conditions", f"w_init={w0}__seed=0")
    t, mw = d["t_ms"] / 1000.0, d["mean_weight"].mean(axis=1)
    col = SEQ[[0.1, 0.3, 0.5, 0.7, 0.9].index(w0) + 1]
    a.plot(t, mw, color=col, lw=1.6)
    # Label at the START: the curves are separated there and all converge at the end,
    # so end-labels would pile up on the same point.
    a.annotate(f"$w_0$={w0}", (0, w0), xytext=(-6, 0), textcoords="offset points",
               fontsize=8.5, color=col, va="center", ha="right")
a.axhline(0.364, ls=(0, (4, 3)), color=MUTED, lw=1.2)
a.annotate("equilibrium 0.364", (200, 0.364), xytext=(0, 7), textcoords="offset points",
           fontsize=8.5, color=MUTED, ha="right")
a.set_xlim(-28, 205)
tidy(a, "Five starting weights, one attractor", "time (s)", "mean PF→PKJ weight")

# 1b: open vs closed weight trajectory
a = ax[0, 1]
for lbl, nm, col in (("closed loop", "ablate=0__cv=0.0__seed=0", BLUE),
                     ("DCN→IO cut", "ablate=1__cv=0.0__seed=0", ORANGE)):
    d = raw("open_loop", nm)
    t, mw = d["t_ms"] / 1000.0, d["mean_weight"].mean(axis=1)
    a.plot(t, mw, color=col, label=lbl)
    a.annotate(lbl, (t[-1], mw[-1]), xytext=(-4, 8 if col == BLUE else 8),
               textcoords="offset points", fontsize=8.5, color=col, ha="right")
a.set_ylim(-0.02, 0.62)
tidy(a, "Cutting the feedback limb: weights collapse to the floor", "time (s)", "mean PF→PKJ weight")
a.legend(loc="center right", fontsize=8.5)

# 1c: CF rate, closed vs cut
a = ax[1, 0]
r = load("open_loop")
groups, vals = [], []
for ab in (False, True):
    for cv in (0.0, 0.1):
        v = [x["metrics"]["io_rate_hz"] for x in r
             if x["overrides"]["ablate_dcn_io"] == ab and x["overrides"]["io_heterogeneity_cv"] == cv]
        groups.append(("closed" if not ab else "cut") + f"\ncv={cv}")
        vals.append(v)
cols = [BLUE, BLUE, ORANGE, ORANGE]
a.bar(range(4), [np.mean(v) for v in vals], 0.62, color=cols,
      yerr=[np.std(v, ddof=1) for v in vals], capsize=4,
      error_kw={"ecolor": INK2, "lw": 1.2})
for i, v in enumerate(vals):
    a.annotate(f"{np.mean(v):.2f}", (i, np.mean(v)), xytext=(0, 7),
               textcoords="offset points", ha="center", fontsize=9, color=INK)
a.axhline(3.00, ls=(0, (4, 3)), color=MUTED, lw=1.2)
a.annotate("isolated cell, intrinsic 3.00 Hz", (-0.42, 3.00), xytext=(0, 6),
           textcoords="offset points", ha="left", fontsize=8.5, color=MUTED)
a.set_xticks(range(4)); a.set_xticklabels(groups, fontsize=8.5)
a.set_ylim(0, 3.5)
tidy(a, "CF rate runs free to the intrinsic rate", None, "CF rate (Hz)")

# 1d: inter-CF interval distributions
a = ax[1, 1]
for lbl, nm, col in (("closed loop", "ablate=0__cv=0.0__seed=0", BLUE),
                     ("DCN→IO cut", "ablate=1__cv=0.0__seed=0", ORANGE)):
    d = raw("open_loop", nm)
    isis = np.concatenate([np.diff(np.asarray(t)) for t in d["io_spikes"] if len(t) > 2])
    a.hist(isis, bins=np.linspace(0, 2000, 70), histtype="step", lw=1.8,
           color=col, density=True, label=lbl)
a.set_xlim(0, 2000)
tidy(a, "Inter-CF interval distribution", "inter-CF interval (ms)", "density")
a.legend(fontsize=8.5)

fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(OUT / "fig1_h1_equilibrium.png", dpi=140)
print("wrote fig1_h1_equilibrium.png")


# ===========================================================================
# FIGURE 2 -- H2 and H3: the plasticity rule sets the operating point
# ===========================================================================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.6))
fig.suptitle("H2 & H3 — the plasticity window sets the equilibrium, and a null window trades rate for stability",
             fontsize=13.5, y=0.975)

r2 = load("h2_window")
g = collections.defaultdict(list)
for x in r2:
    g[(x["overrides"]["ltd_window_ms"], x["overrides"]["delta_minus"])].append(x["metrics"])
keys = sorted(g)
pred = np.array([1000.0 / (w * (1.0 + dm / 0.001)) for (w, dm) in keys])
meas = np.array([np.mean([m["io_rate_hz"] for m in g[k]]) for k in keys])
merr = np.array([np.std([m["io_rate_hz"] for m in g[k]], ddof=1) for k in keys])
wts = np.array([np.mean([m["mean_weight"] for m in g[k]]) for k in keys])

# 2a: predicted vs measured
a = ax[0, 0]
lim = [0.32, 2.15]
a.plot(lim, lim, ls=(0, (4, 3)), color=MUTED, lw=1.2, zorder=1)
a.errorbar(pred, meas, yerr=merr, fmt="o", color=BLUE, capsize=4, ms=7,
           ecolor=INK2, elinewidth=1.2, zorder=3)
# Settings that predict the SAME rate land on the same point -- that agreement is the
# result, so stack their labels instead of letting them overprint each other.
seen = collections.Counter()
for p_, m_, (w, dm) in zip(pred, meas, keys):
    k = round(p_, 3)
    dy = (-10, 7)[seen[k]] if seen[k] < 2 else -10 - 11 * seen[k]
    seen[k] += 1
    a.annotate(f"{w:.0f} ms / {dm/0.001:.0f}", (p_, m_), xytext=(9, dy),
               textcoords="offset points", fontsize=7.8, color=INK2)
a.annotate("y = x", (0.62, 0.55), fontsize=8.5, color=MUTED, rotation=38)
a.set_xlim(lim); a.set_ylim(lim)
tidy(a, "Every setting lands on prediction", "predicted CF rate (Hz)", "measured CF rate (Hz)")

# 2b: residuals -- the deviation, at readable scale
a = ax[0, 1]
resid = meas - pred
a.axhline(0, color=AXIS, lw=1.2)
a.errorbar(pred, resid, yerr=merr, fmt="o", color=BLUE, capsize=4, ms=7,
           ecolor=INK2, elinewidth=1.2)
a.axhspan(-0.02, 0.02, color=SEQ[0], alpha=0.55, zorder=0)
a.annotate("±0.02 Hz", (2.05, 0.021), fontsize=8.5, color=INK2, ha="right")
tidy(a, "Residuals: every point within 0.02 Hz", "predicted CF rate (Hz)", "measured − predicted (Hz)")

# 2c: the equilibrium weight the loop uses to get there
a = ax[1, 0]
order = np.argsort(meas)
a.plot(meas[order], wts[order], "o-", color=ORANGE, ms=7)
a.axhline(1.0, ls=(0, (4, 3)), color=MUTED, lw=1.2)
a.annotate("$w_{max}$ ceiling — not reached anywhere", (2.0, 1.0), xytext=(0, -14),
           textcoords="offset points", fontsize=8.5, color=MUTED, ha="right")
a.set_ylim(0, 1.1)
tidy(a, "Headroom: the weight each equilibrium needs", "measured CF rate (Hz)", "equilibrium mean weight")

# 2d: H3 as the trade-off it actually is (avoids a second y-scale)
a = ax[1, 1]
r3 = load("h3_three_window")
g3 = collections.defaultdict(list)
for x in r3:
    g3[x["overrides"]["null_window_ms"]].append(x["metrics"])
nulls = sorted(g3)
rate = [np.mean([m["io_rate_hz"] for m in g3[n]]) for n in nulls]
sd = [np.mean([m["cross_synapse_std"] for m in g3[n]]) for n in nulls]
a.plot(rate, sd, "-", color=MUTED, lw=1.2, zorder=1)
a.scatter(rate, sd, c=[SEQ[i + 1] for i in range(len(nulls))], s=110,
          edgecolors=SURFACE, linewidths=1.6, zorder=3)
for n, x_, y_ in zip(nulls, rate, sd):
    a.annotate(f"{n:.0f} ms", (x_, y_), xytext=(0, 11), textcoords="offset points",
               ha="center", fontsize=8.5, color=INK2)
a.annotate("wider null window →\nsteadier weights, worse rate control",
           (0.845, 0.1418), xytext=(14, 16), textcoords="offset points",
           fontsize=8.5, color=INK2)
tidy(a, "H3 is a trade, not an improvement", "CF rate (Hz)", "cross-synapse weight SD")

fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(OUT / "fig2_plasticity.png", dpi=140)
print("wrote fig2_plasticity.png")


# ===========================================================================
# FIGURE 3 -- coupling: what gap junctions do, and what sets it
# ===========================================================================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.6))
fig.suptitle("Electrical coupling — synchrony is limited by path length, and heterogeneity does a different job",
             fontsize=13.5, y=0.975)

hc = load("heterogeneity_coupling")

# 3a: correlation measures vs strength (both are correlations: one shared axis)
a = ax[0, 0]
r = [x for x in hc if x["overrides"]["io_heterogeneity_cv"] == 0.1]
for metric, col, lbl in (("vm_correlation", BLUE, "subthreshold $V_m$ r"),
                         ("cf_synchrony", ORANGE, "CF spike synchrony")):
    x, m, sd = agg(r, "gap_g", metric)
    xl = x / 0.06 * 4
    a.errorbar(xl, m, yerr=sd, fmt="o-", color=col, capsize=4, ms=6,
               ecolor=INK2, elinewidth=1.0, label=lbl)
    a.annotate(lbl, (xl[-1], m[-1]), xytext=(-6, 9), textcoords="offset points",
               fontsize=8.5, color=col, ha="right")
a.set_ylim(0, 1.0)
tidy(a, "Coupling strength raises correlation", "total coupling conductance (× leak)", "correlation")
a.legend(fontsize=8.5, loc="upper left")

# 3b: CF rate is untouched -- its own panel, because Hz is not a correlation
a = ax[0, 1]
x, m, sd = agg(r, "gap_g", "io_rate_hz")
a.errorbar(x / 0.06 * 4, m, yerr=sd, fmt="o-", color=AQUA, capsize=4, ms=6,
           ecolor=INK2, elinewidth=1.0)
a.axhline(1.0, ls=(0, (4, 3)), color=MUTED, lw=1.2)
a.annotate("CF rate", (3.8, m[-1]), xytext=(-6, 10), textcoords="offset points",
           fontsize=8.5, color=AQUA, ha="right")
a.set_ylim(0.8, 1.2)
tidy(a, "…but does not touch the rate", "total coupling conductance (× leak)", "CF rate (Hz)")

# 3c: reach -- same conductance, different topology
a = ax[1, 0]
prof = collections.defaultdict(lambda: collections.defaultdict(list))
for x in load("topology"):
    if x["overrides"]["io_heterogeneity_cv"] != 0.0:
        continue
    for d, v in x["metrics"]["vm_correlation_by_ring_distance"].items():
        prof[x["overrides"]["gap_topology"]][int(d)].append(v)
# The three local topologies converge to nearly the same far-field value, so their
# end-labels are staggered vertically rather than left to overprint each other.
for topo, col, dy in (("all_to_all", BLUE, 0), ("small_world", ORANGE, 11),
                      ("nearest_k", AQUA, 0), ("ring", YELLOW, -11)):
    ds = sorted(prof[topo])
    ys = [np.mean(prof[topo][d]) for d in ds]
    a.plot(ds, ys, "-o", color=col, ms=4.5, label=topo)
    a.annotate(topo, (ds[-1], ys[-1]), xytext=(7, dy), textcoords="offset points",
               fontsize=8.5, color=col, va="center")
a.set_xlim(0.5, 24); a.set_ylim(0, 1.0)
tidy(a, "Identical conductance (0.95× leak), different reach",
     "distance around the IO ring", "subthreshold $V_m$ r")
a.legend(fontsize=8.5, loc="lower left")

# 3d: what heterogeneity actually controls -- rate dispersion, and coupling erases it
a = ax[1, 1]
cvs = (0.0, 0.05, 0.10, 0.20)
for i, cv in enumerate(cvs):
    rr = [x for x in hc if x["overrides"]["io_heterogeneity_cv"] == cv]
    x, m, sd = agg(rr, "gap_g", "io_rate_spread_hz")
    a.plot(x / 0.06 * 4, m, "-o", color=SEQ[i + 2], ms=5.5, label=f"cv = {cv}")
    a.annotate(f"cv={cv}", (0, m[0]), xytext=(-6, 0), textcoords="offset points",
               fontsize=8.5, color=SEQ[i + 2], ha="right", va="center")
a.set_xlim(-1.05, 4.05)
tidy(a, "Heterogeneity spreads the rates; coupling pulls them back",
     "total coupling conductance (× leak)", "cell-to-cell CF rate SD (Hz)")
a.legend(fontsize=8.5)

fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(OUT / "fig3_coupling.png", dpi=140)
print("wrote fig3_coupling.png")


# ===========================================================================
# FIGURE 4 -- robustness: does the answer survive the model's own choices?
# ===========================================================================
fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.6))
fig.suptitle("Robustness — which results are properties of the circuit, and which of the implementation",
             fontsize=13.5, y=0.975)

# 4a: settling over 600 s
a = ax[0, 0]
for cv, col in ((0.0, BLUE), (0.1, ORANGE)):
    d = raw("settling", f"cv={cv}__seed=0")
    t, mw = d["t_ms"] / 1000.0, d["mean_weight"].mean(axis=1)
    a.plot(t, mw, color=col, lw=1.1, alpha=0.9, label=f"cv = {cv}")
a.axvspan(0, 90, color=SEQ[0], alpha=0.5, zorder=0)
a.annotate("a 90 s run ends here.\nIts drift is measured over 45–90 s,\nwhich is still inside the transient.",
           (100, 0.478), fontsize=8.5, color=INK2)
a.annotate("", xy=(90, 0.44), xytext=(150, 0.462),
           arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
a.set_xlim(0, 615); a.set_ylim(0.30, 0.52)
tidy(a, "The transient runs ~100 s; drift must be read after it",
     "time (s)", "mean PF→PKJ weight")
a.legend(fontsize=8.5, loc="lower right")

# 4b: PF pool -- rate
a = ax[0, 1]
pf = load("pf_pool")
x, m, sd = agg(pf, "n_pf_per_pkj", "io_rate_hz")
a.errorbar(x, m, yerr=sd, fmt="o-", color=BLUE, capsize=4, ms=6, ecolor=INK2, elinewidth=1.0)
a.axhline(1.0, ls=(0, (4, 3)), color=MUTED, lw=1.2)
a.axvspan(250, 1000, color=SEQ[0], alpha=0.5, zorder=0)
a.annotate("rate invariant\nover 4× range", (300, 0.45), fontsize=8.5, color=INK2)
a.annotate("breaks:\nweights saturate", (128, 0.30), fontsize=8.5, color=ORANGE)
a.set_xscale("log"); a.set_ylim(0, 1.3)
a.set_xticks([125, 250, 500, 1000, 2000])
a.set_xticklabels(["125", "250", "500", "1000", "2000"])
a.minorticks_off()
tidy(a, "CF rate is robust to the PF-pool simplification",
     "PF fibers per Purkinje cell", "CF rate (Hz)")

# 4c: PF pool -- weight (its own panel: different units)
a = ax[1, 0]
x, m, sd = agg(pf, "n_pf_per_pkj", "mean_weight")
a.errorbar(x, m, yerr=sd, fmt="o-", color=ORANGE, capsize=4, ms=6, ecolor=INK2, elinewidth=1.0)
ref = m[2] * 500.0 / x
a.plot(x[ref <= 1.05], ref[ref <= 1.05], ls=(0, (4, 3)), color=MUTED, lw=1.2)
a.annotate("$w \\propto 1/n_{PF}$", (1150, 0.235), fontsize=8.5, color=MUTED)
a.axvline(500, ls=(0, (2, 2)), color=AXIS, lw=1.2)
a.annotate("shipped", (520, 0.92), fontsize=8.5, color=INK2)
a.set_xscale("log"); a.set_ylim(0, 1.08)
a.set_xticks([125, 250, 500, 1000, 2000])
a.set_xticklabels(["125", "250", "500", "1000", "2000"])
a.minorticks_off()
tidy(a, "…but the weight it settles at is not",
     "PF fibers per Purkinje cell", "equilibrium mean weight")

# 4d: timestep convergence
a = ax[1, 1]
dt = load("dt_convergence")
x, m, sd = agg(dt, "dt_ms", "io_rate_hz")
a.errorbar(x, m, yerr=sd, fmt="o-", color=BLUE, capsize=4, ms=7, ecolor=INK2,
           elinewidth=1.2, label="CF rate (Hz)")
xw, mw_, sdw = agg(dt, "dt_ms", "mean_weight")
a.errorbar(xw, mw_ / mw_[-1], yerr=sdw / mw_[-1], fmt="s--", color=ORANGE, capsize=4,
           ms=7, ecolor=INK2, elinewidth=1.2, label="mean weight (÷ its dt=1 ms value)")
a.axhline(1.0, ls=(0, (4, 3)), color=MUTED, lw=1.2)
a.annotate("CF rate", (0.25, m[0]), xytext=(8, -12), textcoords="offset points",
           fontsize=8.5, color=BLUE)
a.annotate("weight", (0.25, mw_[0] / mw_[-1]), xytext=(8, 4), textcoords="offset points",
           fontsize=8.5, color=ORANGE)
a.set_xscale("log"); a.set_xticks([0.25, 0.5, 1.0]); a.set_xticklabels(["0.25", "0.5", "1.0"])
a.minorticks_off()
a.set_ylim(0.8, 1.25)
tidy(a, "Outer timestep: rate is invariant, weight drifts ~15%",
     "outer timestep dt (ms)", "value, normalised to dt = 1 ms")
a.legend(fontsize=8.5, loc="upper left")

fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(OUT / "fig4_robustness.png", dpi=140)
print("wrote fig4_robustness.png")
