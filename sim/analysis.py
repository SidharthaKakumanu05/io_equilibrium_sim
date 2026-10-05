"""Turning a SimLog into numbers and figures.

run.py uses five things from this file:

  summarize()            the numbers printed at the end of a run
  plot_rasters()         spike rasters for all four cell types   -> rasters.png
  plot_weights()         PF->PKJ weights over time               -> weights.png
  plot_voltage_traces()  membrane voltages at the end of the run -> voltages.png
  plot_io_state()        olive voltage and calcium               -> io_state.png

Everything else here is a helper those use, or a plot run.py does not call.

All firing rates are counted directly from the recorded spikes: number of
spikes divided by the length of the window.
"""
import numpy as np

GROUP_CMAP = "tab10"          # colour scheme for every plot. PF and PKJ are coloured by their climbing
                              # fibre; DCN and IO by their own index.


def _mpl():
    """Import matplotlib set up to save files without a display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _window(log, window_s=None):
    """(start, end) in ms of the last window_s seconds of the run (or the whole run if None)."""
    t_end = float(log.duration_ms)
    if window_s is None:
        return 0.0, t_end
    return max(0.0, t_end - window_s * 1000.0), t_end


# --- metrics ---------------------------------------------------------------

def spike_rate_hz(trains, t_start_ms=None, t_end_ms=None):
    """Firing rate of each cell (Hz) between t_start_ms and t_end_ms: spikes / window length."""
    rates = []
    for s in trains:
        s = np.asarray(s, dtype=float)
        lo = s.min() if t_start_ms is None and len(s) else (t_start_ms or 0.0)
        hi = t_end_ms if t_end_ms is not None else (s.max() if len(s) else 0.0)
        window_s = (hi - lo) / 1000.0
        n = int(((s >= lo) & (s <= hi)).sum())
        rates.append(n / window_s if window_s > 0 else 0.0)
    return np.asarray(rates)


def isi_stats(trains):
    """For each cell, the mean time between spikes (ISI, ms) and how variable
    it is (CV = std / mean). CV near 0 = regular like a clock; near 1 = random.
    Cells with fewer than 3 spikes get nan."""
    means, cvs = [], []
    for s in trains:
        s = np.asarray(s, dtype=float)
        if len(s) < 3:
            means.append(float("nan")); cvs.append(float("nan")); continue
        isi = np.diff(s)
        means.append(float(isi.mean())); cvs.append(float(isi.std() / isi.mean()))
    return np.asarray(means), np.asarray(cvs)


def weight_drift_slope(log, group=None):
    """Slope of a straight line fitted to the mean weight over time (weight
    units per second). Near 0 means the weights have stopped changing.

    The fit covers the whole run, including the early approach to equilibrium,
    so it only means something on runs of 300 s or more."""
    t_s = log.t_ms / 1000.0
    w = log.mean_weight.mean(axis=1) if group is None else log.mean_weight[:, group]
    slope, _ = np.polyfit(t_s, w, 1)
    return float(slope)


def is_saturated(log, group=None, tol=0.02, w_min=0.0, w_max=1.0, tail_frac=0.2):
    """True if, over the last tail_frac of the run, the mean weight is stuck at
    (within tol of) w_min or w_max. A run stuck at a limit has not found an
    equilibrium: the weights wanted to go further but were clipped."""
    w = log.mean_weight.mean(axis=1) if group is None else log.mean_weight[:, group]
    tail = w[int(len(w) * (1 - tail_frac)):]
    return bool(tail.mean() <= w_min + tol or tail.mean() >= w_max - tol)


def predicted_ltd_ltp_ratio(ltd_window_ms, equilibrium_interval_ms=1000.0):
    """The delta_minus/delta_plus ratio that would make the loop settle at a
    given CF interval. The inverse of the formula run.py uses for its
    prediction. Not called by run.py."""
    return (equilibrium_interval_ms - ltd_window_ms) / ltd_window_ms


def summarize(log, tail_frac=0.5):
    """The numbers printed at the end of a run, as a dict.

    Rates and ISI CV use only the last tail_frac (default: second half) of the
    run, after the weights have mostly settled."""
    t_end = float(log.duration_ms)
    lo = t_end * (1.0 - tail_frac)                            # start of the analysis window
    io_r = spike_rate_hz(log.io_spikes, lo, t_end)
    _, io_cv = isi_stats([s[s >= lo] for s in log.io_spikes])
    # Average the CV over cells that have one; if none do (e.g. a very short run), report nan.
    mean_cv = float(np.nanmean(io_cv)) if np.any(np.isfinite(io_cv)) else float("nan")
    return {
        "pkj_rate_hz": float(spike_rate_hz(log.pkj_spikes, lo, t_end).mean()),
        "dcn_rate_hz": float(spike_rate_hz(log.dcn_spikes, lo, t_end).mean()),
        "io_rate_hz": float(io_r.mean()),
        "io_rate_per_cell": io_r,                             # one rate per olive cell
        "io_isi_cv": mean_cv,
        "mean_weight": float(log.mean_weight[-1].mean()),     # at the last sample
        "weight_drift_slope": weight_drift_slope(log),
        "weight_saturated": is_saturated(log),
        "cross_synapse_std": float(log.final_weights.std()),  # spread of all 160,000 final weights
    }


# --- rasters ---------------------------------------------------------------

def _raster_panel(ax, trains, group_of_cell, n_groups, t0, t1, label, plt, markersize=1.2):
    """Draw one raster: a row per cell, a tick per spike, coloured by group."""
    cmap = plt.get_cmap(GROUP_CMAP)
    for i, s in enumerate(trains):
        s = np.asarray(s, dtype=float)
        s = s[(s >= t0) & (s <= t1)]
        if len(s):
            colour = cmap(int(group_of_cell[i]) % 10) if group_of_cell is not None else "black"
            ax.plot(s / 1000.0, np.full(len(s), i), "|", color=colour, markersize=markersize * 4,
                     markeredgewidth=markersize)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(-1, max(len(trains), 1))
    ax.set_xlim(t0 / 1000.0, t1 / 1000.0)
    ax.tick_params(labelsize=8)


def plot_rasters(log, path, window_s=10.0, title="Network rasters"):
    """Spike rasters for PF, PKJ, DCN and IO, stacked on one time axis.
    Shows only the last window_s seconds; a whole run would be a solid block."""
    plt = _mpl()
    t0, t1 = _window(log, window_s)
    n_groups = int(log.meta.get("n_io", 1))

    # Colour PF and PKJ by their climbing fibre. Each recorded PF's colour comes from the PKJ it belongs to.
    io_of_pkj = log.io_of_pkj if log.io_of_pkj is not None else log.group_of_pkj
    pf_group = io_of_pkj[log.pf_recorded[:, 0]] if log.pf_recorded is not None else None
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                              gridspec_kw={"height_ratios": [1.1, 1.6, 0.6, 0.5]})
    _raster_panel(axes[0], log.pf_spikes, pf_group, n_groups, t0, t1,
                   f"PF\n({len(log.pf_spikes)} of {log.meta.get('n_pf_per_pkj', 0) * len(log.pkj_spikes)})", plt, 0.9)
    _raster_panel(axes[1], log.pkj_spikes, io_of_pkj, n_groups, t0, t1,
                   f"PKJ\n({len(log.pkj_spikes)} cells)", plt, 0.7)
    _raster_panel(axes[2], log.dcn_spikes, log.group_of_dcn, n_groups, t0, t1,
                   f"DCN\n({len(log.dcn_spikes)} cells)", plt, 1.1)
    _raster_panel(axes[3], log.io_spikes, np.arange(len(log.io_spikes)), n_groups, t0, t1,
                   f"IO / CF\n({len(log.io_spikes)} cells)", plt, 2.2)

    axes[3].set_xlabel("time (s)")
    axes[0].set_title(f"{title}  --  last {(t1 - t0) / 1000.0:.0f} s of the run; "
                       "PF/PKJ coloured by climbing fiber, DCN/IO by group")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- membrane potentials ---------------------------------------------------

def plot_voltage_traces(log, path, n_pkj_shown=6, n_dcn_shown=4, title="Membrane potentials"):
    """Voltages over the last trace_window_s seconds, four panels:
    every IO cell, a few PKJ, a few DCN, and the DCN inhibition each IO receives.
    Traces are stacked with a vertical offset so they don't overlap."""
    plt = _mpl()
    t_s = log.trace_t_ms / 1000.0
    cmap = plt.get_cmap(GROUP_CMAP)
    n_io = log.trace_io_v.shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 0.8]})

    # Panel 1: each olive cell, shifted up 90 mV from the one before.
    for g in range(n_io):
        axes[0].plot(t_s, log.trace_io_v[:, g] + g * 90.0, color=cmap(g % 10), linewidth=0.8,
                      label=f"IO {g}" if g < 10 else None)
    axes[0].set_ylabel("IO $V_m$ (mV)\n+90 mV offset per cell", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=7, ncol=min(n_io, 8))
    axes[0].set_title(f"{title}  --  final {t_s[-1] - t_s[0]:.1f} s")

    # Panels 2 and 3: an evenly spaced handful of PKJ and DCN cells.
    pkj_idx = np.linspace(0, log.trace_pkj_v.shape[1] - 1, min(n_pkj_shown, log.trace_pkj_v.shape[1])).astype(int)
    for n, i in enumerate(pkj_idx):
        axes[1].plot(t_s, log.trace_pkj_v[:, i] + n * 75.0, linewidth=0.6,
                      color=cmap(int((log.io_of_pkj if log.io_of_pkj is not None
                                       else log.group_of_pkj)[i]) % 10))
    axes[1].set_ylabel(f"PKJ $V_m$ (mV)\n{len(pkj_idx)} of {log.trace_pkj_v.shape[1]}", fontsize=9)

    dcn_idx = np.linspace(0, log.trace_dcn_v.shape[1] - 1, min(n_dcn_shown, log.trace_dcn_v.shape[1])).astype(int)
    for n, i in enumerate(dcn_idx):
        axes[2].plot(t_s, log.trace_dcn_v[:, i] + n * 75.0, linewidth=0.6,
                      color=cmap(int(log.group_of_dcn[i]) % 10))
    axes[2].set_ylabel(f"DCN $V_m$ (mV)\n{len(dcn_idx)} of {log.trace_dcn_v.shape[1]}", fontsize=9)

    # Panel 4: DCN -> IO inhibitory conductance, one line per olive cell.
    for g in range(n_io):
        axes[3].plot(t_s, log.trace_io_gaba[:, g], color=cmap(g % 10), linewidth=0.7)
    axes[3].set_ylabel("DCN$\\to$IO $g_{GABA}$\n(mS/cm$^2$)", fontsize=9)
    axes[3].set_xlabel("time (s)")

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_io_state(log, path, title="IO membrane potential and calcium"):
    """Olive voltage (top) and calcium (bottom, log scale) over the last
    trace_window_s seconds. Each calcium spike shows as a jump in both."""
    plt = _mpl()
    t_s = log.trace_t_ms / 1000.0
    cmap = plt.get_cmap(GROUP_CMAP)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for g in range(log.trace_io_v.shape[1]):
        ax1.plot(t_s, log.trace_io_v[:, g], color=cmap(g % 10), linewidth=0.8, alpha=0.85, label=f"IO {g}")
        ax2.plot(t_s, log.trace_io_ca[:, g], color=cmap(g % 10), linewidth=0.8, alpha=0.85)
    ax1.set_ylabel("IO $V_m$ (mV)")
    ax1.legend(loc="upper right", fontsize=7, ncol=4)
    ax1.set_title(title)
    ax2.set_ylabel(r"IO [Ca$^{2+}$]$_i$ ($\mu$M)")
    ax2.set_yscale("log")
    ax2.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- weights ---------------------------------------------------------------

def plot_weights(log, path, title="PF->PKJ weights"):
    """Top: mean weight for each climbing fibre's group of PKJ, plus the overall
    mean (black). Bottom: the individually tracked synapses. They wander much
    more than the mean does, because the mean averages over 160,000 synapses."""
    plt = _mpl()
    t_s = log.t_ms / 1000.0
    cmap = plt.get_cmap(GROUP_CMAP)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for g in range(log.mean_weight.shape[1]):
        ax1.plot(t_s, log.mean_weight[:, g], color=cmap(g % 10), linewidth=1.2, label=f"group {g}")
    ax1.plot(t_s, log.mean_weight.mean(axis=1), color="black", linewidth=2.2, label="network mean")
    ax1.set_ylabel("mean PF$\\to$PKJ weight")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right", fontsize=7, ncol=3)
    ax1.set_title(title)

    if log.sample_weights is not None and log.sample_weights.size:
        tab20 = plt.get_cmap("tab20")
        for i in range(log.sample_weights.shape[1]):
            ax2.plot(t_s, log.sample_weights[:, i], color=tab20(i % 20), alpha=0.65, linewidth=0.8)
        ax2.plot(t_s, log.mean_weight.mean(axis=1), color="black", linewidth=2.2, label="network mean")
        ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylabel(f"individual synapses\n(n={0 if log.sample_weights is None else log.sample_weights.shape[1]})")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_individual_and_mean_weights(log, title, path, group=0):
    """Individual synapses and the mean of one group in a single panel. Not called by run.py."""
    plt = _mpl()
    sample = log.sample_weights
    if sample is None or sample.size == 0:
        raise ValueError("no tracked synapses in this log -- set cfg.n_tracked_synapses > 0 before run()")
    t_s = log.t_ms / 1000.0
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab20")
    for i in range(sample.shape[1]):
        ax.plot(t_s, sample[:, i], color=cmap(i % 20), alpha=0.6, linewidth=0.8)
    ax.plot(t_s, log.mean_weight[:, group], color="black", linewidth=2.5, label="mean weight")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("PF->PKJ weight")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- plots for experiments run.py doesn't do (parameter sweeps, single olive cells) ---

def plot_ratio_sweep(ratios, drift_slopes, predicted_ratio, title, path):
    """Weight drift against the delta_minus/delta_plus ratio, with the predicted zero point."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, drift_slopes, "o-", color="tab:blue", label="measured drift slope")
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.axvline(predicted_ratio, color="tab:green", linestyle="--",
               label=f"analytical zero-drift ratio ({predicted_ratio:.2f})")
    ax.set_xlabel("delta_minus / delta_plus ratio")
    ax.set_ylabel("weight drift slope (units/s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_io_membrane(traces, title, path, v_spike_mv=None, cf_times_ms=None):
    """Voltage and calcium of a single olive cell, for one or more conditions.
    `traces` is a list of (label, t_ms, v_mv, ca_um)."""
    plt = _mpl()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, (label, t_ms, v_mv, ca_um) in enumerate(traces):
        ax1.plot(t_ms / 1000.0, v_mv, color=cmap(i), linewidth=1.0, label=label)
        ax2.plot(t_ms / 1000.0, ca_um, color=cmap(i), linewidth=1.0, label=label)

    if v_spike_mv is not None:
        ax1.axhline(v_spike_mv, color="gray", linestyle="--", linewidth=1,
                     label=f"CF threshold ({v_spike_mv:.0f} mV)")
    if cf_times_ms is not None and len(cf_times_ms):
        ax1.plot(np.asarray(cf_times_ms) / 1000.0,
                  np.full(len(cf_times_ms), v_spike_mv if v_spike_mv is not None else 0.0),
                  "v", color="black", markersize=5, label="CF event")

    ax1.set_ylabel("IO $V_m$ (mV)")
    ax1.set_title(title)
    ax1.legend(loc="upper right", fontsize=8)
    ax2.set_ylabel(r"IO [Ca$^{2+}$]$_i$ ($\mu$M)")
    ax2.set_xlabel("time (s)")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_io_transfer_curve(curves, title, path, operating_point=None):
    """Olive firing rate against the amount of DCN inhibition: how the olive
    responds to the feedback. `curves` is a list of (label, g_gaba, rate_hz)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = [("o-", "tab:blue"), ("s-", "tab:purple"), ("^-", "tab:brown")]
    for i, (label, g_gaba, rate_hz) in enumerate(curves):
        marker, colour = styles[i % len(styles)]
        ax.plot(g_gaba, rate_hz, marker, color=colour, label=label)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="1 Hz equilibrium target")
    if operating_point is not None:
        ax.axvline(operating_point, color="tab:green", linestyle="--",
                    label=f"closed-loop operating point (mean $g$={operating_point:.2f})")
    ax.set_xlabel(r"DCN GABA$_A$ conductance on IO, time-average (mS/cm$^2$)")
    ax.set_ylabel("CF rate (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
