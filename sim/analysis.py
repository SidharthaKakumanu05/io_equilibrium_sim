"""Metrics and plotting for a SimLog.

Every rate here is counted from the recorded spike trains rather than read off a
smoothed trace, so "1.02 Hz" means 61 climbing-fiber events in 60 seconds and
not the current value of a low-pass filter.
"""
import numpy as np

GROUP_CMAP = "tab10"          # one colour per closed-loop group, shared across every raster and trace panel


def _mpl():
    import matplotlib
    matplotlib.use("Agg")                                     # headless backend, no display needed
    import matplotlib.pyplot as plt
    return plt


def _window(log, window_s=None):
    """(t_start_ms, t_end_ms) for the last `window_s` seconds of a run."""
    t_end = float(log.duration_ms)
    if window_s is None:
        return 0.0, t_end
    return max(0.0, t_end - window_s * 1000.0), t_end


# --- metrics ---------------------------------------------------------------

def spike_rate_hz(trains, t_start_ms=None, t_end_ms=None):
    """Mean firing rate (Hz) per cell over a window, as an array."""
    rates = []
    for s in trains:
        s = np.asarray(s, dtype=float)
        lo = s.min() if t_start_ms is None and len(s) else (t_start_ms or 0.0)
        hi = t_end_ms if t_end_ms is not None else (s.max() if len(s) else 0.0)
        window_s = (hi - lo) / 1000.0
        n = int(((s >= lo) & (s <= hi)).sum())
        rates.append(n / window_s if window_s > 0 else 0.0)
    return np.asarray(rates)


def cf_rate_hz(log, tail_frac=1.0):
    """Per-IO climbing-fiber rate over the last tail_frac of the run."""
    t_end = float(log.duration_ms)
    return spike_rate_hz(log.io_spikes, t_end * (1.0 - tail_frac), t_end)


def isi_stats(trains):
    """(mean ISI ms, CV) per cell. CV ~ 0 is a clock, CV ~ 1 is Poisson. The
    conductance IO lands in between: its spiking is gated by the subthreshold
    oscillation, so ISIs cluster near multiples of the oscillation period."""
    means, cvs = [], []
    for s in trains:
        s = np.asarray(s, dtype=float)
        if len(s) < 3:
            means.append(float("nan")); cvs.append(float("nan")); continue
        isi = np.diff(s)
        means.append(float(isi.mean())); cvs.append(float(isi.std() / isi.mean()))
    return np.asarray(means), np.asarray(cvs)


def weight_drift_slope(log, group=None):
    """Linear drift of the mean PF->PKJ weight, in weight units per second."""
    t_s = log.t_ms / 1000.0
    w = log.mean_weight.mean(axis=1) if group is None else log.mean_weight[:, group]
    slope, _ = np.polyfit(t_s, w, 1)
    return float(slope)


def is_saturated(log, group=None, tol=0.02, w_min=0.0, w_max=1.0, tail_frac=0.2):
    w = log.mean_weight.mean(axis=1) if group is None else log.mean_weight[:, group]
    tail = w[int(len(w) * (1 - tail_frac)):]
    return bool(tail.mean() <= w_min + tol or tail.mean() >= w_max - tol)


def predicted_ltd_ltp_ratio(ltd_window_ms, equilibrium_interval_ms=1000.0):
    return (equilibrium_interval_ms - ltd_window_ms) / ltd_window_ms  # analytical zero-drift delta_minus/delta_plus


def summarize(log, tail_frac=0.5):
    """One dict of the numbers a run is judged on."""
    t_end = float(log.duration_ms)
    lo = t_end * (1.0 - tail_frac)
    io_r = spike_rate_hz(log.io_spikes, lo, t_end)
    _, io_cv = isi_stats([s[s >= lo] for s in log.io_spikes])
    # A cell with fewer than 3 spikes in the window has no defined CV; if that is every cell
    # (a very short run, or one that silenced the olive) report nan rather than warn on an empty slice.
    mean_cv = float(np.nanmean(io_cv)) if np.any(np.isfinite(io_cv)) else float("nan")
    return {
        "pkj_rate_hz": float(spike_rate_hz(log.pkj_spikes, lo, t_end).mean()),
        "dcn_rate_hz": float(spike_rate_hz(log.dcn_spikes, lo, t_end).mean()),
        "io_rate_hz": float(io_r.mean()),
        "io_rate_per_cell": io_r,
        "io_isi_cv": mean_cv,
        "mean_weight": float(log.mean_weight[-1].mean()),
        "weight_drift_slope": weight_drift_slope(log),
        "weight_saturated": is_saturated(log),
        "cross_synapse_std": float(log.final_weights.std()),
    }


# --- rasters ---------------------------------------------------------------

def _raster_panel(ax, trains, group_of_cell, n_groups, t0, t1, label, plt, markersize=1.2):
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
    """PF / PKJ / DCN / IO spike rasters on a shared time axis, one colour per
    closed-loop group. Restricted to the last `window_s` seconds: at the default
    scale a whole run is a quarter-million spikes, which renders as a solid block."""
    plt = _mpl()
    t0, t1 = _window(log, window_s)
    n_groups = int(log.meta.get("n_io", 1))

    # PF and PKJ are coloured by the climbing fiber that owns them -- the unit plasticity
    # actually resolves against. Nuclear cells get their own colour each, since CbmSim's
    # wiring shares them across the whole microzone rather than assigning them a territory.
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
    """One panel per cell type over the high-resolution trace window, plus the
    DCN GABA conductance the IO is actually seeing. PKJ/DCN traces have their
    spikes painted at v_peak (see sim/simulate.py) -- an integrate-and-fire cell
    resets rather than producing an upstroke of its own, so the raw trace would
    be a sawtooth."""
    plt = _mpl()
    t_s = log.trace_t_ms / 1000.0
    cmap = plt.get_cmap(GROUP_CMAP)
    n_io = log.trace_io_v.shape[1]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1.3, 1.0, 1.0, 0.8]})

    for g in range(n_io):                                              # IO membrane potential, offset per cell
        axes[0].plot(t_s, log.trace_io_v[:, g] + g * 90.0, color=cmap(g % 10), linewidth=0.8,
                      label=f"IO {g}" if g < 10 else None)
    axes[0].set_ylabel("IO $V_m$ (mV)\n+90 mV offset per cell", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=7, ncol=min(n_io, 8))
    axes[0].set_title(f"{title}  --  final {t_s[-1] - t_s[0]:.1f} s")

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
    """IO V and [Ca]i together -- the Ca2+ spike and the transient that gates
    I_KCa, which is what paces the next one."""
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
    """Top: mean weight per group. Bottom: the individually tracked synapses
    against the network mean -- the spread between them is the random walk the
    three-window variant is meant to slow."""
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
    """Single-panel individual-vs-mean weight plot, used by the H2/H3 sweeps."""
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


# --- curves used by the single-cell / sweep experiments --------------------

def plot_ratio_sweep(ratios, drift_slopes, predicted_ratio, title, path):
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
    """Membrane potential and [Ca2+]i for one or more single-cell protocols.

    `traces` is a list of (label, t_ms, v_mv, ca_um) tuples so several
    conditions can be overlaid. Sample at 1 ms or finer -- coarser aliases the
    Ca2+ spike badly."""
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
    """CF rate as a function of the DCN GABA_A conductance on IO -- the loop's
    negative-feedback limb, which must be monotone decreasing for H1 to hold.

    `curves` is a list of (label, g_gaba, rate_hz), so the constant-conductance
    curve and the spike-driven one can be shown together. They are not the same
    curve, and the difference is the point: held constant, the olive's response
    to inhibition is nearly a cliff; delivered as real synaptic events the
    conductance fluctuates, the cell fires in the troughs, and the cliff becomes
    a gentle slope. The closed loop rides the second curve, not the first, so
    the operating point is marked against that one."""
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
