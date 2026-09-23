"""Analysis for the drift diagnostics (experiments/run_drift_diagnostics.py).

Read-only: consumes the .npz bundles and prints/plots the quantities each
candidate failure mode is judged on. Nothing here touches the model.
"""
import argparse, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def synchrony_index(ev_io, ev_t, n_io, t0, t1, bin_ms=20.0):
    edges = np.arange(t0, t1 + bin_ms, bin_ms)
    counts = np.zeros((n_io, len(edges) - 1))
    for i in range(n_io):
        m = (ev_io == i) & (ev_t >= t0) & (ev_t < t1)
        counts[i] = np.histogram(ev_t[m], bins=edges)[0]
    active = counts.std(axis=1) > 0
    counts = counts[active]
    if len(counts) < 2:
        return float("nan")
    c = np.corrcoef(counts)
    return float(c[~np.eye(len(c), dtype=bool)].mean())


def variance_decomposition(w_pkj_mean, w_pkj_sd, io_of_pkj):
    """(within_pkj, between_pkj_within_io, between_io) variances at one snapshot.

    Law of total variance over the nested grouping synapse < PKJ < IO territory.
    Every PKJ has the same number of synapses and every IO the same number of
    PKJ, so the group means are unweighted."""
    n_io = int(io_of_pkj.max()) + 1
    within = float((w_pkj_sd ** 2).mean())
    io_means = np.array([w_pkj_mean[io_of_pkj == i].mean() for i in range(n_io)])
    between_pkj = float(np.mean([w_pkj_mean[io_of_pkj == i].var() for i in range(n_io)]))
    between_io = float(io_means.var())
    return within, between_pkj, between_io


def analyze(path):
    d = np.load(path, allow_pickle=True)
    cfg = json.loads(str(d["cfg_json"]))
    dur = float(d["duration_s"]); bin_s = float(d["bin_s"])
    n_io = d["cf_bins"].shape[1]
    io_of_pkj = d["io_of_pkj"]
    ev_io, ev_t = d["ev_io"], d["ev_t"]
    half = len(d["cf_bins"]) // 2

    # --- A: per-IO rate ---
    rate_bins = d["cf_bins"] / bin_s                      # (n_bins, n_io) Hz
    rate_2h = rate_bins[half:].mean(axis=0)               # per-cell rate, second half
    pop_rate = rate_2h.mean()
    # within-cell temporal variability of the rate estimate, for a noise floor on the SD across cells
    sem_per_cell = rate_bins[half:].std(axis=0) / np.sqrt(len(rate_bins) - half)

    # --- B: synchrony ---
    sync = synchrony_index(ev_io, ev_t, n_io, dur * 1000 / 2, dur * 1000)
    tv = d["trace_v"]; ts = int(d["trace_steps"])
    late_v = tv[ts:]
    cv = np.corrcoef(late_v.T)
    vm_corr = float(cv[~np.eye(n_io, dtype=bool)].mean())
    # fraction of CF events that have another cell's CF within +-20 ms
    order = np.argsort(ev_t); et = ev_t[order]
    near = np.zeros(len(et), dtype=bool)
    j0 = np.searchsorted(et, et - 20.0, side="left")
    j1 = np.searchsorted(et, et + 20.0, side="right")
    near = (j1 - j0) > 1
    frac_near = float(near.mean())
    expected_near = 1 - np.exp(-(n_io - 1) * pop_rate * 0.040)   # chance level, independent Poisson cells

    # --- C: gap current ---
    gap = d["gap_mean"][half:]                            # (bins, n_io) mean I_gap per bin
    rate_dev = rate_bins[half:] - rate_bins[half:].mean(axis=1, keepdims=True)
    gap_flat, dev_flat = gap.ravel(), rate_dev.ravel()
    gap_rate_corr = float(np.corrcoef(gap_flat, dev_flat)[0, 1]) if gap_flat.std() > 0 else float("nan")
    # per-cell DC coupling current vs that cell's mean rate deviation
    cell_gap = gap.mean(axis=0); cell_dev = rate_2h - pop_rate
    cell_corr = float(np.corrcoef(cell_gap, cell_dev)[0, 1]) if cell_gap.std() > 0 else float("nan")
    igap_rms = float(np.sqrt((d["trace_igap"][ts:] ** 2).mean()))
    # reference scale: leak current swing, g_leak * (V - E_leak)
    v_late = late_v
    ileak_rms = float(np.sqrt((0.06 * (v_late + 60.0)) ** 2).mean())

    # --- D: burst / AHP ---
    capeak, isi_prev = d["ev_capeak"], d["ev_isi_prev"]
    ok = np.isfinite(capeak)
    ca_cv = float(capeak[ok].std() / capeak[ok].mean())
    # correlation between a spike's Ca peak and the interval to the NEXT spike of the same cell
    nxt = []
    for i in range(n_io):
        m = np.flatnonzero(ev_io == i)
        t_i = ev_t[m]; c_i = capeak[m]
        if len(t_i) > 2:
            nxt.append(np.stack([c_i[:-1], np.diff(t_i)]))
    nxt = np.concatenate(nxt, axis=1) if nxt else np.zeros((2, 0))
    good = np.isfinite(nxt[0]) & np.isfinite(nxt[1])
    ca_isi_corr = float(np.corrcoef(nxt[0][good], nxt[1][good])[0, 1]) if good.sum() > 10 else float("nan")
    isi_all = isi_prev[np.isfinite(isi_prev)]
    frac_burst = float((isi_all < 100.0).mean())          # multi-spike bursts, if any

    # --- E: weight variance decomposition over time ---
    snap_t, wm, wsd = d["snap_t"], d["snap_pkj_mean"], d["snap_pkj_sd"]
    decomp = np.array([variance_decomposition(wm[k], wsd[k], io_of_pkj) for k in range(len(snap_t))])
    within, btw_pkj, btw_io = decomp[:, 0], decomp[:, 1], decomp[:, 2]
    pinned = float(d["snap_frac_lo"][-1].mean() + d["snap_frac_hi"][-1].mean())

    # --- F: does a PKJ's weight track its own IO's rate? ---
    io_rate_of_pkj = rate_2h[io_of_pkj]
    dw = wm[-1] - wm[half if half < len(wm) else 0]
    r_dw_rate = float(np.corrcoef(io_rate_of_pkj, dw)[0, 1])
    r_w_rate = float(np.corrcoef(io_rate_of_pkj, wm[-1])[0, 1])

    # --- G: within-cell diffusion vs the rule's own prediction ---
    dm, dp = float(cfg["delta_minus"]), float(cfg["delta_plus"])
    ltd_ms = float(cfg["ltd_window_ms"]); pf_hz = float(cfg["pf_rate_hz"])
    p_ltd = min(1.0, pop_rate * ltd_ms / 1000.0)          # near-regular CF train: P ~ rate * window
    lam_ltd = pf_hz * p_ltd                               # LTD events per synapse per second
    pred_sd_unbounded = (dm + dp) * np.sqrt(lam_ltd * dur)

    return dict(
        label=str(d["label"]), path=path, duration_s=dur,
        pop_rate_hz=pop_rate, io_rate_sd=float(rate_2h.std()),
        io_rate_min=float(rate_2h.min()), io_rate_max=float(rate_2h.max()),
        io_rate_sem=float(sem_per_cell.mean()),
        sync_index=sync, vm_corr=vm_corr, frac_cf_near=frac_near, frac_cf_near_chance=float(expected_near),
        gap_rate_corr=gap_rate_corr, gap_cell_corr=cell_corr,
        igap_rms=igap_rms, ileak_rms=ileak_rms,
        ca_peak_cv=ca_cv, ca_isi_corr=ca_isi_corr, frac_isi_lt_100ms=frac_burst,
        sd_within=float(np.sqrt(within[-1])), sd_between_pkj=float(np.sqrt(btw_pkj[-1])),
        sd_between_io=float(np.sqrt(btw_io[-1])),
        sd_total=float(np.sqrt(within[-1] + btw_pkj[-1] + btw_io[-1])),
        frac_var_within=float(within[-1] / (within[-1] + btw_pkj[-1] + btw_io[-1])),
        frac_pinned=pinned,
        mean_weight=float(wm[-1].mean()),
        pred_sd_within_unbounded=float(pred_sd_unbounded),
        corr_pkjweight_iorate=r_w_rate, corr_dweight_iorate=r_dw_rate,
        _series=dict(t=snap_t, within=within, btw_pkj=btw_pkj, btw_io=btw_io,
                     rate_bins=rate_bins, wall_s=float(d["wall_s"])),
    )


GROUPS = [
    ("C1/C4  IO rate regulation", [
        ("pop Hz", "pop_rate_hz", "{:.4f}"), ("per-cell SD", "io_rate_sd", "{:.4f}"),
        ("min", "io_rate_min", "{:.3f}"), ("max", "io_rate_max", "{:.3f}"),
        ("bin SEM", "io_rate_sem", "{:.4f}")]),
    ("C1  coupling current", [
        ("I_gap rms", "igap_rms", "{:.4f}"), ("I_leak rms", "ileak_rms", "{:.4f}"),
        ("corr(I_gap, rate dev)", "gap_cell_corr", "{:+.3f}")]),
    ("C2  synchrony", [
        ("CF sync idx", "sync_index", "{:.3f}"), ("Vm corr", "vm_corr", "{:.3f}"),
        ("frac CF <20ms", "frac_cf_near", "{:.3f}"), ("chance", "frac_cf_near_chance", "{:.3f}")]),
    ("C3  burst / AHP", [
        ("Ca peak CV", "ca_peak_cv", "{:.4f}"), ("corr(Ca peak, next ISI)", "ca_isi_corr", "{:+.3f}"),
        ("frac ISI<100ms", "frac_isi_lt_100ms", "{:.4f}")]),
    ("C5/THE DRIFT  weight spread", [
        ("SD within PKJ", "sd_within", "{:.4f}"), ("SD between PKJ", "sd_between_pkj", "{:.4f}"),
        ("SD between IO", "sd_between_io", "{:.4f}"), ("var frac within", "frac_var_within", "{:.4f}"),
        ("free-diff pred", "pred_sd_within_unbounded", "{:.3f}"),
        ("frac at bound", "frac_pinned", "{:.4f}"), ("mean w", "mean_weight", "{:.4f}")]),
    ("F  does weight track its own IO's rate?", [
        ("corr(w, IO Hz)", "corr_pkjweight_iorate", "{:+.3f}"),
        ("corr(dw, IO Hz)", "corr_dweight_iorate", "{:+.3f}")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    paths = sorted(p for pat in a.paths for p in glob.glob(pat))
    rows = [analyze(p) for p in paths]
    labels = [r["label"] for r in rows]
    w0 = max(len(x) for x in labels + ["metric"]) + 2
    for title, metrics in GROUPS:
        print(f"\n### {title}")
        namew = max(len(m[0]) for m in metrics) + 2
        print(" " * namew + "".join(f"{l:>{w0}}" for l in labels))
        for name, key, fmt in metrics:
            cells = "".join(f"{fmt.format(r[key]):>{w0}}" for r in rows)
            print(f"{name:<{namew}}" + cells)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], f, indent=1)


if __name__ == "__main__":
    main()
