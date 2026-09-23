"""Instrumented run for diagnosing individual-synapse (PF->PKJ) weight drift.

INSTRUMENTATION ONLY -- this script does not modify the model. It drives
sim.Simulation._step() itself instead of Simulation.run() so it can record
quantities the normal SimLog does not carry:

  * per-IO CF counts in coarse time bins           -> per-cell rate vs population mean
  * per-IO net gap-junction current, binned        -> is coupling restoring outliers?
  * per-CF-event peak [Ca]i and the following ISI  -> burst-size-dependent AHP
  * per-PKJ mean weight AND within-cell weight SD  -> the variance decomposition
  * a high-resolution V/Ca/I_gap/g_GABA window     -> subthreshold phase analysis

Everything is written to one .npz per run; analysis lives in
experiments/analyze_drift.py.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimConfig
from sim.simulate import Simulation


def run(cfg, duration_s, out_path, bin_s=10.0, snap_s=10.0, trace_s=5.0, label=""):
    sim = Simulation(cfg)
    dt = cfg.dt_ms
    n_steps = int(round(duration_s * 1000.0 / dt))
    bin_steps = int(round(bin_s * 1000.0 / dt))
    snap_steps = int(round(snap_s * 1000.0 / dt))
    trace_steps = int(round(trace_s * 1000.0 / dt))
    n_io, n_pkj, n_dcn = cfg.n_io, sim.conn.n_pkj, sim.conn.n_dcn

    # --- burn-in (cells only; plasticity and recording off), exactly as Simulation.run does ---
    sim._plasticity_on = False
    for _ in range(int(round(cfg.burn_in_s * 1000.0 / dt))):
        sim._step()
    sim._plasticity_on = True
    sim.t_ms = 0.0

    n_bins = (n_steps + bin_steps - 1) // bin_steps
    cf_bins = np.zeros((n_bins, n_io), dtype=np.int32)
    pkj_bins = np.zeros((n_bins, n_pkj), dtype=np.int32)
    dcn_bins = np.zeros((n_bins, n_dcn), dtype=np.int32)
    gap_bins = np.zeros((n_bins, n_io))        # summed I_gap (uA/cm^2 * steps) -> mean below
    gaba_bins = np.zeros((n_bins, n_io))       # summed g_GABA seen by each IO
    v_bins = np.zeros((n_bins, n_io))          # summed V

    n_snaps = (n_steps + snap_steps - 1) // snap_steps + 1
    snap_t = np.zeros(n_snaps)
    snap_pkj_mean = np.zeros((n_snaps, n_pkj), dtype=np.float32)   # per-PKJ mean weight
    snap_pkj_sd = np.zeros((n_snaps, n_pkj), dtype=np.float32)     # within-cell SD of the 500 weights
    snap_frac_lo = np.zeros((n_snaps, n_pkj), dtype=np.float32)    # fraction pinned at w_min
    snap_frac_hi = np.zeros((n_snaps, n_pkj), dtype=np.float32)    # fraction pinned at w_max
    n_track = 400
    trk = np.random.default_rng(12345)
    trk_rows = trk.integers(0, n_pkj, n_track)
    trk_cols = trk.integers(0, cfg.n_pf_per_pkj, n_track)
    snap_track = np.zeros((n_snaps, n_track), dtype=np.float32)

    cf_times = [[] for _ in range(n_io)]

    # per-CF-event [Ca]i peak and the interval to the next event, for the AHP story
    ca_countdown = np.zeros(n_io, dtype=int)
    ca_peak = np.zeros(n_io)
    last_cf_ms = np.full(n_io, np.nan)
    ev_io, ev_t, ev_capeak, ev_isi_prev = [], [], [], []

    trace = {k: np.zeros((2 * trace_steps, n_io), dtype=np.float32)
             for k in ("v", "ca", "gaba", "igap")}
    trace_t = np.zeros(2 * trace_steps)
    late_start = n_steps - trace_steps

    snap_i = 0
    t0 = time.time()
    for i in range(n_steps):
        _, pkj_spiked, dcn_spiked, cf = sim._step()
        sim.t_ms += dt
        b = i // bin_steps

        cf_bins[b] += cf
        pkj_bins[b] += pkj_spiked
        dcn_bins[b] += dcn_spiked
        v = sim.io.V
        if sim.io.g_gap is not None:
            igap = sim.io.g_gap @ v - sim.io._gap_row_sum * v      # net current INTO cell i
        else:
            igap = np.zeros(n_io)
        gap_bins[b] += igap
        gaba_bins[b] += sim.io_gaba.g
        v_bins[b] += v

        if cf.any():
            for j in np.flatnonzero(cf):
                cf_times[j].append(sim.t_ms)
                ev_io.append(j); ev_t.append(sim.t_ms)
                ev_isi_prev.append(sim.t_ms - last_cf_ms[j])
                last_cf_ms[j] = sim.t_ms
            ca_countdown[cf] = 30
            ca_peak[cf] = 0.0
        act = ca_countdown > 0
        if act.any():
            np.maximum(ca_peak, np.where(act, sim.io.Ca, 0.0), out=ca_peak)
            ca_countdown[act] -= 1
            done = act & (ca_countdown == 0)
            for j in np.flatnonzero(done):
                ev_capeak.append((j, ca_peak[j]))

        if i % snap_steps == 0:
            w = sim.weights
            snap_t[snap_i] = sim.t_ms
            snap_pkj_mean[snap_i] = w.mean(axis=1)
            snap_pkj_sd[snap_i] = w.std(axis=1)
            snap_frac_lo[snap_i] = (w <= cfg.w_min + 1e-12).mean(axis=1)
            snap_frac_hi[snap_i] = (w >= cfg.w_max - 1e-12).mean(axis=1)
            snap_track[snap_i] = w[trk_rows, trk_cols]
            snap_i += 1

        k = None
        if i < trace_steps:
            k = i
        elif i >= late_start:
            k = trace_steps + (i - late_start)
        if k is not None:
            trace_t[k] = sim.t_ms
            trace["v"][k], trace["ca"][k] = v, sim.io.Ca
            trace["gaba"][k], trace["igap"][k] = sim.io_gaba.g, igap

    # final snapshot
    w = sim.weights
    snap_t[snap_i] = sim.t_ms
    snap_pkj_mean[snap_i] = w.mean(axis=1); snap_pkj_sd[snap_i] = w.std(axis=1)
    snap_frac_lo[snap_i] = (w <= cfg.w_min + 1e-12).mean(axis=1)
    snap_frac_hi[snap_i] = (w >= cfg.w_max - 1e-12).mean(axis=1)
    snap_track[snap_i] = w[trk_rows, trk_cols]
    snap_i += 1

    ev_capeak_arr = np.full(len(ev_t), np.nan)
    # ev_capeak entries are appended in completion order; match them back by cell
    seen = {}
    for (j, pk) in ev_capeak:
        seen.setdefault(j, []).append(pk)
    ptr = {j: 0 for j in seen}
    for idx, j in enumerate(ev_io):
        lst = seen.get(j)
        if lst is not None and ptr[j] < len(lst):
            ev_capeak_arr[idx] = lst[ptr[j]]; ptr[j] += 1

    np.savez_compressed(
        out_path,
        label=label, duration_s=duration_s, bin_s=bin_s, snap_s=snap_s,
        cf_bins=cf_bins, pkj_bins=pkj_bins, dcn_bins=dcn_bins,
        gap_mean=gap_bins / bin_steps, gaba_mean=gaba_bins / bin_steps, v_mean=v_bins / bin_steps,
        snap_t=snap_t[:snap_i], snap_pkj_mean=snap_pkj_mean[:snap_i], snap_pkj_sd=snap_pkj_sd[:snap_i],
        snap_frac_lo=snap_frac_lo[:snap_i], snap_frac_hi=snap_frac_hi[:snap_i],
        snap_track=snap_track[:snap_i], trk_rows=trk_rows, trk_cols=trk_cols,
        ev_io=np.asarray(ev_io, dtype=np.int32), ev_t=np.asarray(ev_t),
        ev_capeak=ev_capeak_arr, ev_isi_prev=np.asarray(ev_isi_prev),
        trace_t=trace_t, trace_v=trace["v"], trace_ca=trace["ca"],
        trace_gaba=trace["gaba"], trace_igap=trace["igap"], trace_steps=trace_steps,
        io_of_pkj=sim.io_of_pkj, gap_matrix=sim.gap_matrix,
        final_weights=sim.weights.astype(np.float32),
        cf_times=np.array([np.asarray(c) for c in cf_times], dtype=object),
        wall_s=time.time() - t0,
        cfg_json=json.dumps({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                             for k, v in vars(cfg).items()}),
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--gap-g", type=float, default=None)
    ap.add_argument("--gap-topology", default=None)
    ap.add_argument("--gap-n-neighbors", type=int, default=None)
    ap.add_argument("--het-cv", type=float, default=None)
    ap.add_argument("--io-noise-sigma", type=float, default=None)
    ap.add_argument("--open-loop", action="store_true", help="enforce_closed_loop=False (misroute DCN->IO)")
    ap.add_argument("--ablate", action="store_true", help="cut DCN->IO entirely")
    ap.add_argument("--delta-plus", type=float, default=None)
    ap.add_argument("--delta-minus", type=float, default=None)
    ap.add_argument("--null-window-ms", type=float, default=None)
    ap.add_argument("--weight-dependence", type=float, default=None)
    ap.add_argument("--n-dcn-per-io", type=int, default=None)
    ap.add_argument("--bin-s", type=float, default=10.0)
    ap.add_argument("--snap-s", type=float, default=10.0)
    a = ap.parse_args()

    cfg = SimConfig()
    cfg.seed = a.seed
    if a.gap_g is not None: cfg.gap_g = a.gap_g
    if a.gap_topology is not None: cfg.gap_topology = a.gap_topology
    if a.gap_n_neighbors is not None: cfg.gap_n_neighbors = a.gap_n_neighbors
    if a.het_cv is not None: cfg.io_heterogeneity_cv = a.het_cv
    if a.io_noise_sigma is not None: cfg.io_channels.noise_sigma = a.io_noise_sigma
    if a.open_loop: cfg.enforce_closed_loop = False
    if a.ablate: cfg.ablate_dcn_io = True
    if a.delta_plus is not None: cfg.delta_plus = a.delta_plus
    if a.delta_minus is not None: cfg.delta_minus = a.delta_minus
    if a.null_window_ms is not None: cfg.null_window_ms = a.null_window_ms
    if a.weight_dependence is not None: cfg.weight_dependence = a.weight_dependence
    if a.n_dcn_per_io is not None: cfg.n_dcn_per_io = a.n_dcn_per_io

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    run(cfg, a.duration_s, a.out, bin_s=a.bin_s, snap_s=a.snap_s,
        label=a.label or os.path.basename(a.out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
