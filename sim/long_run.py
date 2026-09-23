"""The run loop for hour-scale simulations.

Same physics as `Simulation.run` -- it calls the very same `Simulation._step`,
so there is one implementation of the circuit and this module cannot drift from
it -- with the recording layer swapped for the streaming one in
`sim/recording.py` and three changes that only matter at length:

  * Spikes go to disk as they happen (see sim/recording.py for why).
  * Membrane potentials are recorded in SEVERAL short windows placed through the
    run, not one at the end. Over 5 hours the interesting comparison is early
    against late -- the weights are still moving for the first few hundred
    seconds -- and a single end-window cannot show that.
  * The slow weight trace runs at a 1 s cadence rather than 10 ms. 18,000 rows
    of 10,000 tracked synapses is 720 MB; at the 10 ms default it would be
    72 GB, and nothing in the weight dynamics moves fast enough to need it.

Progress goes to stdout with an ETA, because the runs this is for take hours and
a silent process for that long is indistinguishable from a hung one.
"""
import json
import time
from pathlib import Path

import numpy as np

from sim.recording import SpikeStream


def run_long(sim, out_dir, duration_s=None, burn_in_s=None, record_every_s=1.0,
             trace_windows_s=None, trace_len_s=10.0, progress_every_s=300.0, label=""):
    """Run `sim` for `duration_s` simulated seconds, writing a RunBundle to `out_dir`.

    trace_windows_s: start times (simulated seconds) of the membrane-potential
    windows. Defaults to one at the very start, one at the middle and one ending
    at the run's end, which brackets the weight transient.
    """
    cfg = sim.cfg
    duration_s = cfg.duration_s if duration_s is None else duration_s
    burn_in_s = cfg.burn_in_s if burn_in_s is None else burn_in_s
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = cfg.dt_ms
    n_steps = int(round(duration_s * 1000.0 / dt))
    record_every = max(1, int(round(record_every_s * 1000.0 / dt)))
    n_trace = int(round(trace_len_s * 1000.0 / dt))
    if trace_windows_s is None:
        trace_windows_s = [0.0, duration_s / 2.0, max(0.0, duration_s - trace_len_s)]
    trace_windows_s = [float(w) for w in trace_windows_s if w + trace_len_s <= duration_s + 1e-9]
    trace_starts = [int(round(w * 1000.0 / dt)) for w in trace_windows_s]

    n_io, n_pkj, n_dcn = cfg.n_io, sim.conn.n_pkj, sim.conn.n_dcn
    sim._choose_recorded()
    pf_rows, pf_cols = sim.pf_recorded[:, 0], sim.pf_recorded[:, 1]
    tr_rows, tr_cols = sim.tracked_synapses[:, 0], sim.tracked_synapses[:, 1]
    n_tracked = len(sim.tracked_synapses)

    # --- burn-in: settle the cells before plasticity or recording start (see sim/simulate.py) ---
    t_wall0 = time.time()
    if burn_in_s > 0:
        sim._plasticity_on = False
        for _ in range(int(round(burn_in_s * 1000.0 / dt))):
            sim._step()
        sim._plasticity_on = True
    sim.t_ms = 0.0
    print(f"[{label}] burn-in {burn_in_s:.0f} s done in {time.time() - t_wall0:.0f} s wall; "
          f"{n_steps:,} steps to go", flush=True)

    streams = {"io": SpikeStream("io", n_io, out_dir),
               "pkj": SpikeStream("pkj", n_pkj, out_dir),
               "dcn": SpikeStream("dcn", n_dcn, out_dir),
               "pf": SpikeStream("pf", len(sim.pf_recorded), out_dir)}

    n_slow = (n_steps + record_every - 1) // record_every
    t_slow = np.empty(n_slow, dtype=np.float64)
    mean_w = np.empty((n_slow, n_io), dtype=np.float32)        # per climbing-fiber territory
    pkj_mean_w = np.empty((n_slow, n_pkj), dtype=np.float32)   # per Purkinje cell
    sample_w = np.empty((n_slow, n_tracked), dtype=np.float32) # individually tracked synapses
    pkj_group_slices = [np.flatnonzero(sim.io_of_pkj == i) for i in range(n_io)]

    traces = [{"t_ms": np.empty(n_trace), "io_v": np.empty((n_trace, n_io), dtype=np.float32),
               "io_ca": np.empty((n_trace, n_io), dtype=np.float32),
               "io_gaba": np.empty((n_trace, n_io), dtype=np.float32),
               "pkj_v": np.empty((n_trace, n_pkj), dtype=np.float32),
               "dcn_v": np.empty((n_trace, n_dcn), dtype=np.float32)} for _ in trace_starts]

    slow_i = 0
    t_wall0 = time.time()
    next_report = progress_every_s
    for i in range(n_steps):
        pf_spikes, pkj_spiked, dcn_spiked, cf_events = sim._step()
        sim.t_ms += dt

        # Step i+1, not i: sim.t_ms was just advanced, so this step's spikes happened AT
        # (i+1)*dt. Matching Simulation.run's convention keeps times comparable across the
        # two recorders rather than off by one dt.
        streams["io"].record(i + 1, cf_events)
        streams["pkj"].record(i + 1, pkj_spiked)
        streams["dcn"].record(i + 1, dcn_spiked)
        streams["pf"].record(i + 1, pf_spikes[pf_rows, pf_cols])

        if i % record_every == 0:
            t_slow[slow_i] = sim.t_ms
            row_means = sim.weights.mean(axis=1)
            pkj_mean_w[slow_i] = row_means
            for g, sl in enumerate(pkj_group_slices):
                mean_w[slow_i, g] = row_means[sl].mean()
            sample_w[slow_i] = sim.weights[tr_rows, tr_cols]
            slow_i += 1

        for k, start in enumerate(trace_starts):
            if start <= i < start + n_trace:
                j, tr = i - start, traces[k]
                tr["t_ms"][j] = sim.t_ms
                tr["io_v"][j], tr["io_ca"][j], tr["io_gaba"][j] = sim.io.V, sim.io.Ca, sim.io_gaba.g
                # An I&F cell resets rather than producing an upstroke; painting v_peak on spiking
                # steps makes the trace read as a spike train (same convention as sim/simulate.py).
                tr["pkj_v"][j] = np.where(pkj_spiked, cfg.pkj.v_peak_mv, sim.pkj.V)
                tr["dcn_v"][j] = np.where(dcn_spiked, cfg.dcn.v_peak_mv, sim.dcn.V)

        elapsed = time.time() - t_wall0
        if elapsed >= next_report:
            frac = (i + 1) / n_steps
            print(f"[{label}] {sim.t_ms / 1000.0:8.0f} / {duration_s:.0f} s sim "
                  f"({100 * frac:5.1f}%) | {elapsed / 60:6.1f} min wall | "
                  f"ETA {(elapsed / frac - elapsed) / 60:6.1f} min | "
                  f"mean w {mean_w[max(0, slow_i - 1)].mean():.4f} | "
                  f"{streams['pkj'].n_spikes / 1e6:.0f}M PKJ spikes", flush=True)
            next_report = elapsed + progress_every_s

    wall_s = time.time() - t_wall0
    for s in streams.values():
        s.close()

    np.savez_compressed(out_dir / "slow.npz", t_s=t_slow[:slow_i] / 1000.0,
                        mean_weight=mean_w[:slow_i], pkj_mean_weight=pkj_mean_w[:slow_i],
                        sample_weights=sample_w[:slow_i], tracked_synapses=sim.tracked_synapses)
    for k, tr in enumerate(traces):
        np.savez_compressed(out_dir / f"trace_{k}.npz", **tr)
    np.savez_compressed(out_dir / "final.npz", final_weights=sim.weights.astype(np.float32),
                        gap_matrix=sim.gap_matrix, io_of_pkj=sim.io_of_pkj,
                        group_of_dcn=sim.group_of_dcn, pf_recorded=sim.pf_recorded)

    meta = {"label": label, "dt_ms": dt, "duration_s": duration_s, "burn_in_s": burn_in_s,
            "record_every_s": record_every_s, "trace_windows_s": trace_windows_s,
            "trace_len_s": trace_len_s, "n_io": n_io, "n_pkj": n_pkj, "n_dcn": n_dcn,
            "n_pf_recorded": len(sim.pf_recorded), "n_tracked_synapses": n_tracked,
            "n_pf_per_pkj": cfg.n_pf_per_pkj, "n_pkj_per_io": cfg.n_pkj_per_io,
            "pf_rate_hz": cfg.pf_rate_hz, "gap_g": cfg.gap_g, "gap_topology": cfg.gap_topology,
            "cf_pkj_reversal_mv": cfg.cf_pkj_reversal_mv, "ablate_dcn_io": cfg.ablate_dcn_io,
            "ltd_window_ms": cfg.ltd_window_ms, "null_window_ms": cfg.null_window_ms,
            "delta_plus": cfg.delta_plus, "delta_minus": cfg.delta_minus, "w_init": cfg.w_init,
            "seed": cfg.seed, "wall_s": wall_s,
            "n_spikes": {k: s.n_spikes for k, s in streams.items()},
            "connectivity": sim.conn.describe()}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[{label}] done: {duration_s:.0f} s sim in {wall_s / 3600:.2f} h wall "
          f"({wall_s / duration_s:.2f}x realtime); "
          f"{sum(s.n_spikes for s in streams.values()) / 1e6:.0f}M spikes -> {out_dir}", flush=True)
    return meta
