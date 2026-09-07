"""Parallel condition sweeps with full provenance.

One process per condition, one JSON sidecar per run. Every sidecar carries the
complete resolved config, a hash of the source that produced it, the seeds, the
wall time, and every scalar the run is judged on -- so a result can be re-read,
re-plotted and re-checked months later without re-running anything, and two
results can always be compared on identical footing.

The sweep is RESUMABLE: a condition whose sidecar already exists is skipped, so
a crash, a reboot or a deliberate interrupt costs only the runs that were in
flight. Raw arrays (spike trains, weight trajectories, Vm traces) are written as
compressed .npz only for conditions that ask for them, because writing them for
every run of a large sweep costs far more disk than it saves re-computation.

    python experiments/sweep.py --list
    python experiments/sweep.py open_loop --workers 32
    python experiments/sweep.py all --workers 60
"""
import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

# One BLAS thread per process: these arrays are tiny, and letting each of 60
# workers spawn its own thread pool oversubscribes the machine and slows
# everything down.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config import SimConfig                                    # noqa: E402
from sim.simulate import Simulation                             # noqa: E402
from sim.analysis import summarize                              # noqa: E402
from sim.io_coupling import synchrony_index                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "sweeps"


# --- provenance ------------------------------------------------------------

def source_fingerprint():
    """SHA256 over every source file that can change a result. This project is not
    a git repo, so the code itself is the version stamp."""
    h = hashlib.sha256()
    for rel in sorted(["config.py"] + [f"sim/{p.name}" for p in sorted((ROOT / "sim").glob("*.py"))]):
        h.update(rel.encode())
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()[:16]


def config_to_dict(cfg):
    """Fully resolved config, nested dataclasses expanded, JSON-safe."""
    def conv(v):
        if dataclasses.is_dataclass(v):
            return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    return {f.name: conv(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}


# --- metrics ---------------------------------------------------------------

def subthreshold_vm_stats(log, n_io, v_max_mv=-50.0, min_samples=500):
    """Global mean pairwise subthreshold Vm correlation, plus the same resolved by
    distance around the IO ring. The distance profile is what distinguishes 'the
    coupling is weak' from 'the coupling is strong but cannot reach across the
    olive' -- they give the same global number and mean opposite things."""
    v = log.trace_io_v
    if v is None or v.shape[1] < 2:
        return float("nan"), {}
    below = v < v_max_mv
    allr, by_d = [], {}
    for i in range(n_io):
        for j in range(i + 1, n_io):
            both = below[:, i] & below[:, j]
            if int(both.sum()) <= min_samples:
                continue
            r = float(np.corrcoef(v[both, i], v[both, j])[0, 1])
            if not np.isfinite(r):
                continue
            allr.append(r)
            d = min(abs(i - j), n_io - abs(i - j))
            by_d.setdefault(d, []).append(r)
    return (float(np.mean(allr)) if allr else float("nan"),
            {str(d): float(np.mean(rs)) for d, rs in sorted(by_d.items())})


def metrics(log, cfg, sim):
    s = summarize(log)
    half = float(log.duration_ms) / 2.0
    vm_r, vm_by_d = subthreshold_vm_stats(log, cfg.n_io)
    g_cal = np.atleast_1d(sim.io._g_cal)
    per_cell = np.asarray(s.pop("io_rate_per_cell"), dtype=float)
    out = dict(s)
    out.update({
        "io_rate_per_cell": per_cell.tolist(),
        "io_rate_spread_hz": float(per_cell.std()),
        "cf_synchrony": float(synchrony_index(log.io_spikes, half, log.duration_ms)),
        "vm_correlation": vm_r,
        "vm_correlation_by_ring_distance": vm_by_d,
        "mean_io_gaba": (float(np.mean(log.trace_io_gaba))
                         if log.trace_io_gaba is not None else float("nan")),
        "realized_g_cal_cv": (float(g_cal.std() / g_cal.mean()) if g_cal.size > 1 else 0.0),
        "gap_total_conductance": float(np.asarray(sim.gap_matrix).sum(axis=1).mean()),
        "gap_degree": float((np.asarray(sim.gap_matrix) > 0).sum(axis=1).mean()),
        "dcn_io_drive": float(sim.dcn_io_drive),
        "n_pkj": int(sim.conn.n_pkj),
    })
    return out


# --- one run ---------------------------------------------------------------

def apply_overrides(cfg, overrides):
    """Set config fields from a flat dict. Dotted keys reach nested dataclasses
    ('io_channels.noise_sigma'). SimConfig rejects unknown fields, so a typo in a
    condition raises here instead of silently running the default."""
    for key, value in overrides.items():
        target, _, leaf = key.rpartition(".")
        obj = cfg
        for part in filter(None, target.split(".")):
            obj = getattr(obj, part)
        setattr(obj, leaf, value)
    return cfg


def run_condition(cond):
    """Run one condition and write its sidecar. Returns a short status string."""
    name = cond["_name"]
    out_dir = OUT_ROOT / cond["_experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    if json_path.exists():
        return f"skip  {name}"

    overrides = {k: v for k, v in cond.items() if not k.startswith("_")}
    try:
        cfg = apply_overrides(SimConfig(), overrides)
        t0 = time.time()
        sim = Simulation(cfg)
        log = sim.run()
        elapsed = time.time() - t0
        rec = {
            "name": name,
            "experiment": cond["_experiment"],
            "overrides": overrides,
            "metrics": metrics(log, cfg, sim),
            "config": config_to_dict(cfg),
            "source_fingerprint": source_fingerprint(),
            "wall_seconds": round(elapsed, 1),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = json_path.with_suffix(".json.tmp")          # atomic: a killed run leaves no half sidecar
        tmp.write_text(json.dumps(rec, indent=2))
        tmp.replace(json_path)

        if cond.get("_save_raw"):
            np.savez_compressed(
                out_dir / f"{name}.npz",
                io_spikes=np.array([np.asarray(t) for t in log.io_spikes], dtype=object),
                dcn_spikes=np.array([np.asarray(t) for t in log.dcn_spikes], dtype=object),
                pkj_spikes=np.array([np.asarray(t) for t in log.pkj_spikes], dtype=object),
                t_ms=log.t_ms, mean_weight=log.mean_weight, sample_weights=log.sample_weights,
                trace_t_ms=log.trace_t_ms, trace_io_v=log.trace_io_v,
                trace_io_gaba=log.trace_io_gaba, final_weights=log.final_weights,
                gap_matrix=log.gap_matrix)
        m = rec["metrics"]
        return (f"done  {name}  IO {m['io_rate_hz']:.2f} Hz  w {m['mean_weight']:.3f}  "
                f"drift {m['weight_drift_slope']:+.5f}  sync {m['cf_synchrony']:.3f}  [{elapsed:.0f}s]")
    except Exception as exc:                              # one bad condition must not kill the sweep
        (out_dir / f"{name}.FAILED.txt").write_text(f"{type(exc).__name__}: {exc}")
        return f"FAIL  {name}: {type(exc).__name__}: {exc}"


# --- experiment matrices ---------------------------------------------------
#
# Common settings: trace_window_s is kept long enough that the subthreshold Vm
# correlation has thousands of samples per pair, and burn_in_s is left at the
# default so plasticity starts from a settled membrane state.

def _name(**kw):
    return "__".join(f"{k}={v}" for k, v in kw.items())


def exp_open_loop():
    """THE ABLATION. Cut DCN->IO and nothing else.

    H1 says the ~1 Hz CF rate is produced by the feedback limb. If so, removing
    only that limb must abolish it: CF rate should run free toward the cell's
    intrinsic rate and the weights, no longer balanced, should be driven to a
    bound. Run long (180 s) because the prediction is about where the weights
    END UP, not where they start. Both heterogeneity levels, so the result
    cannot be an artifact of identical cells."""
    out = []
    for ablate in (False, True):
        for cv in (0.0, 0.10):
            for seed in range(12):
                out.append({
                    "_experiment": "open_loop",
                    "_name": _name(ablate=int(ablate), cv=cv, seed=seed),
                    "_save_raw": seed == 0,
                    "ablate_dcn_io": ablate, "io_heterogeneity_cv": cv,
                    "seed": seed, "duration_s": 180.0, "trace_window_s": 10.0,
                })
    return out


def exp_initial_conditions():
    """The other half of H1, and the stronger half: an equilibrium must be an
    ATTRACTOR, not a starting point. Runs from five different initial weights;
    if the loop is really self-correcting they must all converge on the same
    CF rate and the same final weight regardless of where they began."""
    return [{
        "_experiment": "initial_conditions",
        "_name": _name(w_init=w, seed=seed),
        "_save_raw": seed == 0,
        "w_init": w, "seed": seed, "duration_s": 180.0, "trace_window_s": 10.0,
    } for w in (0.1, 0.3, 0.5, 0.7, 0.9) for seed in range(5)]


def exp_heterogeneity_coupling():
    """Heterogeneity x gap coupling, fully crossed.

    These two interact and cannot be read separately: coupling can only
    synchronize cells that would otherwise differ, so the effect of gap_g is
    conditional on cv, and the whole 'synchrony = 1.000' degeneracy lived at
    cv = 0. Crossing them measures that surface instead of assuming it."""
    return [{
        "_experiment": "heterogeneity_coupling",
        "_name": _name(cv=cv, g=g, seed=seed),
        "_save_raw": seed == 0 and cv in (0.0, 0.10),
        "io_heterogeneity_cv": cv, "gap_g": g,
        "seed": seed, "duration_s": 90.0, "trace_window_s": 15.0,
    } for cv in (0.0, 0.05, 0.10, 0.20)
      for g in (0.0, 0.00715, 0.0143, 0.0286, 0.0572)
      for seed in range(6)]


def exp_h2_window():
    """H2 re-validation at the current scale, with replication.

    The equilibrium interval should be window x (1 + delta-/delta+), so pairs of
    settings with the same product must land on the same rate. The table in the
    README is from an 8-IO build that no longer exists; this replaces it."""
    conds = [(100.0, 0.019, 0.001), (200.0, 0.009, 0.001), (100.0, 0.009, 0.001),
             (150.0, 0.009, 0.001), (100.0, 0.004, 0.001), (50.0, 0.009, 0.001),
             (100.0, 0.014, 0.001), (250.0, 0.009, 0.001)]
    return [{
        "_experiment": "h2_window",
        "_name": _name(win=w, dm=dm, seed=seed),
        "_save_raw": seed == 0,
        "ltd_window_ms": w, "delta_minus": dm, "delta_plus": dp,
        "seed": seed, "duration_s": 120.0, "trace_window_s": 8.0,
    } for (w, dm, dp) in conds for seed in range(10)]


def exp_h3_three_window():
    """H3: LTD -> null -> LTP. A null window should let delta-/delta+ sit closer
    together and slow the weights' random walk, at the cost of blocking some LTP.
    Never validated on a spiking network; this is its first real test."""
    return [{
        "_experiment": "h3_three_window",
        "_name": _name(null=nw, seed=seed),
        "_save_raw": seed == 0,
        "null_window_ms": nw, "seed": seed,
        "duration_s": 120.0, "trace_window_s": 8.0,
    } for nw in (0.0, 25.0, 50.0, 100.0, 200.0) for seed in range(10)]


def exp_settling():
    """How long does the loop actually take to settle? Every equilibrium number
    in the README is read off a 90 s run, and the current drift (-0.00085/s) says
    that is not long enough. 600 s runs, so the settling time can be measured
    rather than assumed."""
    return [{
        "_experiment": "settling",
        "_name": _name(cv=cv, seed=seed),
        "_save_raw": True,
        "io_heterogeneity_cv": cv, "seed": seed,
        "duration_s": 600.0, "trace_window_s": 10.0,
    } for cv in (0.0, 0.10) for seed in range(4)]


def exp_dt_convergence():
    """Numerical validity. The IO substep is verified against a 4x finer one, but
    the 1 ms OUTER loop -- which the plasticity windows, synaptic decay and the
    whole feedback limb run on -- has never been checked. If halving dt moves the
    equilibrium, the results are integration artifacts."""
    return [{
        "_experiment": "dt_convergence",
        "_name": _name(dt=dt, seed=seed),
        "_save_raw": False,
        "dt_ms": dt, "seed": seed, "duration_s": 90.0, "trace_window_s": 8.0,
    } for dt in (1.0, 0.5, 0.25) for seed in range(4)]


def exp_topology():
    """Coupling topology at MATCHED total conductance per cell. Measured earlier:
    directly coupled pairs already reach Vm r = +0.90 while distant pairs plateau
    at +0.44, so global synchrony is limited by path length, not by conductance.
    If that is right, small_world should beat nearest_k at identical cost."""
    out = []
    for topo, g, k in (("nearest_k", 0.0143, 2), ("small_world", 0.0143, 2),
                       ("ring", 0.0286, 1), ("all_to_all", 0.001467, 2)):
        for cv in (0.0, 0.10):
            for seed in range(4):
                out.append({
                    "_experiment": "topology",
                    "_name": _name(topo=topo, cv=cv, seed=seed),
                    "_save_raw": seed == 0,
                    "gap_topology": topo, "gap_g": g, "gap_n_neighbors": k,
                    "io_heterogeneity_cv": cv, "seed": seed,
                    "duration_s": 90.0, "trace_window_s": 15.0,
                })
    return out


def exp_pf_pool():
    """Sensitivity to the one scaled-down number. Each Purkinje cell gets 500
    private Poisson fibers instead of CbmSim's 32,768 shared granule cells. If
    the equilibrium moves with the pool size, the simplification is load-bearing
    and has to be reported as such."""
    return [{
        "_experiment": "pf_pool",
        "_name": _name(npf=n, seed=seed),
        "_save_raw": False,
        "n_pf_per_pkj": n, "seed": seed, "duration_s": 90.0, "trace_window_s": 8.0,
    } for n in (125, 250, 500, 1000, 2000) for seed in range(5)]


EXPERIMENTS = {
    "open_loop": exp_open_loop,
    "initial_conditions": exp_initial_conditions,
    "heterogeneity_coupling": exp_heterogeneity_coupling,
    "h2_window": exp_h2_window,
    "h3_three_window": exp_h3_three_window,
    "settling": exp_settling,
    "dt_convergence": exp_dt_convergence,
    "topology": exp_topology,
    "pf_pool": exp_pf_pool,
}

# Longest-running experiments first, so the slow tail starts early and the short
# runs backfill the cores instead of the other way round.
ORDER = ["settling", "open_loop", "initial_conditions", "dt_convergence",
         "h2_window", "h3_three_window", "heterogeneity_coupling",
         "topology", "pf_pool"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default="all",
                    help="experiment name, 'all', or a comma-separated list")
    ap.add_argument("--workers", type=int, default=max(1, min(60, (os.cpu_count() or 8) - 4)))
    ap.add_argument("--list", action="store_true", help="show the matrix and exit")
    ap.add_argument("--dry-run", action="store_true", help="count runs without running them")
    args = ap.parse_args()

    if args.list:
        total = 0
        for k in ORDER:
            n = len(EXPERIMENTS[k]())
            total += n
            print(f"  {k:24s} {n:4d} runs")
        print(f"  {'TOTAL':24s} {total:4d} runs")
        return

    names = ORDER if args.experiment == "all" else [s.strip() for s in args.experiment.split(",")]
    conds = []
    for k in names:
        if k not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {k!r}; known: {', '.join(EXPERIMENTS)}")
        conds.extend(EXPERIMENTS[k]())

    pending = [c for c in conds
               if not (OUT_ROOT / c["_experiment"] / f"{c['_name']}.json").exists()]
    print(f"[sweep] {len(conds)} conditions, {len(conds) - len(pending)} already done, "
          f"{len(pending)} to run on {args.workers} workers", flush=True)
    if args.dry_run:
        return

    t0 = time.time()
    with mp.Pool(processes=args.workers, maxtasksperchild=4) as pool:
        for i, line in enumerate(pool.imap_unordered(run_condition, pending), 1):
            print(f"[{i}/{len(pending)}] {line}  (+{(time.time()-t0)/60:.0f} min)", flush=True)
    print(f"[sweep] finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
