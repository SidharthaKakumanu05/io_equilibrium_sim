"""Aggregate sweep sidecars into replicated tables. Writes results/SWEEP_RESULTS.md.

Every row is mean +/- SD across seeds, because a single run is one draw from a
distribution and the point of the sweep is to stop reporting n=1. It reads only
results/sweeps/**.json, never the simulation, so it is instant and can be re-run
against a partial sweep -- experiments still in flight simply show fewer seeds
in their `n` column.

    python experiments/analyze_sweeps.py            # every experiment with sidecars
    python experiments/analyze_sweeps.py open_loop  # just one

`experiments/figures.py` is the graphical counterpart, reading the same sidecars
plus the .npz raw trajectories.
"""
import argparse
import collections
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "results" / "sweeps"


def load(experiment=None):
    recs = []
    for p in sorted(SWEEPS.glob("*/*.json")):
        if experiment and p.parent.name != experiment:
            continue
        try:
            recs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass                                   # a sidecar being written right now
    return recs


def group(recs, keys):
    """Bucket records by the listed override keys."""
    out = collections.defaultdict(list)
    for r in recs:
        out[tuple(r["overrides"].get(k) for k in keys)].append(r)
    return out


def stat(rows, metric):
    """(mean, sample SD, n) of one metric across a bucket, skipping runs where it
    is missing or non-finite -- a very short run has no defined ISI CV, and a run
    that silenced the olive has no defined synchrony."""
    v = np.array([r["metrics"][metric] for r in rows
                  if r["metrics"].get(metric) is not None
                  and np.isfinite(r["metrics"][metric])], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, v.size


def table(recs, keys, metrics, title, key_fmt=None):
    if not recs:
        return f"\n### {title}\n\n(no runs yet)\n"
    g = group(recs, keys)
    head = " | ".join(list(keys) + [f"{m}" for m in metrics] + ["n"])
    sep = "|".join(["---"] * (len(keys) + len(metrics) + 1))
    lines = [f"\n### {title}\n", f"| {head} |", f"|{sep}|"]
    for k in sorted(g, key=lambda t: tuple((x is None, x) for x in t)):
        cells = [key_fmt(k) if key_fmt else " | ".join(str(x) for x in k)]
        n = 0
        for m in metrics:
            mu, sd, n = stat(g[k], m)
            cells.append(f"{mu:+.4f} ± {sd:.4f}" if abs(mu) < 1 else f"{mu:.3f} ± {sd:.3f}")
        lines.append("| " + " | ".join(cells + [str(n)]) + " |")
    return "\n".join(lines) + "\n"


REPORTS = {
    "open_loop": lambda r: table(
        r, ["ablate_dcn_io", "io_heterogeneity_cv"],
        ["io_rate_hz", "mean_weight", "weight_drift_slope", "pkj_rate_hz", "dcn_rate_hz"],
        "Open-loop ablation: cutting DCN->IO (180 s runs, 12 seeds each)"),
    "initial_conditions": lambda r: table(
        r, ["w_init"],
        ["io_rate_hz", "mean_weight", "weight_drift_slope", "cross_synapse_std"],
        "Initial-condition independence: is the equilibrium an attractor? (180 s, 5 seeds)"),
    "heterogeneity_coupling": lambda r: table(
        r, ["io_heterogeneity_cv", "gap_g"],
        ["vm_correlation", "cf_synchrony", "io_rate_hz", "io_rate_spread_hz", "mean_weight"],
        "Heterogeneity x gap coupling (90 s, 6 seeds)"),
    "h2_window": lambda r: table(
        r, ["ltd_window_ms", "delta_minus"],
        ["io_rate_hz", "mean_weight", "weight_drift_slope", "weight_saturated"],
        "H2: plasticity window sets the equilibrium rate (120 s, 10 seeds)"),
    "h3_three_window": lambda r: table(
        r, ["null_window_ms"],
        ["io_rate_hz", "mean_weight", "weight_drift_slope", "cross_synapse_std"],
        "H3: three-window plasticity (120 s, 10 seeds)"),
    "settling": lambda r: table(
        r, ["io_heterogeneity_cv"],
        ["io_rate_hz", "mean_weight", "weight_drift_slope"],
        "Settling: 600 s runs"),
    "dt_convergence": lambda r: table(
        r, ["dt_ms"],
        ["io_rate_hz", "mean_weight", "pkj_rate_hz", "dcn_rate_hz", "io_isi_cv"],
        "Timestep convergence: does the outer loop's dt change the answer?"),
    "topology": lambda r: table(
        r, ["gap_topology", "io_heterogeneity_cv"],
        ["vm_correlation", "cf_synchrony", "gap_total_conductance", "io_rate_hz"],
        "Coupling topology at matched total conductance (90 s, 4 seeds)"),
    "pf_pool": lambda r: table(
        r, ["n_pf_per_pkj"],
        ["io_rate_hz", "mean_weight", "weight_drift_slope", "pkj_rate_hz"],
        "PF pool size: is 500 private fibers enough? (90 s, 5 seeds)"),
}


def h2_prediction_check(recs):
    """H2's real claim is that the equilibrium interval is window x (1 + d-/d+),
    so settings with the same product must give the same rate. Check predicted
    against measured rather than eyeballing the table."""
    if not recs:
        return ""
    lines = ["\n### H2: predicted vs measured equilibrium rate\n",
             "| LTD window | d-/d+ | predicted Hz | measured Hz | error | n |", "|---|---|---|---|---|---|"]
    g = group(recs, ["ltd_window_ms", "delta_minus"])
    for (w, dm) in sorted(g):
        dp = g[(w, dm)][0]["overrides"].get("delta_plus", 0.001)
        ratio = dm / dp
        predicted = 1000.0 / (w * (1.0 + ratio))
        mu, sd, n = stat(g[(w, dm)], "io_rate_hz")
        lines.append(f"| {w:.0f} ms | {ratio:.0f} | {predicted:.3f} | {mu:.3f} ± {sd:.3f} | "
                     f"{mu - predicted:+.3f} | {n} |")
    return "\n".join(lines) + "\n"


def vm_distance_profile(recs):
    """Vm correlation vs ring distance, which separates 'coupling too weak' from
    'coupling cannot reach across the olive'."""
    rows = [r for r in recs if r["metrics"].get("vm_correlation_by_ring_distance")]
    if not rows:
        return ""
    g = collections.defaultdict(list)
    for r in rows:
        key = (r["overrides"].get("gap_topology", "nearest_k"),
               r["overrides"].get("io_heterogeneity_cv", 0.0))
        g[key].append(r["metrics"]["vm_correlation_by_ring_distance"])
    lines = ["\n### Subthreshold Vm correlation by ring distance\n",
             "| topology | cv | d=1 | d=2 | d=5 | d=10 | d=20 |", "|---|---|---|---|---|---|---|"]
    for key in sorted(g):
        prof = collections.defaultdict(list)
        for d in g[key]:
            for k, v in d.items():
                prof[int(k)].append(v)
        cells = [f"{np.mean(prof[d]):+.3f}" if prof.get(d) else "-" for d in (1, 2, 5, 10, 20)]
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", default=None,
                    help=f"restrict the report to one experiment. "
                         f"Known: {', '.join(REPORTS)} (default: all of them)")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "SWEEP_RESULTS.md"),
                    help="markdown file to write (default: results/SWEEP_RESULTS.md)")
    args = ap.parse_args()

    parts = ["# Sweep results\n",
             "Every row is mean ± SD across seeds. Generated by "
             "`experiments/analyze_sweeps.py` from `results/sweeps/**.json`.\n"]
    counts = collections.Counter()
    for p in SWEEPS.glob("*/*.json"):
        counts[p.parent.name] += 1
    parts.append("\n| experiment | runs complete |\n|---|---|\n" +
                 "\n".join(f"| {k} | {v} |" for k, v in sorted(counts.items())) + "\n")

    for name, fn in REPORTS.items():
        if args.experiment and name != args.experiment:
            continue
        recs = load(name)
        parts.append(fn(recs))
        if name == "h2_window":
            parts.append(h2_prediction_check(recs))
        if name in ("topology", "heterogeneity_coupling"):
            parts.append(vm_distance_profile(recs))

    text = "\n".join(parts)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)     # results/ is gitignored, so it may not exist yet
    out_path.write_text(text)
    print(text)
    print(f"\n[written to {args.out}]")


if __name__ == "__main__":
    main()
