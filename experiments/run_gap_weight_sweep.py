#!/usr/bin/env python3
"""What does IO-IO electrical coupling do to the SPREAD of per-cell weights?

experiments/run_gap_sweep.py asks what the gap junctions do to the loop's rates
and synchrony. This asks the downstream question: does coupling the olive pull
the 320 Purkinje cells' mean PF->PKJ weights together, or push them apart?

The quantity plotted is the mean weight of each Purkinje cell's own 500 PF
synapses -- 320 traces per condition, not the 40 climbing-fiber territory means
of run_baseline.py and not individual synapses. That is the right level for the
question: a cell's mean weight is what sets its firing rate, so it is the only
part of the weight vector the loop can see or act on. The spread WITHIN a cell
is a free random walk that no amount of coupling touches (see the README).

Four conditions, one seed each:

    gap_g = 0        uncoupled -- each olivary cell on its own
    gap_g = 0.005    weak      -- about a third of the fitted value
    gap_g = 0.0143   moderate  -- config.SimConfig.gap_g, the shipped default
    gap_g = 0.04     strong    -- ~2.8x default, ~2.3x g_leak per cell at 6 neighbours

Writes results/gap_weight_sweep.png and prints the summary table.
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import summarize
from sim.io_coupling import synchrony_index
from sim.simulate import Simulation

CONDITIONS = [("off", 0.0), ("weak", 0.005), ("moderate", 0.0143), ("strong", 0.04)]


def one_run(args):
    label, gap_g, duration_s, seed, record_every_ms = args
    cfg = SimConfig(duration_s=duration_s, seed=seed, n_tracked_synapses=0,
                     record_every_ms=record_every_ms)
    cfg.gap_g = gap_g
    log = Simulation(cfg).run()

    s = summarize(log)
    w = log.pkj_mean_weight                                   # (T, n_pkj)
    half = log.duration_ms / 2.0
    tail = log.t_ms >= log.t_ms[-1] * 0.8                      # last 20%, after the transient
    final = log.final_weights
    return dict(
        label=label, gap_g=gap_g,
        t_s=log.t_ms / 1000.0,
        pkj_mean_weight=w,
        between_sd=w.std(axis=1),                              # spread of the 320 cell means, over time
        between_sd_final=float(w[tail].std(axis=1).mean()),
        within_sd_final=float(final.std(axis=1).mean()),       # spread inside a cell, for contrast
        network_mean=w.mean(axis=1),
        sync=synchrony_index(log.io_spikes, half, log.duration_ms),
        io_hz=s["io_rate_hz"], pkj_hz=s["pkj_rate_hz"], dcn_hz=s["dcn_rate_hz"],
        drift=s["weight_drift_slope"],
    )


def plot(results, path, duration_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.28, wspace=0.16)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    lo = min(r["pkj_mean_weight"].min() for r in results)
    hi = max(r["pkj_mean_weight"].max() for r in results)
    pad = 0.06 * (hi - lo)

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        # 320 traces: thin and translucent, so density reads as darkness rather than as 320 legend entries
        ax.plot(r["t_s"], r["pkj_mean_weight"], color=colors[i], linewidth=0.35, alpha=0.14)
        ax.plot(r["t_s"], r["network_mean"], color="black", linewidth=1.8)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"coupling {r['label']}  ($g_{{gap}}$ = {r['gap_g']:g})   "
                     f"between-cell SD = {r['between_sd_final']:.4f}", fontsize=10)
        if i % 2 == 0:
            ax.set_ylabel("mean PF$\\to$PKJ weight\nper Purkinje cell (n=320)")
        if i // 2 == 1:
            ax.set_xlabel("time (s)")

    ax = fig.add_subplot(gs[2, :])
    for i, r in enumerate(results):
        ax.plot(r["t_s"], r["between_sd"], color=colors[i], linewidth=1.4,
                label=f"{r['label']} ($g_{{gap}}$ = {r['gap_g']:g}), "
                      f"CF sync {r['sync']:.3f} $\\to$ SD {r['between_sd_final']:.4f}")
    wsd = np.mean([r["within_sd_final"] for r in results])
    ax.axhline(wsd, color="black", linestyle="--", linewidth=1.2,
               label=f"spread WITHIN a cell ({wsd:.3f}) -- coupling does not touch this")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SD of the 320\nper-cell mean weights")
    ax.set_title("How tightly the loop holds the Purkinje cells together (log scale)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"IO-IO electrical coupling vs. the spread of per-cell PF$\\to$PKJ weight "
                 f"({duration_s:.0f} s, one seed)", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration-s", type=float, default=350.0,
                         help="simulated seconds per condition. The weights need a few hundred "
                              "seconds to leave the w_init transient (default: 350)")
    parser.add_argument("--seed", type=int, default=0, help="one seed, all four conditions (default: 0)")
    parser.add_argument("--record-every-ms", type=float, default=50.0,
                         help="weight sampling cadence; 320 cells x the run length (default: 50)")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--jobs", type=int, default=4, help="conditions to run in parallel (default: 4)")
    args = parser.parse_args()

    jobs = [(label, g, args.duration_s, args.seed, args.record_every_ms) for label, g in CONDITIONS]
    ref = SimConfig()
    print(f"[gap-weight] {len(jobs)} conditions x {args.duration_s:.0f} s, seed {args.seed}; "
          f"CF->PKJ reversal {ref.cf_pkj_reversal_mv:+.0f} mV "
          f"({'excitatory' if ref.cf_pkj_reversal_mv > ref.pkj.v_th_mv else 'inhibitory pause'})")
    with mp.Pool(min(args.jobs, len(jobs))) as pool:
        results = pool.map(one_run, jobs)

    print(f"\n{'coupling':10s} {'g_gap':>7s} {'CF sync':>8s} {'between-cell SD':>16s} "
          f"{'within-cell SD':>15s} {'IO Hz':>7s} {'PKJ Hz':>7s} {'DCN Hz':>7s} {'drift':>9s}")
    for r in results:
        print(f"{r['label']:10s} {r['gap_g']:7g} {r['sync']:8.3f} {r['between_sd_final']:16.4f} "
              f"{r['within_sd_final']:15.4f} {r['io_hz']:7.3f} {r['pkj_hz']:7.2f} {r['dcn_hz']:7.2f} "
              f"{r['drift']:+9.5f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gap_weight_sweep.png"
    plot(results, path, args.duration_s)
    print(f"\n[gap-weight] plot written to {path}")


if __name__ == "__main__":
    main()
