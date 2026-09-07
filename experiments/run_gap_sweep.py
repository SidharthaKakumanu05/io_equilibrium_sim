#!/usr/bin/env python3
"""What the IO-IO gap junctions actually do to the closed loop.

Sweeps the conductance of a single gap junction from zero upward and measures,
in the assembled network:

  * subthreshold Vm correlation between IO cells -- the direct read-out of
    electrical coupling. Ca2+ spikes are masked out, so this measures the
    coupling of the subthreshold oscillation rather than spike co-occurrence,
    and it is far more sensitive than a spike measure at ~1 Hz firing.
  * CF synchrony -- pairwise correlation of the binned climbing-fiber trains,
    which is the thing coupling is supposed to produce downstream.
  * CF rate and equilibrium weight -- because coupling is not free. A gap
    junction is a conductance, so cells held at different phases load each
    other, and past a point that load suppresses the Ca2+ spike, drops the CF
    rate below the plasticity's balance point, and lets the weights climb.

The default `gap_g` is chosen from this curve: strong enough to be in the
physiological regime (total coupling conductance ~ g_leak) and to produce
measurable correlation, weak enough to leave the loop's equilibrium intact.

Writes results/gap_coupling_sweep.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import summarize
from sim.io_coupling import synchrony_index


def subthreshold_vm_correlation(log, v_max_mv=-50.0, min_samples=500):
    """Mean pairwise Pearson correlation of IO membrane potential, over samples
    where BOTH cells of a pair are below v_max_mv. Masking the Ca2+ spikes out
    keeps the measure about the subthreshold oscillation the junctions couple,
    rather than about the spikes those junctions are only indirectly shaping."""
    v = log.trace_io_v
    below = v < v_max_mv
    out = []
    for i in range(v.shape[1]):
        for j in range(i + 1, v.shape[1]):
            both = below[:, i] & below[:, j]
            if int(both.sum()) > min_samples:
                out.append(np.corrcoef(v[both, i], v[both, j])[0, 1])
    return float(np.mean(out)) if out else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap-g", type=float, nargs="+",
                         default=[0.0, 0.008, 0.02, 0.03, 0.05, 0.08],
                         help="single-junction conductances to sweep, mS/cm^2")
    parser.add_argument("--duration-s", type=float, default=40.0)
    parser.add_argument("--trace-window-s", type=float, default=8.0,
                         help="Vm window the subthreshold correlation is measured over")
    parser.add_argument("--topology", type=str, default=None,
                         choices=["all_to_all", "ring", "nearest_k"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    from sim.simulate import Simulation
    base = SimConfig()
    topology = args.topology or base.gap_topology
    rows = []
    print(f"{'g_gap':>7} {'total':>7} {'x leak':>7} {'IO Hz':>7} {'Vm r':>7} {'CF sync':>8} "
          f"{'weight':>7} {'drift':>9}")
    for g in args.gap_g:
        cfg = SimConfig(n_io=base.n_io, duration_s=args.duration_s, seed=args.seed,
                         gap_g=g, gap_topology=topology, trace_window_s=args.trace_window_s)
        log = Simulation(cfg).run()
        s = summarize(log)
        half = log.duration_ms / 2.0
        row = {
            "g": g,
            "total": float(log.gap_matrix.sum(axis=1).mean()),
            "vm_r": subthreshold_vm_correlation(log),
            "sync": synchrony_index(log.io_spikes, half, log.duration_ms, bin_ms=50.0),
            "io": s["io_rate_hz"], "w": s["mean_weight"], "drift": s["weight_drift_slope"],
        }
        rows.append(row)
        print(f"{g:7.3f} {row['total']:7.3f} {row['total'] / cfg.io_channels.g_leak:7.1f} "
              f"{row['io']:7.2f} {row['vm_r']:+7.3f} {row['sync']:+8.3f} {row['w']:7.3f} {row['drift']:+9.5f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [r["g"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(g, [r["vm_r"] for r in rows], "o-", color="tab:purple", label="subthreshold $V_m$ correlation")
    ax1.plot(g, [r["sync"] for r in rows], "s-", color="tab:blue", label="CF synchrony (50 ms bins)")
    ax1.axvline(base.gap_g, color="gray", ls="--", lw=1, label=f"default $g_{{gap}}$ = {base.gap_g}")
    ax1.set_xlabel("$g_{gap}$ per junction (mS/cm$^2$)")
    ax1.set_ylabel("mean pairwise correlation")
    ax1.set_title("coupling does what it should ...")
    ax1.legend(fontsize=8)

    ax2.plot(g, [r["io"] for r in rows], "o-", color="tab:red", label="CF rate (Hz)")
    ax2.axhline(1.0, color="tab:red", ls=":", lw=1)
    ax2.set_xlabel("$g_{gap}$ per junction (mS/cm$^2$)")
    ax2.set_ylabel("CF rate (Hz)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2b = ax2.twinx()
    ax2b.plot(g, [r["w"] for r in rows], "s-", color="tab:orange", label="equilibrium weight")
    ax2b.axhline(0.5, color="tab:orange", ls=":", lw=1)
    ax2b.set_ylabel("mean PF$\\to$PKJ weight", color="tab:orange")
    ax2b.tick_params(axis="y", labelcolor="tab:orange")
    ax2.axvline(base.gap_g, color="gray", ls="--", lw=1)
    ax2.set_title("... but it is a conductance, and it loads the cell")

    fig.suptitle(f"IO-IO gap-junction coupling ({topology}, {base.n_io} cells), "
                 f"{args.duration_s:.0f} s closed-loop runs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "gap_coupling_sweep.png", dpi=150)
    plt.close(fig)
    print(f"[gap_sweep] plot written to {out_dir}/gap_coupling_sweep.png")


if __name__ == "__main__":
    main()
