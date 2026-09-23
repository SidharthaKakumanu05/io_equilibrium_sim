"""Figures for the individual-synapse drift diagnosis.

Reads the .npz bundles written by experiments/run_drift_diagnostics.py and the
JSON written by experiments/run_rule_montecarlo.py. Read-only.
"""
import argparse, glob, json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.analyze_drift import analyze


def fig_decomposition(runs, out):
    """Where the weight spread lives, and how each level grows with time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for r in runs:
        t = r["_series"]["t"] / 1000.0
        for ax, key, name in zip(axes, ("within", "btw_pkj", "btw_io"),
                                 ("within one Purkinje cell\n(500 synapses, one CF)",
                                  "between Purkinje cells of one CF\n(8 cells, one CF)",
                                  "between CF territories\n(40 olivary cells)")):
            ax.plot(t, np.sqrt(r["_series"][key]), label=r["label"], lw=1.4)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("time (s)"); ax.set_ylabel("SD of PF->PKJ weight")
    # free-diffusion reference on the within panel
    r0 = runs[0]
    t = r0["_series"]["t"] / 1000.0
    d = 0.010  # delta_plus + delta_minus, shipped
    lam = 20.0 * 0.1
    axes[0].plot(t, d * np.sqrt(lam * t), "k--", lw=1.0, label="free diffusion  (d+ + d-)*sqrt(lam*t)")
    axes[0].axhline(1 / np.sqrt(12), color="0.5", ls=":", lw=1.0, label="uniform on [0,1]")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes[0].set_ylim(0, 0.33)
    fig.suptitle("Weight spread decomposed by level of the circuit "
                 "(law of total variance: synapse < Purkinje cell < CF territory)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_io(runs, out):
    """Per-IO rate regulation, coupling current, synchrony, AHP."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    for r in runs:
        rb = r["_series"]["rate_bins"]
        half = len(rb) // 2
        ax.plot(np.sort(rb[half:].mean(axis=0)), "o-", ms=3, lw=1, label=r["label"])
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("olivary cell (sorted)"); ax.set_ylabel("CF rate, 2nd half (Hz)")
    ax.set_title("per-cell CF rate around the population mean"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    # Direct view of what the coupling actually does: the CF raster. Under coupling every cell
    # emits the same number of events, on the same cycles, staggered by tens of ms.
    for r, marker, colour in zip(runs[:2], ("|", "|"), ("C0", "C1")):
        d = np.load(r["path"], allow_pickle=True)
        ev_io, ev_t = d["ev_io"], d["ev_t"] / 1000.0
        m = (ev_t > 300) & (ev_t < 312)
        off = 0 if colour == "C0" else 45
        ax.plot(ev_t[m], ev_io[m] + off, marker, color=colour, ms=5,
                label=f"{r['label']} (counts {np.bincount(ev_io, minlength=40).min()}"
                      f"-{np.bincount(ev_io, minlength=40).max()})")
    ax.set_xlabel("time (s)"); ax.set_ylabel("olivary cell  (two runs, offset)")
    ax.set_title("CF raster, 12 s"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for r in runs:
        d = np.load(r["path"], allow_pickle=True)
        ts = int(d["trace_steps"])
        v = d["trace_v"][ts:]
        c = np.corrcoef(v.T)
        n = c.shape[0]
        dist = np.minimum(np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]),
                          n - np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]))
        xs = np.arange(1, n // 2 + 1)
        ys = [c[dist == k].mean() for k in xs]
        ax.plot(xs, ys, "o-", ms=3, lw=1, label=r["label"])
    ax.set_xlabel("distance along the coupling ring"); ax.set_ylabel("Vm correlation")
    ax.set_title("subthreshold synchrony vs distance"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for r in runs:
        d = np.load(r["path"], allow_pickle=True)
        cp = d["ev_capeak"]; cp = cp[np.isfinite(cp)]
        ax.hist(cp, bins=60, histtype="step", lw=1.3, label=f"{r['label']} (CV {cp.std()/cp.mean():.3f})")
    ax.set_xlabel("peak [Ca]i after a CF event (uM)"); ax.set_ylabel("events")
    ax.set_title("spike-to-spike variability of the Ca2+ load\n(the AHP's only source of variability)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_fix(runs, mc_json, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    if mc_json and os.path.exists(mc_json):
        mc = json.load(open(mc_json))
        for name, r in mc.items():
            ax.plot(r["t_s"], r["sd"], lw=1.4, label=name)
        ax.axhline(1 / np.sqrt(12), color="0.5", ls=":", lw=1.0, label="uniform on [0,1]")
        ax.set_xscale("log")
    ax.set_xlabel("time (s)"); ax.set_ylabel("within-cell weight SD")
    ax.set_title("the rule alone, CF held at equilibrium\n(no olive, no coupling, no loop)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for r in runs:
        t = r["_series"]["t"] / 1000.0
        ax.plot(t, np.sqrt(r["_series"]["within"]), lw=1.4, label=r["label"])
    ax.axhline(1 / np.sqrt(12), color="0.5", ls=":", lw=1.0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("within-cell weight SD")
    ax.set_title("in the full network"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    labels = [r["label"] for r in runs]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [r["sync_index"] for r in runs], 0.4, label="CF synchrony index")
    ax.bar(x + 0.2, [r["vm_corr"] for r in runs], 0.4, label="mean Vm correlation")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("olivary synchrony -- must not rise"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--outdir", default="results/drift")
    ap.add_argument("--mc-json", default=None)
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    paths = sorted(p for pat in a.paths for p in glob.glob(pat))
    runs = [analyze(p) for p in paths]
    os.makedirs(a.outdir, exist_ok=True)
    if a.which in ("all", "decomp"):
        fig_decomposition(runs, os.path.join(a.outdir, "decomposition.png"))
    if a.which in ("all", "io"):
        fig_io(runs, os.path.join(a.outdir, "io_mechanisms.png"))
    if a.which in ("all", "fix"):
        fig_fix(runs, a.mc_json, os.path.join(a.outdir, "fix.png"))
    print("wrote figures to", a.outdir)


if __name__ == "__main__":
    main()
