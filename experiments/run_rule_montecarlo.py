"""The plasticity rule ALONE, driven by a climbing fiber held at the loop's
equilibrium by construction.

This is the control the network cannot provide. It instantiates the shipped
sim.plasticity.Plasticity -- the same object the network uses, not a
re-implementation -- gives it ONE Purkinje cell whose climbing fiber is a
perfectly regular train at exactly the equilibrium rate, and lets its synapses
run. Every mechanism the drift diagnostics are about is removed: there is no
olive, so no per-cell rate dispersion, no gap junctions, no synchrony, and no
loop to be closed or open. Whatever spread appears here is the rule's own.

The CF train is regular rather than Poisson on purpose. At 1 Hz with a 100 ms
window a regular train gives P(LTD per PF spike) = rate * window = 0.100
exactly, which is the balance point delta_plus / (delta_plus + delta_minus) --
so the drift of the mean is zero by construction and only the spread is left.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.plasticity import Plasticity


def run(duration_s, cf_hz=1.0, pf_hz=20.0, n_syn=2000, dt_ms=1.0,
        ltd_window_ms=100.0, delta_plus=0.001, delta_minus=0.009,
        null_window_ms=0.0, weight_dependence=0.0, w_init=0.5, seed=0,
        sample_every_s=10.0):
    rng = np.random.default_rng(seed)
    pl = Plasticity(1, n_syn, dt_ms, ltd_window_ms, delta_plus, delta_minus,
                    null_window_ms=null_window_ms, weight_dependence=weight_dependence)
    w = np.full((1, n_syn), w_init)
    n_steps = int(round(duration_s * 1000.0 / dt_ms))
    cf_period_steps = 1000.0 / cf_hz / dt_ms
    p_pf = pf_hz * dt_ms / 1000.0
    sample_every = int(round(sample_every_s * 1000.0 / dt_ms))
    ts, sd, mean, at_lo, at_hi = [], [], [], [], []
    next_cf = cf_period_steps
    for i in range(n_steps):
        cf = False
        if i >= next_cf:
            cf = True
            next_cf += cf_period_steps
        pf = rng.random((1, n_syn)) < p_pf
        pl.step(pf, np.array([cf]), w)
        if i % sample_every == 0:
            ts.append(i * dt_ms / 1000.0); sd.append(w.std()); mean.append(w.mean())
            at_lo.append(float((w <= 1e-12).mean())); at_hi.append(float((w >= 1 - 1e-12).mean()))
    return dict(t_s=np.array(ts), sd=np.array(sd), mean=np.array(mean),
                at_lo=np.array(at_lo), at_hi=np.array(at_hi), w_final=w[0].copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=1800.0)
    ap.add_argument("--n-syn", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    conditions = [
        ("additive  d+=0.001 d-=0.009 (shipped)", dict()),
        ("additive  x0.2", dict(delta_plus=0.0002, delta_minus=0.0018)),
        ("additive  + 200 ms null window (H3)", dict(null_window_ms=200.0)),
        ("soft-bound  wd=1.0, same deltas", dict(weight_dependence=1.0)),
        ("soft-bound  wd=0.5, same deltas", dict(weight_dependence=0.5)),
    ]
    out = {}
    hdr_t = [100, 350, 600, 1800, 3600, 18000]
    ts_used = [t for t in hdr_t if t <= a.duration_s]
    print(f"{'condition':<40}" + "".join(f"{'SD@'+str(t)+'s':>12}" for t in ts_used))
    print("-" * (40 + 12 * len(ts_used)))
    for name, kw in conditions:
        r = run(a.duration_s, n_syn=a.n_syn, **kw)
        vals = []
        for t in ts_used:
            k = int(np.argmin(np.abs(r["t_s"] - t)))
            vals.append(r["sd"][k])
        print(f"{name:<40}" + "".join(f"{v:>12.4f}" for v in vals))
        out[name] = {k: v.tolist() for k, v in r.items()}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
