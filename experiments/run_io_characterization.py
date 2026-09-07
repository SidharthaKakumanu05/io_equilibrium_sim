#!/usr/bin/env python3
"""Characterize the conductance-based IO neuron (sim/io_channels.py), alone.

Four protocols, none of which involve the rest of the circuit. The point is to
establish that the ionic model behaves like an olivary cell BEFORE it is trusted
to drive the closed loop -- H1 is a claim about a self-paced oscillator held at a
rate by inhibition, and it means nothing if the oscillator is an artifact:

  1. traces      -- V and [Ca]i, free-running vs. under the DCN inhibition the
                    closed loop actually delivers (as spikes, not a held conductance).
  2. ablation    -- block each channel in turn (the modeller's version of a
                    pharmacology experiment) and show what each one contributes.
  3. rebound     -- hyperpolarizing step then release. The T-type signature: the
                    step de-inactivates I_CaL, and release triggers a rebound
                    low-threshold Ca2+ spike that would never fire from rest.
  4. transfer    -- CF rate vs. DCN GABA_A conductance, the loop's feedback limb,
                    measured both with the conductance held constant and with it
                    driven by DCN spikes. These are different curves and the
                    difference is the result: held constant the response is a
                    cliff (3 Hz at g = 0, silent by g = 0.10) that no loop could
                    regulate against; delivered as discrete DCN events the same
                    means give a smooth monotone slope. Monotonicity is what H1's
                    equilibrium argument rests on, and tests/test_io_channels.py
                    asserts it.

This is the slowest script here for its size: it drives ONE cell (pooled over
N_POOL identical copies) for a long time at each of ~30 conditions, and the IO's
0.1 ms substepping is where nearly all the project's compute goes. Budget
several minutes, and use --transfer-duration-s to trade precision for time.

Writes results/io_traces.png, io_channel_ablation.png, io_rebound.png and
io_transfer_curve.png.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.analysis import plot_io_membrane, plot_io_transfer_curve
from sim.io_channels import IOConductanceNeuron, IOPopulation


class SpikeDrivenGaba:
    """DCN inhibition delivered the way the network delivers it: `n_dcn` Poisson
    presynaptic trains at `rate_hz`, each spike stepping a single-exponential
    conductance. `g()` returns this millisecond's conductance.

    Protocols that hold the conductance CONSTANT use a plain float instead. Both
    appear in this script deliberately -- the difference between them is a
    result, not an implementation detail. See the transfer-curve protocol."""

    def __init__(self, cfg, rate_hz, rng, n_cells=1):
        self.p = rate_hz * cfg.dt_ms / 1000.0
        self.n_cells = n_cells
        self.n_dcn = cfg.n_dcn_per_io
        # Same convergence normalization sim/simulate.py applies, so this protocol drives the
        # cell with the conductance the assembled network actually delivers.
        self.gain = cfg.dcn_io_gaba_gain * cfg.dcn_io_fitted_at_n_dcn / cfg.n_dcn_per_io
        self.decay = float(np.exp(-cfg.dt_ms / cfg.tau_dcn_io_ms))
        self.rng = rng
        self._g = np.zeros(n_cells)

    def g(self):
        """This millisecond's conductance, one value per cell. Each cell gets its
        own independent DCN draw, exactly as each olivary cell does in the network
        (no two of them read the same block of the nucleus)."""
        arrived = (self.rng.random((self.n_cells, self.n_dcn)) < self.p).sum(axis=1)
        self._g += self.gain * arrived
        value = self._g.copy()
        self._g *= self.decay
        return value


def drive(params, duration_ms, g_gaba=0.0, i_app_schedule=None, seed=0, settle_ms=2000.0):
    """Run one IO cell in isolation. `g_gaba` is either a constant conductance
    or a SpikeDrivenGaba. `i_app_schedule` is an optional f(t_ms) -> uA/cm^2
    applied-current waveform, used by the rebound protocol.
    Returns (t_ms, V, [Ca], cf_times_ms, g_trace), sampled at 1 ms."""
    def g_now():
        return float(g_gaba.g()[0]) if isinstance(g_gaba, SpikeDrivenGaba) else g_gaba

    neuron = IOConductanceNeuron(params, 1.0, rng=np.random.default_rng(seed))
    for _ in range(int(settle_ms)):
        neuron.step(1.0, g_now())                                 # discard the startup transient
    n = int(duration_ms)
    t, v, ca, cf = np.arange(n, dtype=float), np.empty(n), np.empty(n), []
    g_trace = np.empty(n)
    base_i_app = params.i_app
    for i in range(n):
        if i_app_schedule is not None:
            neuron.p = dataclasses.replace(params, i_app=base_i_app + i_app_schedule(float(i)))
        g_trace[i] = g_now()
        if neuron.step(1.0, g_trace[i]):
            cf.append(float(i))
        v[i], ca[i] = neuron.V, neuron.Ca
    neuron.p = params
    return t, v, ca, np.asarray(cf), g_trace


# Both rate measurements pool a POPULATION of identical, uncoupled cells rather than running one
# cell for longer. The ionic integration is vectorized across cells (sim/io_channels.py), so N cells
# cost about the same per step as one -- this buys N times the cell-seconds per point in the same
# wall time, from N independent noise streams. Uncoupled because these protocols characterize the
# single cell; sim/io_coupling.py is where coupling is measured.
N_POOL = 12


def cf_rate(params, g_gaba, duration_ms, seed=0, n=N_POOL, settle_ms=3000):
    """CF rate under a conductance held CONSTANT."""
    pop = IOPopulation(params, n, 1.0, rng=np.random.default_rng(seed))
    for _ in range(settle_ms):
        pop.step(1.0, g_gaba)
    fired = sum(int(pop.step(1.0, g_gaba).sum()) for _ in range(int(duration_ms)))
    return fired / n / (duration_ms / 1000.0)


def cf_rate_spike_driven(cfg, params, dcn_rate_hz, duration_ms, seed=0, n=N_POOL, settle_ms=3000):
    """CF rate under inhibition delivered as DCN spikes. Returns (rate_hz, mean
    conductance) so it can be plotted against the same x-axis as the constant
    curve -- which is the comparison the whole protocol exists to make."""
    syn = SpikeDrivenGaba(cfg, dcn_rate_hz, np.random.default_rng(seed + 99), n_cells=n)
    pop = IOPopulation(params, n, 1.0, rng=np.random.default_rng(seed))
    for _ in range(settle_ms):
        pop.step(1.0, syn.g())
    fired, g_sum, steps = 0, 0.0, int(duration_ms)
    for _ in range(steps):
        g = syn.g()
        g_sum += float(g.mean())
        fired += int(pop.step(1.0, g).sum())
    return fired / n / (duration_ms / 1000.0), g_sum / steps


# DCN's rate at the loop's operating point. There is no closed form for it -- PKJ and DCN are spiking
# cells -- so it is simply the target experiments/run_calibration.py fits pkj_dcn_gain against, and it
# is what the assembled network is observed to deliver. Every conductance in this script is derived
# from it, so changing it changes where the protocols probe, not what the cell does.
DCN_RATE_AT_OPERATING_POINT_HZ = 15.0

# IOChannelParams.noise_sigma is 0 by default, so the cell's rhythm is generated purely by its own
# ionic currents. The ablation protocol still wants a noisy arm for contrast -- this is the level the
# model was originally fitted at, and is what the "with noise" column below is measured with.
NOISE_FOR_CONTRAST = 2.0


def operating_point_conductance(cfg, dcn_rate_hz=DCN_RATE_AT_OPERATING_POINT_HZ):
    """The mean GABA conductance the IO sees at closed-loop equilibrium.

    DCN -> IO is a single-exponential conductance, so driving it at rate R from
    n_dcn_per_io cells settles at n*R*tau*gain. This is only the *mean*: in the
    network the conductance fluctuates widely around it, and that matters more
    than the mean does. Held constant at this value the cell is silent; driven by
    DCN spikes with this same average it fires at ~1 Hz. Both curves are in the
    transfer-curve figure, and the operating point is marked on the second."""
    g = (cfg.n_dcn_per_io * (dcn_rate_hz / 1000.0) * cfg.tau_dcn_io_ms
         * cfg.dcn_io_gaba_gain * cfg.dcn_io_fitted_at_n_dcn / cfg.n_dcn_per_io)
    return g, dcn_rate_hz


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=str, default="results",
                         help="directory the four PNGs are written to (default: results)")
    parser.add_argument("--seed", type=int, default=0,
                         help="seeds the surrogate DCN trains and any membrane noise (default: 0)")
    parser.add_argument("--transfer-duration-s", type=float, default=8.0,
                         help=f"simulated seconds per point on the two rate/conductance curves. Each "
                              f"point pools {N_POOL} identical uncoupled cells, so it is {N_POOL}x "
                              f"that many cell-seconds of data. 26 points, and this is where the "
                              f"script's runtime goes (default: 8)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig()
    p = cfg.io_channels
    g_op, dcn_rate = operating_point_conductance(cfg)
    print(f"[io-char] closed-loop operating point: DCN ~{dcn_rate:.1f} Hz -> g_GABA = {g_op:.4f} mS/cm^2")

    # --- 1. traces: free-running vs. inhibited to the operating point ---
    # Inhibition is delivered as DCN SPIKES, not as a held conductance. Held at this same mean the
    # cell is silent (see the transfer curve below); it fires at ~1 Hz here because the spike-driven
    # conductance fluctuates and it escapes in the troughs. That is the loop's real operating condition.
    t, v, ca, cf, _ = drive(p, 6000.0, g_gaba=0.0, seed=args.seed)
    syn = SpikeDrivenGaba(cfg, dcn_rate, np.random.default_rng(args.seed + 99))
    t2, v2, ca2, cf2, g2 = drive(p, 6000.0, g_gaba=syn, seed=args.seed)
    print(f"[io-char] free-running: {len(cf)} CF events in 6 s; under spike-driven DCN inhibition "
          f"at {dcn_rate:.0f} Hz: {len(cf2)} (conductance {g2.mean():.2f} +- {g2.std():.2f} mS/cm^2)")
    plot_io_membrane(
        [("no inhibition", t, v, ca),
         (f"DCN inhibition, spike-driven ({dcn_rate:.0f} Hz, mean g={g2.mean():.2f})", t2, v2, ca2)],
        "IO conductance model: Ca2+ spikes and the [Ca]i transients that gate I_KCa",
        out_dir / "io_traces.png", v_spike_mv=p.v_spike_mv,
    )

    # --- 2. channel ablation ---
    # Traces are taken noiseless (which is now the default) so each current's contribution is
    # unambiguous. The rate with membrane noise ADDED is printed alongside, because noise lets a
    # blocked cell still fire occasionally -- an I_CaL block abolishes firing outright without it,
    # but only reduces it sharply with it. That contrast is the point, so the noisy arm asks for
    # noise explicitly rather than relying on the default.
    quiet_ctl = dataclasses.replace(p, noise_sigma=0.0)
    noisy_ctl = dataclasses.replace(p, noise_sigma=NOISE_FOR_CONTRAST)
    conditions = [
        ("control", {}),
        ("I_CaL block (T-type)", {"g_cal": 0.0}),
        ("I_KCa block", {"g_kca": 0.0}),
        ("I_h block", {"g_h": 0.0}),
    ]
    traces = []
    for label, override in conditions:
        tt, vv, cc, ff, _ = drive(dataclasses.replace(quiet_ctl, **override), 4000.0, g_gaba=0.0, seed=args.seed)
        ff_noisy = drive(dataclasses.replace(noisy_ctl, **override), 8000.0, g_gaba=0.0, seed=args.seed)[3]
        traces.append((f"{label}  ({len(ff) / 4.0:.2f} Hz)", tt, vv, cc))
        print(f"[io-char] {label:22s}: {len(ff) / 4.0:.2f} Hz noiseless, "
              f"{len(ff_noisy) / 8.0:.2f} Hz with noise_sigma={NOISE_FOR_CONTRAST}; "
              f"V range {vv.min():.1f}..{vv.max():.1f} mV, peak [Ca] {cc.max():.2f} uM")
    print("[io-char] note: with I_KCa blocked nothing repolarizes the Ca2+ spike -- this reduction has no "
          "delayed-rectifier K+, so the cell latches into depolarization block rather than firing broad spikes.")
    plot_io_membrane(traces, "IO channel ablation: what each current contributes (noiseless)",
                      out_dir / "io_channel_ablation.png", v_spike_mv=p.v_spike_mv)

    # --- 3. rebound burst: the T-type de-inactivation signature ---
    # Hold the cell hyperpolarized for 800 ms, which de-inactivates I_CaL (raises its l gate), then release.
    # A rebound Ca2+ spike should follow release almost immediately -- much sooner than the free-running
    # period -- and must disappear when I_CaL is blocked. That contrast is the actual test.
    step_i, step_start, step_end = -4.0, 500.0, 1300.0
    schedule = lambda ms: step_i if step_start <= ms < step_end else 0.0
    quiet = dataclasses.replace(p, noise_sigma=0.0)               # noiseless, so the rebound is unambiguous
    rebound_traces, rebound_cf = [], None
    for label, params in [("control", quiet), ("I_CaL blocked", dataclasses.replace(quiet, g_cal=0.0))]:
        tr, vr, car, cfr, _ = drive(params, 3000.0, i_app_schedule=schedule, seed=args.seed)
        after = cfr[(cfr >= step_end) & (cfr < step_end + 400)]
        latency = f"{after[0] - step_end:.0f} ms" if len(after) else "none"
        print(f"[io-char] rebound ({label:13s}): V held at {vr[int(step_start) + 400]:.1f} mV during the step; "
              f"{len(after)} CF event(s) within 400 ms of release, first at {latency}")
        rebound_traces.append((f"{label} (rebound: {latency})", tr, vr, car))
        if label == "control":
            rebound_cf = cfr
    plot_io_membrane(rebound_traces,
                      "IO rebound burst: hyperpolarization de-inactivates I_CaL (T-type)",
                      out_dir / "io_rebound.png", v_spike_mv=p.v_spike_mv, cf_times_ms=rebound_cf)

    # --- 4. CF rate vs. DCN inhibition, held constant AND delivered as spikes ---
    # These are different curves and the difference is the result. Held constant, the olive's
    # response to inhibition is nearly a cliff, and the loop's operating conductance sits well
    # inside the silent region. Delivered as DCN spikes the same mean conductance fluctuates,
    # the cell fires in the troughs, and the cliff becomes a gentle slope that a feedback loop
    # can actually sit on. config.dcn_io_gaba_gain is fitted against the spike-driven curve.
    g_values = np.round(np.linspace(0.0, 1.4, 15), 4)
    const_rates = [cf_rate(p, float(g), args.transfer_duration_s * 1000.0, seed=args.seed) for g in g_values]

    dcn_rates = np.linspace(0.0, 30.0, 11)
    spike_pairs = [cf_rate_spike_driven(cfg, p, float(r), args.transfer_duration_s * 1000.0, seed=args.seed)
                    for r in dcn_rates]
    spike_rates = [r for r, _ in spike_pairs]
    spike_g = [g for _, g in spike_pairs]

    for g, r in zip(g_values, const_rates):
        print(f"[io-char] constant     g_GABA={g:.3f} -> CF {r:.2f} Hz")
    for d, (r, g) in zip(dcn_rates, spike_pairs):
        print(f"[io-char] spike-driven DCN={d:4.1f} Hz (mean g={g:.3f}) -> CF {r:.2f} Hz")

    plot_io_transfer_curve(
        [("held constant", g_values, const_rates),
         (f"spike-driven ({cfg.n_dcn_per_io} DCN, tau={cfg.tau_dcn_io_ms:.0f} ms)", spike_g, spike_rates)],
        "IO CF rate vs. DCN inhibition -- the loop's feedback limb.\n"
        "Held constant the response is all-or-nothing; delivered as DCN spikes it is a usable slope.",
        out_dir / "io_transfer_curve.png", operating_point=g_op,
    )
    print(f"[io-char] plots written to {out_dir}")


if __name__ == "__main__":
    main()
