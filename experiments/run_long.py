#!/usr/bin/env python3
"""One hour-scale run of the microzone, streamed to disk as a RunBundle.

This is experiments/run_baseline.py's job at a length run_baseline cannot reach.
Everything the short scripts hold in memory -- every spike of every cell -- is
written out incrementally instead (see sim/recording.py), so run length is
bounded by disk rather than by RAM, and the figures are made afterwards by
experiments/make_long_figures.py off the saved bundle rather than in the same
process. That split matters when the run takes half a day: a plotting bug at the
end no longer costs you the simulation.

Budget before you start one. At the shipped scale one simulated second costs
about 2.5 s of wall-clock on one core, so 5 simulated hours is ~12.5 wall-clock
hours, and the PKJ spike file alone is ~2 GB. Cost scales with n_pf_per_pkj
(the per-step Poisson draw is the second-largest term after the olive's ionic
sub-stepping), so the 2000-fiber arm of a pool-size sweep costs about 4x the
125-fiber arm.

Presets (--preset) set the condition; --duration-s sets the length.

    baseline    the shipped configuration: excitatory CF, gap junctions on, loop closed
    cf_inh      CF -> PKJ reversal at -75 mV, the original post-complex-spike pause
    gap_off     gap_g = 0, the olive uncoupled
    open_loop   ablate_dcn_io, the nucleo-olivary limb cut -- H1's control
    pf<N>       baseline with n_pf_per_pkj = N (e.g. --preset pf250)
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SimConfig
from sim.long_run import run_long
from sim.simulate import Simulation


def build_config(preset, duration_s, seed, n_tracked, n_pf_recorded):
    cfg = SimConfig(duration_s=duration_s, seed=seed, n_tracked_synapses=n_tracked)
    cfg.n_pf_recorded = n_pf_recorded
    if preset == "baseline":
        pass
    elif preset == "cf_inh":
        cfg.cf_pkj_reversal_mv = -75.0
    elif preset == "gap_off":
        cfg.gap_g = 0.0
    elif preset == "open_loop":
        cfg.ablate_dcn_io = True
    elif re.fullmatch(r"pf\d+", preset or ""):
        cfg.n_pf_per_pkj = int(preset[2:])
    else:
        raise SystemExit(f"unknown preset {preset!r}; see --help")
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="baseline", help="condition to run (see docstring)")
    p.add_argument("--duration-s", type=float, default=18000.0,
                   help="simulated seconds after burn-in; ~2.5x that in wall-clock (default: 18000 = 5 h)")
    p.add_argument("--seed", type=int, default=0, help="PF Poisson draws and membrane noise (default: 0)")
    p.add_argument("--out-dir", required=True, help="directory the bundle is written to")
    p.add_argument("--n-tracked-synapses", type=int, default=10000,
                   help="individual PF->PKJ synapses logged every --record-every-s. 10,000 of the "
                        "160,000 at a 1 s cadence over 5 h is ~720 MB (default: 10000)")
    p.add_argument("--n-pf-recorded", type=int, default=200,
                   help="PF units whose spike trains are kept, for the raster (default: 200)")
    p.add_argument("--record-every-s", type=float, default=1.0,
                   help="cadence of the weight traces, simulated seconds (default: 1.0)")
    p.add_argument("--trace-len-s", type=float, default=10.0,
                   help="length of each membrane-potential window (default: 10)")
    p.add_argument("--trace-windows-s", type=float, nargs="*", default=None,
                   help="start times of the membrane-potential windows "
                        "(default: start, middle, and the end of the run)")
    p.add_argument("--burn-in-s", type=float, default=None,
                   help="settling run before plasticity and recording start (default: config, 8 s)")
    args = p.parse_args()

    cfg = build_config(args.preset, args.duration_s, args.seed, args.n_tracked_synapses,
                       args.n_pf_recorded)
    sim = Simulation(cfg)
    label = f"{args.preset}/s{args.seed}"
    print(f"[{label}] {sim.conn.describe()}", flush=True)
    print(f"[{label}] n_pf_per_pkj {cfg.n_pf_per_pkj} | gap_g {cfg.gap_g} | "
          f"CF reversal {cfg.cf_pkj_reversal_mv:+.0f} mV | "
          f"dcn->io {'ABLATED' if cfg.ablate_dcn_io else 'intact'} | "
          f"{args.duration_s:.0f} s sim", flush=True)

    run_long(sim, args.out_dir, duration_s=args.duration_s, burn_in_s=args.burn_in_s,
             record_every_s=args.record_every_s, trace_windows_s=args.trace_windows_s,
             trace_len_s=args.trace_len_s, label=label)


if __name__ == "__main__":
    main()
