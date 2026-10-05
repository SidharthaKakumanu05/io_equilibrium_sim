#!/usr/bin/env python3
"""io_equilibrium_sim -- start here.

    python3 run.py                     # run with the settings below
    python3 run.py NUM_TRIALS=120      # override any setting on the command line
    python3 run.py COUPLING=0 SEED=3   # ...as many as you like

WHAT IS BEING SIMULATED
-----------------------
A small piece of cerebellum (a "microzone") wired as one closed loop:

    PF --(+, plastic)--> PKJ --(-)--> DCN --(-)--> IO --(CF)--> PKJ
                                                    ^                |
                                          IO <-> IO gap junctions    |
                                                                     |
                                 (the CF also gates LTD at PF->PKJ) -+

    PF   parallel fibres     random (Poisson) excitatory input to Purkinje cells
    PKJ  Purkinje cells      excited by PFs, inhibit the DCN
    DCN  deep nuclear cells  inhibit the inferior olive
    IO   inferior olive      fires climbing-fibre (CF) events back onto PKJ

The question the model asks: does this loop settle at a stable operating point?

  * When a CF event arrives, every PF synapse on that Purkinje cell that fired
    in the last LTD_WINDOW_MS gets weaker (LTD). Every PF spike NOT followed by a
    CF gets slightly stronger (LTP).
  * Weaker PF synapses -> PKJ fires less -> DCN fires more -> IO is inhibited
    more -> fewer CF events -> less LTD. And the reverse. That is negative
    feedback, so the weights should settle where LTD and LTP cancel, which
    pins the CF rate at about 1 Hz with the default settings.
  * CUT_DCN_TO_IO=1 removes the DCN->IO link, the only path the feedback has.
    If the equilibrium really comes from the loop, it should disappear.

HOW TO READ THE CODE
--------------------
Read the files in this order. Each one is commented to be read top to bottom.

  1. run.py              (this file)  settings, then build -> run the trials -> report
  2. config.py           every parameter, with units and what it does
  3. sim/simulate.py     THE MAIN LOOP: builds the network and steps it in time
  4. sim/connectivity.py who connects to whom
  5. sim/poisson_input.py the PF input
  6. sim/neurons.py      Purkinje and DCN cells (integrate-and-fire)
  7. sim/io_channels.py  the olive cell (ion channels, much more detailed)
  8. sim/io_coupling.py  gap junctions between olive cells
  9. sim/plasticity.py   the LTD/LTP learning rule
 10. sim/analysis.py     the summary numbers and the figures
 11. sim/rasters.py      raster files in CbmSim's format (writing and reading)

sim/network_graph.py and sim/network_viz.py draw the wiring diagram. run.py
does not call them.
"""

# ==========================================================================
#  SETTINGS -- edit these, or override any of them as NAME=VALUE
# ==========================================================================

# --- how long: trials, as in CbmSim ---------------------------------------
# The run is NUM_TRIALS trials of TRIAL_MS each, back to back. The network is not
# reset between trials; a trial is a block of time that data is organised by
# (one progress line per trial, and trial markers in the raster files).
NUM_TRIALS            = 60      # number of trials
TRIAL_MS              = 5000.0  # length of one trial, ms (CbmSim's default trialTime)
                                # Total = 60 x 5 s = 300 s. The weights take a few
                                # hundred seconds to settle, so use >= 300 s in total
                                # before trusting the drift number.
BURN_IN_S             = 8.0     # seconds run first, with learning off and nothing
                                # recorded, so the cells start from a realistic state
SEED                  = None       # random seed for the PF input; None = different every run

# --- gap junctions between olive cells -----------------------------------
COUPLING              = 1       # 0 = off, 1 = on
COUPLING_STRENGTH     = 0.0143  # conductance of one junction, mS/cm^2
COUPLING_NEIGHBOURS   = 2       # each olive cell is coupled to this many neighbours
                                # on each side (cells sit on a ring)

# --- how many cells, and how they connect --------------------------------
# These have to be consistent with each other: the number of PKJ->DCN synapses
# counted from the PKJ side must equal the number counted from the DCN side.
# check_settings() below tells you which numbers disagree if they don't.
N_IO                  = 40      # olive cells (= number of climbing fibres)
N_DCN                 = 80      # deep nuclear cells
N_PKJ_PER_IO          = 8       # Purkinje cells each climbing fibre contacts
                                #   -> total PKJ = N_IO * N_PKJ_PER_IO = 320
N_PKJ_PER_DCN         = 12      # Purkinje cells inhibiting each DCN cell
N_DCN_PER_PKJ         = 3       # DCN cells each Purkinje cell inhibits
N_DCN_PER_IO          = 8       # DCN cells inhibiting each olive cell
N_PF_PER_PKJ          = 500     # parallel fibres onto each Purkinje cell. More
                                # fibres -> each synapse settles at a smaller weight;
                                # the loop holds (fibres x weight) roughly fixed.

# --- parallel-fibre input ------------------------------------------------
PF_RATE_HZ            = 20.0    # firing rate of each parallel fibre

# --- learning rule (PF->PKJ plasticity) ----------------------------------
LTD_WINDOW_MS         = 100.0   # a CF this soon after a PF spike weakens that synapse
NULL_WINDOW_MS        = 0.0     # optional gap after the LTD window where a CF causes
                                # no change at all. 0 = off (the standard rule).
DELTA_MINUS           = 0.009   # how much one LTD event weakens a synapse
DELTA_PLUS            = 0.001   # how much one LTP event strengthens it
W_INIT                = 0.5     # starting weight of every PF->PKJ synapse
W_MIN                 = 0.0     # weights are clipped to [W_MIN, W_MAX]
W_MAX                 = 1.0
# Predicted equilibrium: the time between CF events settles at
#     LTD_WINDOW_MS x (1 + DELTA_MINUS / DELTA_PLUS) = 100 x (1 + 9) = 1000 ms -> 1 Hz.

# --- lesions and variants ------------------------------------------------
CUT_DCN_TO_IO         = 0       # 1 = remove DCN->IO inhibition. The control experiment:
                                # with no feedback the equilibrium should break.
CLOSED_LOOP           = 1       # 0 = keep DCN->IO but wire each olive cell to the WRONG
                                # block of DCN cells (feedback misrouted, not removed)
IO_HETEROGENEITY_CV   = 0.0     # make olive cells differ from each other by this much
                                # (0.15 = 15% spread in their channel densities).
                                # Fixed per cell, not random noise over time.

# --- output: what gets saved in OUTPUT_DIR (1 = save, 0 = skip) ---------
OUTPUT_DIR            = "results/run"

# figures
PLOT_RASTERS          = 1       # rasters.png   spikes of all four cell types, last 10 s
PLOT_WEIGHTS          = 1       # weights.png   PF->PKJ weights over the whole run
PLOT_VOLTAGES         = 1       # voltages.png  membrane voltages, last 3 s
PLOT_IO_STATE         = 1       # io_state.png  olive voltage and calcium, last 3 s

# spike rasters, in CbmSim's binary format (read back with sim.rasters.read_cbm_raster).
# File names are <OUTPUT_DIR's folder name> + the extension, e.g. results/run/run.pcr.
SAVE_RASTER_IO        = 1       # .ior  olive spikes = CF events (~1 MB)
SAVE_RASTER_PKJ       = 1       # .pcr  Purkinje spikes (~13 MB)
SAVE_RASTER_DCN       = 1       # .ncr  DCN spikes (~2 MB)
SAVE_RASTER_PF        = 0       # .grr  only the N recorded PFs (n_pf_recorded in config.py),
                                #       numbered PKJ x N_PF_PER_PKJ + fibre; int32 like CbmSim's GR

# other data (load with json.load / np.load; sizes are for a default 300 s run)
SAVE_SUMMARY          = 1       # summary.json        every setting above + the printed results (tiny)
SAVE_WEIGHTS          = 1       # weights.npz         mean weight per CF territory and the tracked
                                #                     synapses, every 10 ms (~5 MB)
SAVE_FINAL_WEIGHTS    = 1       # final_weights.npy   all 160,000 weights at the end (~1 MB)
SAVE_TRACES           = 0       # traces.npz          voltages, olive calcium and DCN->IO inhibition
                                #                     at every step of the last 3 s (~10 MB)

# ==========================================================================
#  ADVANCED -- the calibrated operating point.
#  The synaptic gains below were fitted so the loop sits at sensible rates
#  (PKJ ~60 Hz, DCN ~15 Hz, IO ~1 Hz). Changing them moves the model away from
#  that point. config.py explains what each one sets.
# ==========================================================================
DT_MS                 = 1.0       # time step of the main loop, ms
PF_PKJ_GAIN           = 0.000137  # PF -> PKJ excitation per (weight x PF spike)
PKJ_DCN_GAIN          = 0.007717  # PKJ -> DCN inhibition per PKJ spike
DCN_IO_GABA_GAIN      = 0.0974    # DCN -> IO inhibition per DCN spike
PKJ_BASELINE_HZ       = 50.0      # PKJ rate with no input at all
DCN_BASELINE_HZ       = 60.0      # DCN rate with no input at all
CF_BURST_SPIKES       = 3         # complex spike: forced PKJ spikes per CF event (0 = none)
CF_BURST_ISI_MS       = 2.0       # time between those spikes
CF_PAUSE_G            = 1.0       # strength of the pause that follows the complex spike
CF_PAUSE_MS           = 10.0      # length of that pause, after the last complex-spike spike
CONNECTIVITY_SEED     = 0         # seed for the random part of the PKJ->DCN wiring
IO_HETEROGENEITY_SEED = 0         # seed for IO_HETEROGENEITY_CV's per-cell draw

# ==========================================================================
#  Machinery below: command-line overrides, sanity checks, and the run itself.
# ==========================================================================
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))             # so `import config` and `import sim...` work from any directory


def _apply_overrides(argv):
    """Apply NAME=VALUE arguments by overwriting the matching setting above.

    Each VALUE is parsed as a Python literal (600, 0.5, None, True); anything
    that isn't one, such as a folder name, is kept as a string."""
    g = globals()
    for arg in argv:
        if "=" not in arg:
            raise SystemExit(f"cannot parse {arg!r}; expected NAME=VALUE, e.g. NUM_TRIALS=120")
        key, _, raw = arg.partition("=")
        key = key.strip().upper()
        if key not in g:
            close = [k for k in g if k.isupper() and key[:4] in k]   # suggest settings with a similar name
            hint = f"  did you mean: {', '.join(sorted(close))}" if close else ""
            raise SystemExit(f"unknown setting {key!r}.{hint}")
        try:
            g[key] = eval(raw, {"__builtins__": {}}, {"None": None, "True": True, "False": False})
        except Exception:
            g[key] = raw


def check_settings():
    """Check the settings can build a network before anything runs.

    Collects every problem and reports them all at once. Returns the total
    number of Purkinje cells."""
    problems = []
    n_pkj = N_IO * N_PKJ_PER_IO
    if n_pkj % N_DCN:
        problems.append(
            f"N_IO x N_PKJ_PER_IO = {n_pkj} Purkinje cells is not a multiple of N_DCN = {N_DCN}.\n"
            f"    The DCN are laid out topographically, so each must own the same number of PKJ.")
    if N_PKJ_PER_DCN * N_DCN != N_DCN_PER_PKJ * n_pkj:
        problems.append(
            f"the PKJ->DCN synapse count does not close:\n"
            f"    from the DCN side: N_DCN x N_PKJ_PER_DCN = {N_DCN} x {N_PKJ_PER_DCN} = {N_DCN*N_PKJ_PER_DCN}\n"
            f"    from the PKJ side: {n_pkj} PKJ x N_DCN_PER_PKJ = {n_pkj} x {N_DCN_PER_PKJ} = {n_pkj*N_DCN_PER_PKJ}\n"
            f"    These must be equal. Adjust N_PKJ_PER_DCN or N_DCN_PER_PKJ.")
    if N_DCN_PER_IO > N_DCN:
        problems.append(f"N_DCN_PER_IO = {N_DCN_PER_IO} exceeds N_DCN = {N_DCN}.")
    if COUPLING and 2 * COUPLING_NEIGHBOURS >= N_IO:
        problems.append(
            f"COUPLING_NEIGHBOURS = {COUPLING_NEIGHBOURS} needs {2*COUPLING_NEIGHBOURS} distinct "
            f"partners, but there are only {N_IO} olivary cells.")
    if DELTA_PLUS <= 0 or DELTA_MINUS <= 0:
        problems.append("DELTA_PLUS and DELTA_MINUS must both be positive.")
    if not (W_MIN <= W_INIT <= W_MAX):
        problems.append(f"W_INIT = {W_INIT} is outside [W_MIN, W_MAX] = [{W_MIN}, {W_MAX}].")
    if not isinstance(NUM_TRIALS, int) or NUM_TRIALS < 1:
        problems.append(f"NUM_TRIALS = {NUM_TRIALS!r} must be a whole number, 1 or more.")
    if TRIAL_MS < DT_MS:
        problems.append(f"TRIAL_MS = {TRIAL_MS} must be at least one time step (DT_MS = {DT_MS}).")
    # CbmSim's raster files store trial numbers, time steps and cell indices as int16.
    if any((SAVE_RASTER_IO, SAVE_RASTER_PKJ, SAVE_RASTER_DCN, SAVE_RASTER_PF)):
        if round(TRIAL_MS / DT_MS) > 32767 or NUM_TRIALS > 32767 or max(n_pkj, N_DCN) > 32767:
            problems.append("raster files store trial numbers, time steps and cell indices as int16 (max 32767):\n"
                            f"    a trial is {round(TRIAL_MS / DT_MS)} steps, there are {NUM_TRIALS} trials, "
                            f"and {max(n_pkj, N_DCN)} cells of the largest type.")
    if problems:
        raise SystemExit("run.py: these settings cannot build a network:\n\n  - "
                         + "\n\n  - ".join(problems) + "\n")
    return n_pkj


def build_config():
    """Copy the SETTINGS block into a SimConfig (config.py), the object the
    simulator reads. Anything not listed here keeps its default from config.py."""
    from config import SimConfig
    return SimConfig(
        dt_ms=DT_MS, n_trials=NUM_TRIALS, trial_ms=TRIAL_MS, burn_in_s=BURN_IN_S, seed=SEED,
        n_io=N_IO, n_dcn=N_DCN, n_pkj_per_io=N_PKJ_PER_IO, n_pkj_per_dcn=N_PKJ_PER_DCN,
        n_dcn_per_pkj=N_DCN_PER_PKJ, n_dcn_per_io=N_DCN_PER_IO, n_pf_per_pkj=N_PF_PER_PKJ,
        pf_rate_hz=PF_RATE_HZ,
        gap_g=(COUPLING_STRENGTH if COUPLING else 0.0),
        gap_n_neighbors=COUPLING_NEIGHBOURS,
        ltd_window_ms=LTD_WINDOW_MS, null_window_ms=NULL_WINDOW_MS,
        delta_minus=DELTA_MINUS, delta_plus=DELTA_PLUS,
        w_init=W_INIT, w_min=W_MIN, w_max=W_MAX,
        ablate_dcn_io=bool(CUT_DCN_TO_IO),
        enforce_closed_loop=bool(CLOSED_LOOP),
        io_heterogeneity_cv=IO_HETEROGENEITY_CV,
        io_heterogeneity_seed=IO_HETEROGENEITY_SEED,
        connectivity_seed=CONNECTIVITY_SEED,
        pf_pkj_gain=PF_PKJ_GAIN, pkj_dcn_gain=PKJ_DCN_GAIN, dcn_io_gaba_gain=DCN_IO_GABA_GAIN,
        pkj_baseline_hz=PKJ_BASELINE_HZ, dcn_baseline_hz=DCN_BASELINE_HZ,
        cf_pause_g=CF_PAUSE_G, cf_pause_ms=CF_PAUSE_MS,
        cf_burst_spikes=CF_BURST_SPIKES, cf_burst_isi_ms=CF_BURST_ISI_MS,
    )


def main(argv):
    # 1. Read the command line and check the settings.
    _apply_overrides(argv)
    n_pkj = check_settings()

    from sim.analysis import (plot_io_state, plot_rasters, plot_voltage_traces,
                              plot_weights, summarize)
    from sim.rasters import FORMATS
    from sim.simulate import Simulation

    # 2. Build the configuration, and work out the CF rate the learning rule predicts.
    cfg = build_config()
    predicted_hz = 1000.0 / (LTD_WINDOW_MS * (1.0 + DELTA_MINUS / DELTA_PLUS))
    total_s = NUM_TRIALS * TRIAL_MS / 1000.0

    # 3. Print what is about to run.
    print(f"""
io_equilibrium_sim
  network        {N_IO} IO / {N_DCN} DCN / {n_pkj} PKJ / {n_pkj*N_PF_PER_PKJ:,} PF
  coupling       {'ON  (%.4g mS/cm^2, %d neighbours)' % (COUPLING_STRENGTH, COUPLING_NEIGHBOURS)
                  if COUPLING else 'OFF'}
  plasticity     LTD window {LTD_WINDOW_MS:g} ms, null {NULL_WINDOW_MS:g} ms, """
          f"""d-/d+ = {DELTA_MINUS/DELTA_PLUS:g}
  lesions        {'DCN->IO CUT' if CUT_DCN_TO_IO else 'loop intact'}""" +
          ('' if CLOSED_LOOP else ', feedback MISROUTED') +
          (f', IO heterogeneity cv {IO_HETEROGENEITY_CV:g}' if IO_HETEROGENEITY_CV else '') + f"""
  trials         {NUM_TRIALS} x {TRIAL_MS / 1000:g} s = {total_s:g} s simulated (+{BURN_IN_S:g} s burn-in), seed {SEED}
  predicted CF   {predicted_hz:.3f} Hz   (from the plasticity window and ratio)
""")
    if total_s < 300:
        print("  note: under 300 s the weights are still approaching their balance point,\n"
              "        so the drift below is measuring the transient, not an equilibrium.\n")

    # 4. Run the simulation, printing one line per trial. All results come back in one SimLog
    #    object (sim/simulate.py); raster files are written as it goes, at the end of each trial.
    out = Path(OUTPUT_DIR)
    raster_paths = {kind: out / (out.name + FORMATS[kind][0])
                    for kind, switch in (("io", SAVE_RASTER_IO), ("pkj", SAVE_RASTER_PKJ),
                                         ("dcn", SAVE_RASTER_DCN), ("pf", SAVE_RASTER_PF)) if switch}
    if raster_paths:
        out.mkdir(parents=True, exist_ok=True)

    def report_trial(trial, cf_counts, mean_weight, seconds):
        print(f"  trial {trial + 1:>{len(str(NUM_TRIALS))}}/{NUM_TRIALS}   "
              f"CF {cf_counts.mean() * 1000.0 / TRIAL_MS:6.3f} Hz   "
              f"mean weight {mean_weight:.4f}   ({seconds:.1f} s)")

    log = Simulation(cfg).run(raster_paths=raster_paths, on_trial_end=report_trial)

    # 5. Reduce the log to a handful of numbers (sim/analysis.py) and print them.
    s = summarize(log)

    print(f"""
results (second half of the run)
  CF / IO rate         {s['io_rate_hz']:.4f} Hz      predicted {predicted_hz:.3f} Hz"""
          f"""   ({s['io_rate_hz']-predicted_hz:+.4f})
  IO ISI CV            {s['io_isi_cv']:.4f}
  PKJ rate             {s['pkj_rate_hz']:.2f} Hz
  DCN rate             {s['dcn_rate_hz']:.2f} Hz
  mean PF->PKJ weight  {s['mean_weight']:.4f}
  weight drift         {s['weight_drift_slope']:+.6f} units/s
  saturated?           {'YES - weights are pinned at a bound' if s['weight_saturated'] else 'no'}
  spread across synapses {s['cross_synapse_std']:.4f}
""")
    if s['weight_saturated']:
        print("  The weights hit a bound, so the loop could not reach its balance point.\n"
              "  Usually N_PF_PER_PKJ is too small to supply the drive the target rate needs.\n")

    # 6. Save whatever the output switches ask for.
    saved = [p.name for p in raster_paths.values()] + save_data(log, s, out)
    for switch, fn, name in ((PLOT_RASTERS, plot_rasters, "rasters"), (PLOT_WEIGHTS, plot_weights, "weights"),
                             (PLOT_VOLTAGES, plot_voltage_traces, "voltages"),
                             (PLOT_IO_STATE, plot_io_state, "io_state")):
        if not switch:
            continue
        out.mkdir(parents=True, exist_ok=True)
        try:
            fn(log, out / f"{name}.png")
            saved.append(f"{name}.png")
        except Exception as exc:                       # if one figure fails, still save the others
            print(f"  figure {name} failed: {exc!r}")
    if saved:
        print(f"saved -> {out}/  " + ", ".join(saved))
    return 0


def save_data(log, summary, out):
    """Write the data files the SAVE_* switches ask for into `out`.
    Returns the names of the files written. All times are in ms, with t = 0 at
    the end of burn-in."""
    import json
    saved = []
    if any((SAVE_SUMMARY, SAVE_WEIGHTS, SAVE_FINAL_WEIGHTS, SAVE_TRACES)):
        out.mkdir(parents=True, exist_ok=True)

    if SAVE_SUMMARY:
        # The settings are every upper-case name in this file, as they stood for this run
        # (command-line overrides included).
        settings = {k: v for k, v in globals().items() if k.isupper() and k != "HERE"}
        results = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in summary.items()}
        trials = {"cf_rate_hz": (log.trial_cf_counts.mean(axis=1) * 1000.0 / log.trial_ms).tolist(),
                  "mean_weight_at_end": log.trial_mean_weight.tolist()}   # one entry per trial
        with open(out / "summary.json", "w") as f:
            json.dump({"settings": settings, "results": results, "trials": trials}, f, indent=2)
        saved.append("summary.json")

    if SAVE_WEIGHTS:
        np.savez_compressed(out / "weights.npz",
                            t_ms=log.t_ms,                         # (T,) sample times
                            mean_weight=log.mean_weight,           # (T, n_io) mean per CF territory
                            sample_weights=log.sample_weights,     # (T, n_tracked) individual synapses
                            tracked_synapses=log.tracked_synapses, # (n_tracked, 2) their (PKJ, fibre)
                            io_of_pkj=log.io_of_pkj)               # (n_pkj,) CF territory of each PKJ
        saved.append("weights.npz")

    if SAVE_FINAL_WEIGHTS:
        np.save(out / "final_weights.npy", log.final_weights)     # (n_pkj, n_pf_per_pkj)
        saved.append("final_weights.npy")

    if SAVE_TRACES:
        np.savez_compressed(out / "traces.npz",
                            t_ms=log.trace_t_ms,                   # (S,) one sample per step
                            io_v=log.trace_io_v, io_ca=log.trace_io_ca, io_gaba=log.trace_io_gaba,
                            pkj_v=log.trace_pkj_v, dcn_v=log.trace_dcn_v)
        saved.append("traces.npz")

    return saved


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
