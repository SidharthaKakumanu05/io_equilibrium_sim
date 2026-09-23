#!/usr/bin/env bash
# Wait for the campaign to finish, then make every figure from it.
#
# Runs unattended: the 5-hour runs take ~13 wall-clock hours and the pool-size
# sweep a few, and there is no reason for either to need a person present when
# it lands. This polls for the bundles, draws the pool-size sweep as soon as
# that tier is complete (it finishes first), and draws the per-run and
# cross-condition figures when the 5-hour runs land.
#
# Safe to re-run: every figure is regenerated from the bundle, nothing is
# consumed, and a tier that is already drawn is simply drawn again.
set -u
ROOT=${1:-/home/sk57289/longsim}
OUT=${2:-/home/sk57289/io_equilibrium_sim/results/long}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1
mkdir -p "$OUT"

LONG_RUNS="long_baseline long_cf_inh long_gap_off long_open_loop"
N_SWEEP=39

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- tier 2 first: it is 900 s a run and finishes hours before the others ---
say "waiting for the pool-size sweep ($N_SWEEP bundles)..."
while [ "$(find "$ROOT/pfsweep" -name meta.json 2>/dev/null | wc -l)" -lt "$N_SWEEP" ]; do
  sleep 120
done
say "sweep complete; drawing it"
python3 experiments/pf_sweep_figures.py --pf-sweep "$ROOT/pfsweep" --out-dir "$OUT" \
  2>&1 | tee "$OUT/pf_sweep_stdout.txt"

# --- tier 1 ---
for run in $LONG_RUNS; do
  say "waiting for $run..."
  while [ ! -f "$ROOT/$run/meta.json" ]; do sleep 120; done
  say "$run landed; drawing its figures"
  python3 experiments/make_long_figures.py --bundle "$ROOT/$run" --out-dir "$OUT" \
    2>&1 | tee -a "$OUT/long_runs_stdout.txt"
done

say "all four 5-hour runs in; drawing the cross-condition comparison"
# shellcheck disable=SC2086  # word splitting of the run list is intended
python3 experiments/pf_sweep_figures.py --out-dir "$OUT" \
  --compare $(for r in $LONG_RUNS; do echo "$ROOT/$r"; done) \
  2>&1 | tee -a "$OUT/compare_stdout.txt"

# The wiring model, re-drawn with the weights plasticity actually left behind.
say "re-drawing the 3D model with the final weights"
python3 experiments/run_network_orbit.py --out-dir "$OUT/final_weights" \
  --bundle "$ROOT/long_baseline" --no-gif 2>&1 | tail -3

say "done. everything is under $OUT"
