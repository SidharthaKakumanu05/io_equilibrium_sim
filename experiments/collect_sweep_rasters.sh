#!/usr/bin/env bash
# Per-run figures (rasters, voltages, weights, IO state) for every pool-size
# sweep bundle, seed 0.
#
# Separate from collect_long_campaign.sh because it is the optional half: the
# sweep's result is the six-panel summary in pf_sweep.png, and these are the
# raw per-condition rasters behind it. Seed 0 only -- three seeds of the same
# raster says nothing the first one did not.
set -u
ROOT=${1:-/home/sk57289/longsim}
OUT=${2:-/home/sk57289/io_equilibrium_sim/results/long/pfsweep_runs}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 1

while [ "$(find "$ROOT/pfsweep" -name meta.json 2>/dev/null | wc -l)" -lt 39 ]; do
  sleep 120
done
for d in "$ROOT"/pfsweep/pf*_s0; do
  [ -f "$d/meta.json" ] || continue
  echo "[$(date '+%H:%M:%S')] $d"
  python3 experiments/make_long_figures.py --bundle "$d" --out-dir "$OUT"
done
echo "[$(date '+%H:%M:%S')] sweep rasters done -> $OUT"
