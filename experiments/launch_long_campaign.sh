#!/usr/bin/env bash
# Launch the whole long-run campaign in parallel and return immediately.
#
# Two tiers, because they answer different questions and cost differently:
#
#   Tier 1 -- four conditions at 5 simulated hours (18,000 s). These are the
#   headline runs: the equilibrium claim is about a steady state, and 5 hours is
#   ~600 climbing-fiber intervals per olivary cell, which is enough to say
#   something about drift rather than about the transient. ~13 h wall each, all
#   four at once.
#
#   Tier 2 -- a 13-point parallel-fiber pool-size sweep at 900 s, 3 seeds each.
#   The weights equilibrate in a few hundred seconds, so 900 s is enough for
#   this sweep's question (where does the loop settle, and where does it stop
#   being able to) at 10x the length of the 90 s sweep already in the README.
#   ~40 min to ~1.5 h wall each depending on pool size, all 39 at once.
#
# 43 of the box's 64 cores, ~20 GB of disk. Nothing here is multi-threaded; the
# parallelism is entirely across processes.
set -u
ROOT=${1:-/home/sk57289/longsim}
TIER=${2:-all}                                    # all | 1 | 2 , so a half-failed launch can be resumed
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/logs"
cd "$REPO" || exit 1

launch() {  # launch <preset> <duration_s> <seed> <name>
  local preset=$1 dur=$2 seed=$3 name=$4
  mkdir -p "$(dirname "$ROOT/logs/$name.log")"    # the sweep names contain a / , so the log dir nests too
  setsid nohup python3 experiments/run_long.py \
      --preset "$preset" --duration-s "$dur" --seed "$seed" \
      --out-dir "$ROOT/$name" \
      > "$ROOT/logs/$name.log" 2>&1 < /dev/null &
  echo "  launched $name (pid $!)"
}

if [ "$TIER" = all ] || [ "$TIER" = 1 ]; then
echo "Tier 1: four conditions at 18,000 s (5 simulated hours)"
for preset in baseline cf_inh gap_off open_loop; do
  launch "$preset" 18000 0 "long_$preset"
done
fi

if [ "$TIER" = all ] || [ "$TIER" = 2 ]; then
echo "Tier 2: parallel-fiber pool-size sweep, 900 s x 3 seeds"
for n in 100 125 150 175 200 250 300 350 500 700 1000 1400 2000; do
  for seed in 0 1 2; do
    launch "pf$n" 900 "$seed" "pfsweep/pf${n}_s${seed}"
  done
done
fi

echo
echo "all launched. watch with:  tail -f $ROOT/logs/long_baseline.log"
