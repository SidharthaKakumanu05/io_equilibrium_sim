#!/usr/bin/env bash
# Parallel-fiber pool-size sweep, run through a bounded queue.
#
# Why a queue and not just 39 background processes: this box has 64 cores but
# the simulation is memory-bandwidth-bound, not core-bound -- every step draws
# an (n_pkj x n_pf) Bernoulli array and multiplies it against the weight matrix,
# which is 2.5 MB of traffic per step per process. Launching 43 at once measured
# 11x realtime per process against 2.6x for one, i.e. the box delivered LESS
# total work than a third of the cores would have. CONCURRENCY below is set to
# leave headroom alongside the four 5-hour runs; raise it only after checking
# that per-process throughput in the logs has not degraded.
#
# 13 pool sizes x 3 seeds = 39 runs of 900 simulated seconds.
set -u
ROOT=${1:-/home/sk57289/longsim}
CONCURRENCY=${2:-6}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/logs/pfsweep"
cd "$REPO" || exit 1

for n in 100 125 150 175 200 250 300 350 500 700 1000 1400 2000; do
  for seed in 0 1 2; do
    echo "$n $seed"
  done
done | xargs -P "$CONCURRENCY" -n 2 bash -c '
  n=$0; seed=$1
  out="'"$ROOT"'/pfsweep/pf${n}_s${seed}"
  [ -f "$out/meta.json" ] && { echo "skip pf$n/s$seed (done)"; exit 0; }
  python3 experiments/run_long.py --preset "pf$n" --duration-s 900 --seed "$seed" \
      --out-dir "$out" --trace-len-s 10 \
      > "'"$ROOT"'/logs/pfsweep/pf${n}_s${seed}.log" 2>&1
  echo "finished pf$n/s$seed"
'
echo "pf sweep complete"
