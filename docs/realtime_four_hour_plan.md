# Real-time optimization and four-hour comparison

The user requests significant acceleration toward one biological hour per wall
hour, followed by long runs. Their scope clarification is four primary modes,
two seeds each: additive, binary, Abbott cascade and Mauk cascade, seeds 0 and 1.
Each run covers 14,400 biological seconds after the existing 8-second burn-in.

## Scientific settings

Use the same full-network settings as the completed focused comparison at
`results/workflow_campaign/cascade_20260921T145424Z/`: 40 IO, 320 PKJ, 80 DCN,
160,000 PF synapses, 1 ms outer timestep, unchanged IO substeps, background
activity, native discrete probabilities and shared shallow two-level initial
weights within seed. Additive keeps [0,1] bounds; discrete strengths are
0.25/0.55. Homeostasis is off. No slower-probability or narrow-additive campaign
variants were requested in the clarification. No rate or model retuning.

## Performance and validation gates

The accepted 118-test source snapshot from the focused comparison is the exact
pre-optimization baseline. Prior cleanup is complete; no further scientific
data deletion is authorized by this request. Preserve all existing dirty work.

1. Profile the current representative full network and record repeated runtimes.
2. Optimize a measured bottleneck at a time. Preserve equations, operation
   ordering where numerically significant, float64, every event and all RNG
   streams/draw order. Compile arithmetic if helpful; keep the reference path.
3. Compare complete fixed-seed scientific state, event bytes, recorded values
   and RNG state exactly, including all four plasticity modes and relevant
   noisy, heterogeneous and parameter-change configurations. No silent tolerance
   relaxation. If a numerical departure is necessary, prepare its quantified
   effect and obtain the approval required by WORKFLOW before using it.
4. Run the full suite, then complete-state interrupted checkpoint recovery
   with nonempty pending PF history and nontrivial cascade states.
5. Freeze Python/native source, compiled binary, build flags/identity and
   dependencies. Require matching source/backend provenance on resume.
6. Measure pilots using recording and durable checkpointing, including the
   planned four-worker workload. Report biological seconds / wall second per
   trial; aggregate throughput is a separate quantity. Do not claim real-time
   from a neuron-only microbenchmark or by excluding substantial run overhead.

The target is >=1.0 biological second per wall second. Report both acceleration
relative to the paired baseline and the absolute achieved rate. A speed claim
must identify hardware, threads, recording, duration, initialization and any
compilation cost. A short pilot estimates four-hour runtime, not a guarantee of
the sustained rate over four hours.

## Long execution

Standalone Eccles and the existing ceiling of four single-thread workers,
8 GiB RAM per worker, 32 GiB total RAM, 24 GiB aggregate campaign storage,
20 GiB free reserve and 48-hour wall limit per trial remain applicable. Check
live jobs and filesystem headroom before launch. Project agents remain
sequential, with one simulation-source owner at a time.

Use the durable pipeline, approximately 240–300 biological seconds between
full-state checkpoints, one-second weight/rate/state summaries, fixed selected
synapse trajectories, complete CF events and bounded early/middle/late voltage
windows. Estimate all eight runs' storage from measured pilots before launch.
Each job must perform simulation -> validation -> COMPLETE -> automatic analysis.
Store exact per-run and campaign resume commands and durable process identity.
Record progress, failures and completion; do not count a partial run as success.

After successful completion, analyze full-duration means, individual movement,
switches/hidden states and neural activity across both seeds. Long background
runs still do not by themselves measure retention of a conditioned network
pattern. Preserve the earlier focused comparison and completed homeostasis run.

## Validation and hardware placement update

The optimized scientific source passed 125 tests, eight independent old/new
state comparisons and ten interrupted-recovery comparisons, all exact. The
four 120-second optimized pilots also match the retained preoptimization
120-second datasets exactly in complete model state, recorder state and all
record arrays. Evidence is in
`docs/agent_reports/evidence/realtime_optimization_20260921/`.

Four-worker timing exposed hardware contention absent from isolated timings.
The initial CPUs 4/5 and 16/17 share two L3 caches, and CPUs 16/17 belong to a
NUMA node with no local RAM on this host. Placement on CPUs 2/6/10/14 uses four
distinct L3 caches on the two nodes with RAM. Both placements' results are
preserved and scientifically identical. The better 120-second placement still
falls below the conservative active-time/end-to-end real-time target. Four
600-second pilots with the production 240-second checkpoint cadence subsequently
passed that target at 1.057–1.079 biological seconds per wall second, including
validation and analysis. The coordinator accepted the full launch gate and
started the eight-run campaign at 16:24:31 UTC. Completion and final comparison
remain pending; the durable watcher will hand back control after all eight runs
and their automatic analyses pass.
