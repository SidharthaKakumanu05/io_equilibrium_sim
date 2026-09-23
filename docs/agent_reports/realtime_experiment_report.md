# Four-hour campaign preparation and concurrent throughput pilots

Status: **running**. The coordinator launched the eight-run durable campaign at
2026-09-21 16:24:31 UTC after accepting all validation and resource gates.
Four seed-0 trials are active; four seed-1 trials are queued. All active trials
have committed 720 biological seconds. Their checkpoint-based rates are
1.076–1.097 biological seconds per wall second, including burn-in and earlier
checkpoint costs. Final four-hour outcomes remain pending.
The user selected additive, binary, Abbott cascade and Mauk cascade, seeds 0 and 1.
Each run has 14,400 biological seconds (14,400,000 steps), after 8 seconds of burn-in.

## Exact source and scheduling

The coordinator accepted 125 tests (63.227 s), eight complete old/new scientific
state/event/RNG comparisons and ten fault-recovery comparisons at zero tolerance.
This stage changes only the supervisor: a queued job may start only when its
frozen CPU is unoccupied, preserving CPU exclusivity when workers finish out of
order. Two targeted scheduling tests pass. Scientific source, recorder,
checkpointing, validation and analysis are unchanged from that accepted gate.

Frozen source: `/home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/source`. All Python source/tests, C source, compiled
extension, build script/manifest, README and workflow instructions are preserved
read-only. Provenance retains full file hashes, dependencies, dirty Git diffs,
native backend identity and validation evidence. No scientific data were removed.

## Four-worker measurements

Every pilot uses the full network, original 1 ms step, native discrete
probabilities 0.9/0.1, matched shallow 0.25/0.55 initialization and homeostasis off.
Each has 600 active biological seconds, 8-second burn, one-second summaries,
15 seconds of voltage/event windows, full-state checkpoints every 240 active
seconds, and automatic validation plus analysis. No compilation occurs in runs.
Each process uses one thread. Simulation wall time includes initialization, burn,
recording and checkpoint writes; subprocess wall time additionally includes
imports, final validation and analysis. The simulator's timer is sampled just
before its final empty-record checkpoint, whose remaining write time is included
in subprocess wall time. The 240–480-second interval includes the middle voltage window and checkpoint work. Speedup compares active biological throughput with the previous 120-second seed-0
pipeline trials. Duration, checkpoint interval and placement differ, so this is
a delivered workflow comparison, not a paired kernel benchmark.

| Mode | CPU | Simulation wall s | Active bio s / sim wall s | Full subprocess wall s | Active bio s / full wall s | Steady interval rate | Workflow speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| mauk_cascade | 14 | 552.944 | 1.0851 | 556.282 | 1.0786 | 1.0999 | 3.24× |
| binary | 6 | 558.622 | 1.0741 | 561.904 | 1.0678 | 1.0914 | 3.07× |
| additive | 2 | 562.584 | 1.0665 | 566.918 | 1.0584 | 1.0790 | 2.92× |
| abbott_cascade | 10 | 564.051 | 1.0637 | 567.726 | 1.0568 | 1.0814 | 3.17× |

The initial placement [4,5,16,17] shared L3 caches and placed two workers on a
NUMA node with no RAM. All four initial pilots passed scientific validation but
missed the speed target (0.747–0.813 active biological seconds/wall second).
The repeat uses [2,6,10,14], separate L3 caches on RAM-attached nodes 0 and 2.
The separate-cache 120-second repeat improved to 0.945–0.984 active biological
seconds/wall second but still missed the conservative target. The table above
uses a further 600-second repeat with the production checkpoint cadence. All
three sets of outputs and timing evidence are retained; topology is archived in
`lscpu_topology.txt` and `numa_memory.txt`. The coordinator also compared both 120-second placements against one another
and against the preoptimization 120-second datasets: complete model/RNG/recorder
state and every recorded array match exactly in all four modes
(`realtime_optimization_20260921/cpu_placement_equivalence.json`).
These are bounded measurements, not
a guarantee of sustained four-hour performance.

## Recording and resource gate

The long pipeline retains all 71 generations per run: initial state, eight burn
checkpoints, active-start state, 60 four-minute checkpoints and finished state.
Full float64 model/RNG/connectivity/eligibility state is retained. One-second
records include per-cell rates, weight mean/SD/bound occupancy/histograms and
400 fixed sampled individual weights; discrete runs add hidden-state occupancy,
tracked states, switch counters and event counts. Complete CF events are saved.
Additive one-second records contain LTD/LTP totals; a separate null-event time
series is unavailable.
PF activity covers 40 fixed sampled fibers; PF/PKJ/DCN detailed events and
voltage/calcium/conductance/gap-current traces are bounded to three five-second
windows. CF events are not modeled bursts. These background runs do not test
conditioned pattern retention.

Storage projects bounded trace bytes once and linear record bytes for 14,400 s,
using uncompressed payload sizes. It adds 71 times the largest observed state
checkpoint plus 50% state-size margin, then 20% for metadata, analysis and a
temporary generation. Eight long runs project 6.32 GiB;
all existing campaigns plus this projection total 10.63 GiB, below the
24 GiB cap. Free space is 68.21 GiB before long runs, preserving the
20 GiB reserve. Peak pilot RSS is 160.9 MiB per worker.
Ceilings remain four single-thread workers, 8 GiB each, 32 GiB total, 48 wall
hours per trial. Aggregate storage checks are serialized at checkpoint publication.

## Execution and evidence

`prepared_campaign.json`, `pilot_gate.json`, `storage_gate.json`, scheduler test
logs and all pilot-duration and placement records are in `docs/agent_reports/evidence/realtime_runner_20260921`.
Dedicated campaign progress is `/home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/control/progress.json`; historical
central stage records are preserved. Supervisor PID 1465938 and watcher PID
1466538 have durable boot/start-time identities in
`docs/agent_reports/evidence/realtime_optimization_20260921/launch_verified.json`.
The watcher posts aggregate 10% milestones, failure alerts, and a completion
handoff to the existing conversation. Per-run launch/resume and
analysis-only commands are recorded in `prepared_campaign.json`.

Recorded launch entry point (already launched; do not duplicate):
```bash
cd /home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/source && /usr/bin/python3 /home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/source/experiments/run_background_campaign.py launch --campaign /home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/campaign.json
```
Resume after reconciling live identities/checkpoints:
```bash
cd /home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/source && /usr/bin/python3 /home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/source/experiments/run_background_campaign.py launch --campaign /home/sk57289/io_equilibrium_sim/results/workflow_campaign/realtime_20260921T160422Z/campaign.json --resume
```
The durable per-run sequence is simulation → validation → COMPLETE → analysis.
The supervisor records exact run/analysis resume commands. Completed runs are
validated and their analysis checksums audited on campaign resume; they are not
rerun. Source, environment and native-backend mismatch stop resume. No analyst
synthesis or successful-completion claim is made before all eight runs finish.
