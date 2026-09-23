# Sequential simulation workflow

Status: sequential execution authorized by the user on 2026-09-18; consult
`docs/agent_reports/progress.json` for current stage, decisions and ceilings.
Each stage remains subject to the acceptance gates below. Preserve the dirty working tree and read
`HANDOFF_drift_diagnosis.rtf` before any stage. Its open-work section is a list
of possible follow-ups, not a launch request.

## Agents and activation

The four project agents live in `.codex/agents/{cleaner,optimizer,experiment_runner,analyst}.toml`.
Codex CLI 0.155.0 supports standalone TOML with `name`, `description`, and
`developer_instructions`; no legacy role registration is needed. Model,
reasoning, sandbox, and approval overrides are omitted so they inherit from the
parent. At setup the user configuration selects `gpt-6-astra` with high reasoning.
Project configuration limits open child-agent threads to one. Finish and close
each child before starting the next; the coordinator runs tests/validation gates.
Do not run these agents in parallel, including in separate Codex sessions.

Reference: [official custom-agent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents).
Start a fresh Codex session from this repository before using the newly installed
roles. No CLI upgrade is needed. Existing session instructions/tool definitions
should not be assumed to reload automatically. Installation does not spawn agents.

## Stage order and acceptance

`cleaner -> tests -> optimizer -> validation -> experiment_runner -> successful completion -> analyst`

| Stage | Acceptance criteria | Evidence / report |
| --- | --- | --- |
| cleaner | Record baseline dirty-tree identity; run the full unittest suite before edits and after edits; both pass. Remove only code/comments demonstrated obsolete by call-site, configuration, experiment, and test inspection. Preserve scientific rationale and behavior; no data deletion. A justified no-op is acceptable. | `docs/agent_reports/cleanup_report.md`, command logs, file-level removal rationale and diff |
| tests | Coordinator reviews cleanup diff and passing before/after results for the exact current tree. If the tree changed after testing, rerun affected checks/full suite as appropriate. No unresolved regressions. | `cleanup_report.md`, `progress.json` tests entry with command, exit status, tree identity |
| optimizer | Reproducible fixed-seed baseline and profiler results precede edits. One measured bottleneck per change; report repeated timing, peak RSS, output size, and recording-on/off overhead without changing RNG consumption. No-op is acceptable if no safe benefit. | `docs/agent_reports/performance_report.md`, baseline/optimized profiles and comparisons |
| validation | Full unittest suite passes; fixed-seed state/output comparisons pass for representative baseline and relevant diagnostic configurations. Equations, dt, plasticity, connectivity and RNG behavior unchanged. Compare spike/event indices, connectivity, and RNG states exactly. Default floating tolerance is `rtol=0, atol=0`; any departure from exact equivalence requires explanation, explicit numeric tolerances with units, and user approval **before proceeding**. | `performance_report.md`, comparison artifacts, approval reference if applicable, validation entry in progress |
| experiment_runner | Implement and validate the reproducible pipeline using the existing simulator. Inventory observables, storage, and missing instrumentation; complete-state/RNG checkpoints and interrupted recovery pass; short bounded pilot passes; automatic post-run analysis is wired into the durable job. Resolve launch decisions below before long runs. | `docs/agent_reports/experiment_report.md`, manifest, pilot and recovery logs, exact launch/resume commands |
| successful completion | Every requested trial exits successfully, reaches its declared final simulated step, and passes completeness/finite-value/unit/timestamp/configuration checks. Flush/close all recorders; verify checksums, expected files/rows, and no missing/duplicate resumed samples. Atomically publish a completion marker only after verification. No failed or partial trial counts as success. | Per-run completion manifest, job exit/status logs, automatic analysis exit status, experiment report, run IDs in progress |
| analyst | Read all three preceding reports and Claude's handoff; audit data before interpretation. Analyze weight means AND dispersion/drift/trajectories, neural activity, plasticity, and performance. Identify missing observables, confounds, uncertainty and seed counts. Keep measured results, interpretations and untested hypotheses separate. | `docs/agent_reports/final_report.md`, analysis scripts and figures outside raw-data directories |

Use `python3 -m unittest discover -s tests -q` from the repository root. Do not
install pytest merely to run this repository's tests. The historical suite took
about 500 seconds; do not confuse configuration validation with running it.

No overlapping edits to simulation source (`config.py`, `sim/`, simulation
drivers). The active stage owns source edits until its tests and report finish.
The runner may add instrumentation/persistence only after optimizer validation;
any runner source change must repeat affected scientific validation before pilot
or long-run acceptance. Pin an immutable code snapshot per run so later edits
cannot change a resumed run. The analyst must not edit source or raw datasets.

## Persistent progress and restart

`docs/agent_reports/progress.json` is the durable coordinator record. Update it
on stage transitions, launch, checkpoint, failure, and completion. Write a sibling
temporary JSON file, flush/fsync, then atomically replace it. Only the current
stage/coordinator writes it. Keep completed-stage evidence when recording a blocker.

Record UTC timestamps, completed stages, active stage/owner, blockers, report
paths, test/validation commands and exit codes, code revision, dirty-tree
identity, run IDs, scheduler job IDs or durable process identity, manifest and
checkpoint paths, final-step expectations, and **exact copy-pastable resume
commands** including absolute cwd, Python executable, config, seed, and checkpoint.
Record failed attempts separately. A PID alone is insufficient. Reconcile live
jobs and checkpoint/output integrity before resume; never duplicate a live run.

Current stage commands in progress open Codex with a concrete stage request;
they are instructions for the coordinator to delegate to the named custom agent,
not an invented `codex --agent` flag. Their presence is not authorization to run.
Run-level resume commands are initially empty because no resumable pipeline/run
exists; experiment_runner must fill and test them before launch. Never claim that
rerunning the current `run_long.py --out-dir ...` resumes a checkpoint: it can
truncate streams. Unknown or changed config/code on resume is a stop condition.

## Long-trial decisions and recording contract

Before launching long trials, clarify and record all of:

- Does “4–5 hours” mean wall-clock runtime (per trial or whole campaign) or
  simulated biological time? Record both budgets, burn-in, timestep and horizon
  with units. Do not use `run_long.py`'s default 18,000 simulated seconds as consent.
- Trial count, conditions and exact seeds; whether follow-ups proposed in the
  handoff are wanted. Parameter/model alternatives need a separate decision.
- Scheduler/partition/account policy and permitted compute node; CPU/thread and
  concurrent-job caps, RAM, wall-time, storage directory, quota, maximum output
  bytes, checkpoint retention, and reserved free disk. On an approved standalone
  host use a durable service or `setsid nohup`; never heavy jobs on a login node.

Inventory PF (sampled identities and representativeness), PKJ, CF/IO and DCN
activity, population and per-cell rates, weight distributions and fixed individual
trajectories, LTD/LTP/null event counts, and runtime/peak-memory/recording overhead
where supported. Distinguish CF events from any modeled bursts. Inspect diagnostic
fields in `experiments/run_drift_diagnostics.py` and the handoff before adding
instrumentation. Mark unsupported quantities missing; do not fabricate counts.

For each observable record units, shape/dtype, cadence or event rule, sampled
identities, aggregation/windowing, and expected bytes. A starting proposal is
event-based buffered spikes, 1-second weight summaries/selected trajectories,
and short early/middle/late voltage windows; justify it against the scientific
question and pilot. Do not save everything every timestep. Compute storage from
uncompressed bytes per sample/event times count, including all trials,
checkpoints, temporary copies, logs, provenance and analysis; measure actual
compression and event rates in the pilot rather than assuming them. Check both
filesystem free space and quota, and enforce the agreed reserve during recording.

Each run manifest includes the full resolved configuration (including
`weight_dependence`), all seeds, Git revision plus staged/unstaged binary diffs,
relevant untracked source contents/hashes, dependency versions, Python/platform,
hardware/thread environment, host/job identity, UTC start/end timestamps, units,
recording schema and exact command. Git HEAD alone is insufficient for this tree.
Retain a runnable source snapshot; a list of hashes alone cannot reconstruct it.

Buffer recording. Atomically publish versioned checkpoints on the same filesystem
after flushing/fsyncing buffers and recording committed offsets. Include every
mutable neuron/channel/gating/calcium/synaptic/plasticity state, weights,
connectivity/heterogeneity, delayed-event queues/history, simulated clock/step,
burn-in status, recording selections/cursors/counters and **all RNG states**.
Save full precision; `final.npz` float32 weights are not a complete checkpoint.
Checkpoint and stream offsets must form a consistent generation; recovery must
not duplicate samples or advance RNG. Preserve crash output for diagnosis and
never delete pre-existing data without explicit approval.

Test an interrupted short run against the same uninterrupted fixed-seed run,
including nonzero pending state and interruption around buffer/checkpoint
boundaries. Compare complete final state, RNG and recordings under the approved
tolerances. Run a short pilot before any long trial; use measured throughput,
RSS and bytes to revise budget estimates. This implementation/recovery/pilot work
belongs to the future experiment_runner stage, not initial agent setup.

## Durable basic analysis

Existing saved RunBundle analysis (no simulation) is:

```bash
python3 experiments/make_long_figures.py --bundle "$RUN_DIR" --out-dir "$ANALYSIS_DIR"
```

`ANALYSIS_DIR` must be outside the raw bundle. For a successfully validated bundle
on an approved standalone host, this command survives SSH disconnection:

```bash
mkdir -p "$ANALYSIS_DIR"
setsid nohup bash -c '
  set -euo pipefail
  cd /home/sk57289/io_equilibrium_sim
  test -f "$1/COMPLETE"
  MPLBACKEND=Agg python3 experiments/make_long_figures.py --bundle "$1" --out-dir "$2"
' analysis "$RUN_DIR" "$ANALYSIS_DIR" > "$ANALYSIS_DIR/post_run.log" 2>&1 < /dev/null &
```

`COMPLETE` is a **new pipeline contract**, absent from the old runner: only
experiment_runner's validated success path may atomically create it. Do not
create it manually to bypass completeness checks. Future pipeline code must
validate NaN/Inf, units, timestamps, config and completeness before this command;
document explained undefined derived statistics separately. If the runner
changes bundle format, adapt/test the analysis command during the pilot.

Before long launch the runner must wire basic analysis into the durable job as
`run -> validate -> publish COMPLETE -> analysis`, or submit a scheduler analysis
job with a success dependency (e.g. Slurm `afterok`) on that validated run job.
Submitting analysis later from a Codex chat is insufficient. Persist concrete
absolute paths/environment and the analysis command/job ID in progress. A failed
analysis must have its own resumable command; do not rerun a successful simulation
just to regenerate figures. This deterministic command is basic analysis; the
analyst's cross-report synthesis still follows successful completion.

## Stop conditions

Stop stage advancement on failed tests or equivalence/recovery validation,
unapproved numerical differences, unresolved provenance/configuration mismatch,
missing/corrupt data, insufficient quota/free-disk reserve, exceeded resource
limits, denied filesystem/scheduler permissions, or unknown login-node policy.
Record the blocker and last valid checkpoint; preserve evidence and existing
data. During runs stop safely at a validated checkpoint when possible; never
claim completion after an error or relax scientific criteria to keep going.
Missing launch decisions block long trials, not preparation that is otherwise
authorized. No cleanup or long trial was started as part of the initial setup.
