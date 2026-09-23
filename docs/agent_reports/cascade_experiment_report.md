# Focused cascade comparison: execution evidence

The user authorized implementing both Abbott and Mauk cascade rules and selected
replicated focused comparisons before any additional four-hour network runs.
The completed four-hour homeostasis dataset remains preserved and was not rerun.

## Preparation and validation

The sequential cleaner and optimizer stages are documented in
`cascade_cleanup_report.md` and `cascade_performance_report.md`. The authorized
cleanup removed 1,430,432,567 bytes of old test dumps/bytecode; scientific runs
were retained. Optimization was an evidenced no-op. Implementation is described
in `cascade_implementation_report.md`.

Coordinator evidence is under `evidence/cascade_validation_20260921/`:

- `full_tests.log`: all 118 tests pass, 497.367 seconds.
- `baseline_equivalence.json`: exact pre-change versus post-change scientific
  state, event bytes, RNG states and aliases for default additive, homeostatic
  additive and soft-bound additive configurations. Only explicitly enumerated
  inactive configuration metadata is excluded; rtol=atol=0.
- `recovery.json`: both cascade rules pass SIGKILL recovery at active buffer,
  record-write, pre-cursor and post-cursor boundaries. Eight comparisons include
  nonempty PF history, hidden states, transition RNG, scientific state, recorder
  state and every recorded array. Successful temporary binary dumps are removed;
  logs and comparisons remain.
- `assay_reordering.json`: eight variants match a chronological event reference
  exactly when independent synapses are processed by event rank. Uniforms remain
  attached to their events and within-synapse order is preserved.

## Frozen source and experiments

Campaign: `results/workflow_campaign/cascade_20260921T145424Z/`.
The `source/` snapshot contains 59 read-only source/documentation files. All
hashes were independently rechecked against `provenance/manifest.json`.
The snapshot, dirty-tree diffs, validation evidence and resolved configurations
are retained. The scientific protocol was specified before collecting results
in `docs/cascade_comparison_plan.md` and copied into the snapshot.

The isolated assay uses `experiments/run_cascade_assay.py`: eight variants,
three paired seeds and 512 individually tracked synapses per seed. The phases
are 120 s background, 120 s learning, 600 s retention, 120 s reversal and 600 s
reversed retention. Independent PF counts have the exact per-second marginal
of 1 ms Bernoulli activity at 20 Hz. IID resolved verdicts omit shared-CF timing
and circuit feedback; this is a synthetic rule assay, not a network simulation.
The production plasticity primitive applies every event. Initial assignments,
resolved verdicts and transition uniforms are shared across variants. Native
and shallow-step-matched probabilities separate hidden-state effects from
update magnitude; matched-range additive and binary controls address the
narrower two-level weight range. All 512 weights/states and switch counters are
retained every second, with hashes in the raw manifest.

Four sequential 10 s full-network pilots precede eight 120 s full-network
checks: additive, binary, Abbott and Mauk at seeds 0 and 1. All use the original
network and 1 ms timestep, 8 s burn-in, background activity, common two-level
initial assignments, and no homeostatic scaling. Additive keeps its original
0–1 bounds; discrete strengths are 0.25/0.55. No rates are retuned. The network
checks assess integration and operating point, not long-term memory.

The durable network queue performs simulation, full validation, atomic COMPLETE
publication and automatic basic analysis. `control/progress.json` records live
checkpoint/resume identities independently of historical completed campaigns;
the central progress file links to it. Limits are four single-thread workers,
8 GiB per worker, 32 GiB total RAM, 24 GiB campaign-root storage, a 20 GiB free
reserve and one hour per short trial. Pilot measurements and concurrency
acceptance are recorded before launch in `pilot_gate.json` and `launch_gate.json`.

## Completion and interpretation

All requested focused trials completed: 24 assay conditions (eight variants by
three seeds), four pilots, and eight network runs (four modes by two seeds).
All eight network runs reached step 120000 after 8000 burn-in steps and passed
validation plus automatic analysis. `completion_audit.json` independently
revalidated every raw dataset, completion/analysis checksum and frozen source;
the six cross-mode comparisons verified identical sampled PF counts, PF RNG,
tracked identities and initial tracked weights within each seed.

`assay_audit.json` independently verified all 24 raw artifact hashes, exact
driver/kernel provenance, 1561 finite snapshots per condition, identical initial
weights and eligibility totals within seed, every recorded state/weight
invariant and monotone/consistent transition counters. The assay analysis and
plot are in the campaign's `assay_analysis/`; its exact postprocessing driver is
archived under `provenance/analysis_code/`. Hash-verified CbmSim kernel, scheduler
and parameter references are retained under `provenance/cbmsim_reference/`.

The analyst will synthesize these completed results in
`cascade_comparison_report.md`, distinguishing measured learning/retention/reversal
from network integration and from any untested four-hour prediction.
