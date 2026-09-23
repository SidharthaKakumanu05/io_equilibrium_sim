# Homeostatic scaling replacement: running

The user requested replacing IO heterogeneity and the fixed PF gain multiplier
with homeostatic synaptic scaling, then specified four biological hours and
updates every 10%. The new run uses homogeneous IO conductances, baseline PF
gain, a 60 Hz Purkinje target, a 60-second activity average, and a four-hour
adaptation constant. It preserves the existing additive LTP/LTD rule.

The pre-change 97-test suite and new 105-test suite passed, as did disabled-rule
bitwise equivalence, five fault-injection/recovery comparisons, and the full-size
30-second pilot with automatic validation and analysis. The durable run started
on 2026-09-21 at 04:26:52 UTC; active simulation began at 04:27:10 UTC. The
pilot estimates about 9.94 wall-clock hours. The milestone watcher is running.

Details, biological assumptions, evidence, run paths and notification behavior:
[homeostasis report](homeostasis_report.md). This is an in-progress run, not a
completed four-hour result. Prior experiments below are historical.

---

# Experiment runner report

Status: EIGHT ORIGINAL LONG RUNS COMPLETE; ONE ADDITIONAL COMBINED LONG RUN LAUNCHED.

The original eight conditions each reached 18,000 recorded biological seconds (18,000,000 steps) after 8 seconds of burn-in, passed output validation and automatic analysis, and completed on 2026-09-19 at 19:49:39 UTC. The coordinator independently revalidated every committed recording/state checksum, schema, finite value, timestamp/cadence, final step, saved COMPLETE content and basic-analysis checksum. Evidence: `evidence/coordinator_resume_20260918/original_campaign_completion_audit.json`.

The user subsequently requested one additional condition combining heterogeneity and synaptic scaling. The selected condition uses IO heterogeneity CV 0.15 and PF→PKJ conductance gain 0.000274 (double baseline), with baseline gap conductance 0.0143, seed 0, the same 18,000-second horizon and 8-second burn-in, and background activity only. This provides a matched comparison with the verified baseline and both existing single-factor controls. Additive plasticity, rates, topology, timestep and all remaining scientific defaults are unchanged. There is one seed per condition, so replicated-seed uncertainty remains unavailable.

No simulation or pipeline source changed for the extension. All runnable files in the existing read-only snapshot were rechecked against the original provenance hashes. Prior cleanup, optimizer, recovery and safeguard tests remain valid and were not rerun. All eight existing datasets, manifests and completion markers are preserved.

Extension files are under `results/workflow_campaign/background_20260918T160421Z/extensions/combined_20260920T015726Z/`. The extension's long `combined_campaign.json` lists the eight completed trials plus the one new run; the tested supervisor audits and skips completed trials, preserving all nine entries in central progress. Separate `control/` and `pilot_control/` directories prevent inherited completion markers. The new worker's storage root remains the original aggregate campaign root, so its 24 GiB cap includes existing data. The 30-second combined-condition pilot completed successfully, including automatic validation and analysis, before long launch.

Preflight measured existing campaign storage at 2,329,352,519 bytes. A conservative additional allowance of 2,908,924,780 bytes for the new full-precision run, checkpoints, pilot, analysis and temporary margin gives a combined bound of 5,238,277,299 bytes, below 24 GiB. Free disk was 74,249,674,752 bytes, with a 20 GiB reserve enforced. The extension uses one CPU (core 4), one thread, and an 8 GiB address-space limit. Exact preflight/source/manifest evidence and preserved prior reports/progress are in `evidence/runner_checks/combined_20260920T015726Z/`.

The extension pilot passed all final-step, full-record/state, finite-value, timestamp, configuration and analysis-checksum checks. Its measured wall time was 89.981 seconds including 17.304 seconds of burn-in; 30 active biological seconds took 72.677 wall seconds. Peak simulator RSS was 142,664 KiB, the pilot bundle used 62,435,096 bytes, and the maximum full-precision checkpoint payload was 17,767,814 bytes. The pilot implies approximately 43,606 wall seconds (12.11 hours) for the additional long run, subject to throughput changes and the existing 48-hour cap. The updated conservative aggregate storage bound is 5,300,954,395 bytes, below 24 GiB; free disk was 74,184,073,216 bytes. Evidence: `evidence/runner_checks/combined_20260920T015726Z/pilot_gate.json`.

The coordinator accepted this gate in `evidence/coordinator_resume_20260918/combined_launch_gate_review.json`. The new durable long supervisor launched on 2026-09-20 at 02:00:18 UTC: PID/SID 1354293, PPID 1, boot ID `6a6fe94e-9e94-4ee2-a492-cea2b630b342`, start ticks `520029953`, host PID namespace `pid:[4026531836]`. It audits and skips the eight completed runs before executing exactly one new combined-condition simulation. The new worker is PID 1354582 (start ticks 520033730, session 1354293, core 4), started at 02:00:56 UTC and entered active background data collection after burn-in at 02:01:13 UTC. Full worker identities and current checkpoints are recorded automatically in central progress.json. The pilot supervisor PID 1353262 is completed, not a live long job.

The analyst remains deferred until all nine requested long runs and their automatic analyses are complete. The combined pilot is engineering evidence, not a five-hour scientific result. No source or model defaults were changed.

## Retained implementation and original launch evidence

Completed cleanup, optimization and validation were retained. Current original
source hashes matched the accepted optimization snapshot; all twelve retained
complete-state comparisons passed independent rechecking. Evidence is in
`evidence/coordinator_resume_20260918/`. Existing uncommitted source and data were
preserved. No model files in `config.py` or `sim/` were changed by this stage.

The new `experiments/run_background_campaign.py` and `experiments/pipeline/`
implement complete Simulation and recorder checkpoints, including all RNG states,
pending plasticity history, aliases, counters, selections and burn-in progress.
Immutable compressed recording generations are committed by an atomic CURRENT
cursor. Recovery preserves orphaned crash artifacts and does not truncate old
datasets. Campaign/per-run locks prevent duplicate owners. Serialized storage
checks enforce reserves before checkpoint publication. Output validation precedes
COMPLETE; deterministic analysis follows in the same worker and writes outside
raw datasets. Separate analysis-resume commands avoid rerunning simulations.

Measured validation results

- Full-size unchanged reference loop versus new uninterrupted driver: bitwise
  equality of 35 scientific arrays, four model RNG streams, aliases, all 29
  recorded arrays, and recorder state including its independent RNG.
- Five SIGKILL recovery cases passed: burn-in buffer, active buffer, after record
  write/fsync, before cursor publication, and after cursor publication. Active
  recovery checkpoints retained over 323,000 pending PF history entries. Final
  scientific state, recorder state and concatenated recordings matched exactly;
  `rtol=0`, `atol=0` throughout.
- Six expected-rejection checks passed: held live lock, disk reserve, storage cap,
  corrupted checkpoint manifest, corrupted committed recordings and changed source.
- The first harness attempt failed because the comparator did not support NumPy
  datatype class objects in recorder schemas. Its evidence is preserved as
  `evidence/runner_checks/failed_attempt_1_*`. Type-identity comparison corrected
  the harness; no scientific tolerance was changed. The complete corrected suite
  passed and its source hashes were independently rechecked.

Primary evidence: `evidence/runner_checks/recovery_latest.json`,
`evidence/runner_checks/safeguards.json`, and
`evidence/coordinator_resume_20260918/recovery_gate_review.json`.

Recording and planned campaign

`evidence/runner_checks/observable_inventory.md` inventories units, types, shapes,
cadences, selections, storage bounds and missing observables. Records include
one-second all-cell IO/PKJ/DCN counts and 40 sampled PF identities, full CF timing,
all-PKJ weight means/SD/boundary fractions, weight histograms, 400 fixed individual
trajectories, LTD/LTP counters, and three five-second voltage/event windows.
Null-event instrumentation and continuous full-time PKJ/DCN/PF rasters are absent
and explicitly labeled. CF events are modeled Ca spikes, not multispike bursts.

The prepared campaign retains eight matched-seed conditions from the saved
proposal: baseline; gap off, half and strong; heterogeneity CV 0.15 and 0.30;
and PF conductance gain half and double. Each uses seed 0, 18,000 recorded
biological seconds plus 8 seconds of burn-in, background activity only, and
unchanged baseline defaults including additive plasticity. It is a one-factor
comparison, without replicated-seed uncertainty or interaction estimates.

Frozen runnable source and exact manifests are under
`/home/sk57289/io_equilibrium_sim/results/workflow_campaign/background_20260918T160421Z/`.
The read-only source snapshot includes dirty source, hashes, binary Git diffs,
dependency/environment and hardware provenance. `campaign.json` defines long
runs; `pilot_campaign.json` defines four concurrent 30-second pilots (baseline,
strong gap, heterogeneity 0.30, double PF gain). Pilot control metadata lives in
`pilot_control/` separately from long-campaign completion markers.

Measured pilot and resource gate

Four concurrent full-network pilots each completed 8s burn-in plus30s recorded biological activity. All reached30000 final steps, passed strict validation and automatic basic analysis, and produced overview.png plus checksum-linked summary.json. Pilot supervisor PID/SID988427 ran in the host namespace with PPID1 after the launcher exited; all pilots finished before the 16:39:06UTC campaign marker. This demonstrates the detached execution path used for long jobs. Full process identities and pilot commands are retained in pilot_control/ and the individual attempt/status files.

Measured total wall times were99.34–105.62s including burn-in. Peak simulator RSS was140188–141368KiB per process (563316KiB summed); the four pilot bundles and supporting files use about251MB. Independent re-audits of full committed chains and all analysis checksums pass. `evidence/runner_checks/pilot_gate.json` contains per-condition wall/active/burn time, CF/activity rates, compressed/uncompressed bytes per recording field and checkpoint sizes. These30s measurements validate engineering; they are not long-duration scientific findings.

At the slowest measured four-worker throughput, each18000s trial projects to52466 wall seconds (14.57h), and two waves project to104932s (29.15h). These estimates are conditional on throughput remaining similar; the172800s per-trial cap is3.29times the estimate. Concurrency remains four pinned cores (4,5,16,17), one thread per worker,8GiB address-space limit per worker and32GiB aggregate ceiling. Automatic analysis runs inside the same worker slot and resource limits.

The conservative campaign storage bound is20,905,370,505bytes (19.47GiB), below the24GiB cap, without assuming compression: it includes existing pilot/provenance files, eight conditions of72checkpoints bounded at24MiB each,685,943,660uncompressed recording bytes per condition,64MiB analysis allowance each, and128MiB in-flight reserve. Measured full-precision checkpoint payloads fit comfortably inside the fixed-shape24MiB envelope. Disk free was76,793,065,472bytes; a20GiB free reserve is enforced. Retain every initial/burn/active/final generation; no pre-existing data is deleted. Crash orphans remain preserved and count toward actual storage. Per-generation locked checks stop publication if the cap/reserve would be exceeded.


## Operational contract

The additional long run ID is `heterogeneity_015_pf_strength_double_seed0_18000s`. Frozen runnable source is the original campaign `source/`; extension data, manifests and process evidence have separate paths. Exact pilot/long/resume commands are in central progress.json and extension metadata. The following launch has already executed and must not be duplicated while live:

```bash
cd /home/sk57289/io_equilibrium_sim && /usr/bin/python3 experiments/run_background_campaign.py launch --campaign /home/sk57289/io_equilibrium_sim/results/workflow_campaign/background_20260918T160421Z/extensions/combined_20260920T015726Z/combined_campaign.json
```

Only after confirming the supervisor and workers are no longer live, the same command with `--resume` recovers the validated current generation or runs analysis alone for a finished simulation. Never rerun a completed raw simulation just to regenerate analysis. Never repeat a launch while a supervisor or worker holds its live lock. The original campaign is complete and must not be rerun.

Long checkpoints commit every 300 biological seconds, approximately 12–15 wall minutes depending on measured throughput. Recording continues in bounded buffers between commits. CURRENT.json selects immutable, checksum-linked generations. Successful validation precedes COMPLETE and automatic analysis; errors, disk-reserve violations and permission failures stop the queue. All generations and crash artifacts remain retained. Population mean stability does not demonstrate individual synaptic stability. Final comparative interpretation awaits the analyst after all nine runs complete.


## Restart reconciliation and completion, 2026-09-21

The restart request found the original durable supervisor, worker, and milestone
watcher still active. The original run finished during inspection, reaching
14,400,000 active steps (four biological hours), after 8,000 burn-in steps. No
restart or duplicate simulation was launched. The raw completion marker was
published at 13:35:38 UTC and automatic analysis completed at 13:35:47 UTC.
The supervisor then published campaign completion and exited successfully; the
watcher delivered its 100% milestone and exited.

A fresh validation using the frozen source reproduced the original completion
manifest exactly except its validation timestamp. All 23 pinned source hashes,
resolved configuration, raw-file checksums, expected record counts, finite values,
timestamps, units, and analysis checksums passed. There are 14,401 weight
snapshots, 14,400 rate bins, and 15,000 trace samples. Aggregate retained campaign
storage was 3,128,810,098 bytes with 73,363,992,576 bytes free, within the existing
24 GiB cap and 20 GiB reserve. Existing data and simulation source were preserved.

Evidence: [restart reconciliation](evidence/homeostasis_20260921T033702Z/restart_reconciliation_20260921.json).
Final checkpoint: `g-finished-00008000-014400000-d380d04ed26b`.
Scientific interpretation belongs to the subsequent analyst review; completion
checks alone do not establish individual-synapse stability. Simulation resume is
not applicable to this completed run; any analysis regeneration command is saved
in the evidence and central progress record.
