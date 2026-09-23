# Replacement experiment: activity-dependent synaptic scaling

The user requested removing the olivary heterogeneity and fixed synaptic-gain
perturbation from the next simulation, replacing them with homeostatic synaptic
scaling. The user accepted the original 60 Hz Purkinje target and an hours-scale
adaptation rule. The later request specifies a four-hour simulation and updates
at each 10% milestone. Existing datasets and their frozen source remain preserved.
The previous combined run completed before this request was implemented.

## Model and experiment

The replacement condition has `io_heterogeneity_cv=0`, the original
`pf_pkj_gain=0.000137`, and `homeostatic_scaling=True`. It runs 14,400 biological
seconds after an 8-second burn-in, with seed 0 and background activity only.
Gap coupling, connectivity, membrane dynamics, timestep, and additive CF-gated
LTP/LTD remain at their original settings (`weight_dependence=0`). Historical
heterogeneity and fixed-gain options remain available for reproducing old runs;
neither perturbation is enabled in the replacement run. Homeostasis is opt-in,
so the baseline remains usable as a control.

Every second, the rule updates each Purkinje cell's own firing-rate estimate:

`r <- r + (1 - exp(-1 / 60)) * (spikes_in_second / 1 second - r)`.

After the usual LTP/LTD update, all PF weights onto cell i receive one shared
factor:

`w_ij <- clip(w_ij * exp((dt_active / 14400) * (1 - r_i / 60)), 0, 1)`.

The rate sensor starts at 60 Hz and observes burn-in activity without changing
weights. Partial bins retain spike counts and the number of active-learning
steps; scaling uses only active time. Checkpoints retain the complete sensor,
partial bins, accumulated scaling and original simulator/RNG state. No new RNG
draws are introduced. The integration order is synaptic transmission, neuron
updates, CF-gated LTP/LTD, then homeostatic scaling.

The biological motivation is slow, postsynaptic activity-dependent adjustment of
excitatory synaptic efficacy. The 60 Hz target comes from this model's original
calibration. The 60-second rate average and four-hour adaptation constant are
explicit modeling assumptions, not fitted Purkinje-specific measurements.
Four hours is the log-weight adaptation constant under a unit normalized rate
error, not a promise that firing rates converge within four hours.

Primary references:
- Turrigiano et al. (1998), activity-dependent synaptic scaling:
  https://pubmed.ncbi.nlm.nih.gov/9495341/
- Renart et al. (2003), postsynaptic rate-targeted conductance scaling model:
  https://www.sciencedirect.com/science/article/pii/S0896627303002551

Multiplicative scaling preserves within-cell weight ratios only away from the
bounds and in the absence of intervening LTP/LTD. Clipping is retained and
counted explicitly. A zero weight cannot be revived by multiplication alone,
although additive LTP can increase it. Rate homeostasis does not guarantee
stationary individual synaptic weights, nor does it reproduce every molecular
mechanism of biological synaptic scaling.

## Recording and validation

The pipeline records per-cell sensed rate (Hz), cumulative attempted log factor
(dimensionless), cumulative homeostatic weight change (sum over incoming
synapses), and the number of clipped synapse updates at the existing one-second
cadence. These supplement original individual weight trajectories, distributions,
neural rates/events and LTP/LTD counters. The validation checks field alignment,
finite values, update counts, partial bins, clipping monotonicity and agreement
between final sensor state and recorded values. Automatic analysis reports rate
error against target and homeostatic contributions separately from LTP/LTD.

Evidence directory: `docs/agent_reports/evidence/homeostasis_20260921T033702Z/`.
It contains the exact pre-change source, prior dirty-tree diff, baseline and
post-change test logs, disabled-rule equivalence checks, and fault-injection
checkpoint recovery checks. Unit tests cover feedback direction, per-cell
independence, weight-ratio preservation, slow adaptation, burn-in, clipping,
zero weights, partial-bin recovery, and error reduction in an independent test
plant. The test plant accelerates adaptation only for testing; the experiment
uses the four-hour constant.

Status: all eight focused homeostasis tests, disabled-rule bitwise equivalence,
and five full-state fault-injection/recovery comparisons passed. The original 97-test baseline suite passed in 496.827 seconds; the new 105-test
suite passed in 494.677 seconds. The pilot and launch gates subsequently passed; see below.
No replacement four-hour result is claimed until its validated completion and
automatic analysis markers exist.

## Launch and milestone updates

The 30-second full-network pilot passed validation and automatic analysis. It
took 92.084 wall seconds including 17.553 seconds of burn-in, used 142,088 KiB
peak RSS, and stored 65,945,840 bytes. Its active throughput projects 35,775
wall seconds (about 9.94 hours) for the requested four biological hours. The
conservative aggregate storage bound is 7,014,078,026 bytes, below the 24 GiB
cap, with a 20 GiB free-space reserve. Runtime remains an estimate.

The durable supervisor launched at 2026-09-21 04:26:52 UTC (PID 1399211).
Worker PID 1399212 completed burn-in and entered active simulation at
04:27:10 UTC. Both use the host process namespace, with the worker limited to
one thread on CPU 4. The old completed runs remain archived and untouched.

Run: `results/workflow_campaign/homeostasis_20260921T034911Z/runs/homeostatic_scaling_seed0_14400s`.
Manifest: `results/workflow_campaign/homeostasis_20260921T034911Z/campaign.json`.
Full resolved config: `specs/homeostatic_scaling_seed0_14400s.json` under that campaign.

Checkpoints occur every 240 biological seconds so each 10% threshold lands
exactly on a durable checkpoint (1,440 seconds, or 24 simulated minutes, per
10%). The independent watcher (PID 1399351, started 04:27:15 UTC) checks
checkpoint manifests every 30 wall seconds and queues a concise update to the
existing Codex conversation. A queue self-test was accepted, and seven milestone
boundary/deduplication checks passed. The watcher records delivery receipts in
`progress_notifications/status.json`. It alerts on stopped/failed runs, and
queues 100% only after both completion and automatic-analysis checksums pass.
The watcher is independent of model dynamics and cannot modify the simulation.

The full run is in progress; no four-hour scientific outcome is claimed yet.


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
