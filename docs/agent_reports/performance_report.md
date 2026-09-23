# Optimizer report

Status: COMPLETE; awaiting independent coordinator validation. Only `sim/plasticity.py` changed during optimization. No long experiments were launched and no existing data was deleted or modified.

## Accepted change and scientific constraints

Sparse PF events are located once with `np.flatnonzero`, converted to row/column coordinates with integer `divmod`, and classified by the existing per-PKJ CF verdict. This avoids repeated scans of full-size LTD/LTP masks. Tuple indexing writes through non-contiguous weight views. Event coordinates remain in row-major order; LTD precedes LTP; the soft-bound fractions are still computed before either update; full-array clipping remains after updates, including clipping inactive synapses. No ring-buffer representation, equation, timestep, precision, connectivity, RNG draw, scientific default, or rate was changed. `weight_dependence=0.0` remains the default.

The first candidate used two-dimensional `np.nonzero`. It passed a complete-state comparison but was slower (10.304–10.546 s versus baseline 9.902–9.935 s), so it was rejected. Its source is retained as `candidate1_plasticity.py`. The accepted flat-index candidate initially measured 9.076–9.106 s, but the final, controlled recorded-driver measurements below are the performance claim.

## Baseline and reproducibility

Read the full Claude handoff, workflow, cleaner report and acceptance evidence. Historical diagnosis and historical test claims are not presented as new findings. The exact post-cleanup dirty source was copied to `docs/agent_reports/evidence/optimizer_checks/baseline_source/`, after every relevant source hash matched cleaner acceptance `9b5780224cd6da97ce773c88f498f15b02e6c34a97db23210858388d3b527472`. Git revision is `8437bde0f6828b045caecc698e258c4430b94492`; HEAD alone does not identify this dirty source. `baseline_manifest.json`, `git_status.txt`, and staged/unstaged patches preserve provenance. `optimization.patch` shows only this stage’s change, separate from pre-existing edits.

Evidence paths below are relative to `docs/agent_reports/evidence/optimizer_checks/`. The accepted source identity is `8e4102bd88e40dd077d0bbf56a3662cff898250a978f93f534d88738c3d2a993` (SHA-256 of sorted path/hash JSON with compact separators). `candidate_manifest.json` was captured while validation ran; `final_verification.json` verifies the identical files after all checks. Exactly `sim/plasticity.py` differs from the post-cleanup baseline.

Host: Eccles, AMD Ryzen Threadripper 2990WX, Linux x86-64, Python 3.10.12, NumPy 2.2.6. All benchmark/test children used `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`; no dependencies were installed. Final paired timing children were pinned before initialization to CPU 4, physical core 4, NUMA node 0. They ran sequentially, alternating source versions and rotating recording modes. Hardware details are in `baseline_manifest.json`; exact commands and process identities are in `batch_status.json` and `paired_status.json`.

Timing configuration: full default network (40 IO, 320 PKJ, 80 DCN, 160,000 PF synapses), seed 0, dt 1 ms, IO substep 0.1 ms, 1 s burn-in plus 3 s background activity. Timing includes burn-in and simulation/recording finalization, but excludes model construction and evidence serialization. The shorter timing burn-in is not the campaign burn-in; exact validation separately uses the actual 8 s burn-in plus 5 s activity. Every measurement includes the resolved configuration. Recording uses 40 PF identities, 15 tracked weights, 1 s slow cadence, and one final 0.1 s voltage window.

## Profile and measured performance

Before edits, `baseline_profile_valid/profile.txt` attributed 5.510 s to IO integration, 3.108 s to PF generation, and 1.283 s to plasticity in an 11.136 s profiled run. After the accepted change, `optimized_profile/profile.txt` measured plasticity at 0.643 s in a 9.364 s profiled run. Profiled absolute runtimes are diagnostic, not the speedup estimate. IO integration remains the dominant cost. The first profile attempt completed simulation but failed writing metadata because the harness used `.rng` instead of `._rng`; that attempt is preserved in `baseline_profile/`, was corrected and rerun, and is not used as valid benchmark metadata.

Final CPU-pinned results (three independent repetitions per cell):

| Driver | Baseline median, range (s) | Optimized median, range (s) | Median speedup | Peak simulation RSS, baseline / optimized (KiB) |
| --- | --- | --- | --- | --- |
| Streaming `run_long` | 9.5735, 9.5479–9.5997 | 8.9944, 8.9233–8.9977 | **1.0644x** | 55,048–55,052 / 55,044–55,052 |
| In-memory `Simulation.run` | 9.4236, 9.4191–9.5087 | 8.8620, 8.8114–8.9085 | **1.0634x** | 56,436–56,792 / 55,936–56,336 |
| Recording disabled, direct `_step` loop | 10.5043, 10.4294–10.5127 | 8.9602, 8.8587–11.7640 | Not the headline estimate | 54,480–54,488 / 54,492–54,524 |

Streaming wall time fell **6.05%** (a **6.44% throughput increase**). Streaming sample standard deviations were 0.0259 s baseline and 0.0420 s optimized; in-memory values were 0.0505 s and 0.0485 s. RSS is the process high-water mark through construction and run, sampled before final evidence serialization. The change does not materially reduce model memory. Raw repeats, standard deviations, byte counts and hashes are in `final_verification.json` and `paired_{reference,optimized}_{off,memory,stream}_{0,1,2}/measurement.json`.

These are bounded benchmarks, not a measured five-hour campaign runtime. Event density, allocation patterns, NUMA contention and later recording/checkpoint work can change throughput. The runner must use its own pilot for scheduling and storage predictions.

## Recording cost and outputs

All recording modes use identical selections and RNG consumption; complete scientific state is bitwise equal with recording enabled and disabled. In cross-mode comparisons only `sim.log`, the recording-output container, is excluded; all model fields, buffers, weights, selection arrays, RNG streams and aliases remain compared.

The two existing recorders have different contracts: `Simulation.run` retains float64 slow/voltage arrays and spike trains in memory; `run_long` writes buffered int32 step/int16 cell event streams, float32 slow/voltage/final arrays, and compressed NPZ files. Both were deliberately configured with matching cadence/window for this test, rather than comparing their unequal defaults. Conversion to float32 is pre-existing recording behavior, not a new numerical approximation to model state. Final pickle evidence preserves float64 model state.

End-to-end optimized median differences versus the direct recording-off loop were +0.0341 s (+0.38%) for streaming and -0.0982 s (-1.10%) for in-memory recording. Baseline differences were -0.9308 s and -1.0807 s respectively. **These are net differences between separate driver/allocation paths, not clean estimates of pure recording overhead.** The negative values and the 11.764 s optimized-off outlier prevent a defensible claim of negative cost or precise total overhead from these short runs.

A separate deterministic replay isolates the recorder component: it replays the already-validated exact 3 s events with no simulation/RNG advancement, on CPU 4. Median empty-loop time was 0.000980 s; memory recorder callbacks plus final train conversion took 0.042149 s; streaming callbacks plus flush/close took 0.035285 s. Subtracting the empty loop gives approximately **0.0412 s memory** and **0.0343 s streaming** component cost. Recorder construction, slow summaries, voltage collection and compressed NPZ writing are excluded, so these are not total recording-overhead estimates. All replayed streaming files match the actual simulation’s bytes. See `recorder_replay.py`, `.json`, and `.log`.

The 3 s streaming bundle uses approximately 782,042–782,044 bytes (JSON runtime text changes length slightly), including 404,160 bytes of event streams. In-memory log NumPy payloads sum to 2,264,144 bytes; this excludes Python container overhead. Recorder-disabled runs write no simulation bundle. Full state pickles and validation evidence are separate from simulated output-size measurements. `recording_audit.json` verifies every binary stream byte, every NPZ array’s dtype/shape/bytes, and all metadata except measured `wall_s` across source versions. Raw data has not been deleted to reduce storage.

## Exact correctness evidence

All comparisons use **rtol=0, atol=0**, including bitwise float equality; no approximate change or tolerance relaxation was adopted. `compare.py` traverses the entire saved object graph, compares every array’s dtype/shape/bytes, scalar values/bits, connectivity, neuron/channel/gating/calcium/synaptic state, plasticity rings/counters/pointers, and all four RNG states. Alias relationships are checked, including the PKJ/DCN shared LIF RNG. Each scientific comparison additionally hashes every timestep’s complete PF/PKJ/DCN/CF outputs, every individual weight, and LTD/LTP counters, preserving event order and complete weight trajectories.

Eight full-network baseline-versus-optimized comparisons pass:

- Baseline, seed 0: actual 8 s burn-in followed by 5 s activity.
- Baseline, seed 1: 1 s burn-in followed by 3 s activity.
- Gap off; gap conductance 0.04; IO heterogeneity CV 0.3; doubled PF conductance gain; 200 ms null window; and existing `weight_dependence=1.0` diagnostic path, each seed 0 with 1 s burn-in plus 3 s activity.

These cover representative perturbations; they are not scientific trials or claims that soft bounds are an authorized campaign condition. The default additive model is unchanged. `exact_*.json` contains twelve passing whole-state comparisons, including matched memory/streaming outputs and both recording-on/off comparisons. `plasticity_edges.json` adds 36 differential cases and 5,400 compared steps: zero/default/dense PF activity, contiguous and strided views, null windows, wd=0/0.5/1, boundary decisions and clipping of inactive out-of-range weights.

Full suite: `python3 -m unittest discover -s tests -q`, **97 tests passed**, exit 0, unittest 489.520 s, wrapper 489.618 s. Logs: `full_unittest.log`, `batch_status.json`. The source hashes at completion match the frozen candidate. The cleaner’s 97-test baseline/post-cleanup results remain the pre-optimization test evidence; no historical handoff pass was substituted.

## Handoff and commands

Source ownership is released to the coordinator validation gate. No optimizer blockers remain. The runner still must implement full-state atomic recovery, observables/storage accounting, its own validated pilot and durable analysis before long launch. This optimization does not add checkpointing or make the old `run_long` resumable.

Read-only exact recheck of retained baseline/default-burn-in artifacts (writes a new comparison result):

```bash
cd /home/sk57289/io_equilibrium_sim && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 docs/agent_reports/evidence/optimizer_checks/compare.py docs/agent_reports/evidence/optimizer_checks/reference_verify_baseline_seed0 docs/agent_reports/evidence/optimizer_checks/optimized_verify_baseline_seed0 --out /tmp/io_optimizer_baseline_recheck.json
```

Full-suite command if subsequent source edits require retesting:

```bash
cd /home/sk57289/io_equilibrium_sim && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 -m unittest discover -s tests -q
```

Exact benchmark commands for every retained run are in `batch_status.json` / `paired_status.json`. Their output directories intentionally reject reuse; preserve existing evidence and choose a fresh output path for reruns. Coordinator acceptance should inspect `final_verification.json`, `optimization.patch`, the reports and current file hashes before advancing.
