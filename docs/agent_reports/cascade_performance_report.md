# Cascade preparation performance gate

Completed 2026-09-21. **Accepted no-op:** no scientific source, configuration defaults, equations, timestep, precision, event ordering, connectivity, or RNG behavior was changed. The user requested implementation and comparison of cascade alternatives; this preparation gate does not justify unrelated optimization. Source ownership is released to the coordinator/experiment_runner.

## Provenance and current validation

Current Git HEAD is `8437bde0f6828b045caecc698e258c4430b94492`; the exact dirty source identity is `b8fb0d8aa4815dfe5cc7d28f58a85fd31749cfea97aa7397620e070641d0ac38`. All 54 files match the independently accepted cleanup source before and after this stage. A runnable 493,311-byte source copy and per-file hashes are retained under `evidence/cascade_optimizer_20260921/source/` and `source_manifest.json`.

The coordinator accepted both current 105-test full suites: exit 0, 487.999 s baseline and 495.599 s after cleanup. Evidence: `evidence/cascade_cleanup_20260921/{baseline_unittest.json,post_cleanup_unittest.json,final_verification.json}`. These exact-source results are reused; no third suite was run. This unchanged-source validation gate does not validate the forthcoming cascades: their transition, default-model equivalence, checkpoint recovery and full-suite gates remain mandatory.

Historical `performance_report.md` was reviewed and preserved. Its optimizer results concern an earlier source identity and are not new measurements or a speedup claim for this task. Historical raw test-state dumps were removed by authorized cleanup; their regeneration is unnecessary for a source-identical no-op.

## One bounded current profile

The single full-network sample used seed 0, 40 IO / 320 PKJ / 80 DCN cells, 160,000 PF synapses, 1 ms timestep, 1 second burn-in and 2 seconds active background. Other dynamics were the original defaults, including additive plasticity and disabled homeostasis. `Simulation.run` used its default in-memory recording (10 ms slow traces and its configured final voltage window, limited to the 2-second active interval). Full resolved configuration, seeds, versions, CPU and environment are in `profile/measurement.json`.

Host Eccles is the authorized standalone compute host. Python 3.10.12 / NumPy 2.2.6 ran pinned to CPU 4 with OMP/OpenBLAS/MKL threads each set to 1; available disk was 74.8 GB, above the 20 GiB reserve. No package installation or heavy background job was needed.

Measured profiled wall time: **7.033 s for 3 biological seconds / 3,000 steps**, or **426.55 steps/s** (2.344 wall seconds per biological second). Timing excludes model construction and metadata serialization, includes burn-in and recording, and includes profiler overhead. Peak process RSS through the run was **63,432 KiB**. In-memory NumPy log payloads totaled **10,601,584 bytes** (excludes Python container overhead and may include shared arrays). No simulation bundle or full-state binary dump was written. All 2,000 active steps completed and weights were finite.

Cumulative profile costs were IO integration **4.037 s**, PF generation **1.678 s**, and plasticity **0.414 s**. IO integration remains the measured bottleneck; substep costs are included in its cumulative total. The full compact text profile is `profile/profile.txt`. These are a single profiled sample, not repeated benchmark timings, a variance estimate, a comparison to another model, or a four-hour runtime prediction. No optimization speedup is claimed.

## Recording scope and handoff

Reviewed `sim/simulate.py`, `sim/long_run.py`, `sim/recording.py`, and the diagnostic driver. Their recording contracts differ: in-memory float64 arrays and spike trains; streaming integer event files plus float32 slow/final arrays; and diagnostic binned counts/selected traces, respectively. This profile measures only the in-memory path. Recording-on/off overhead was not remeasured because no recording or scientific implementation changed; no new overhead claim is made. Runner pilots must measure their actual durable pipeline.

`evidence/cascade_optimizer_20260921/final_verification.json` records the accepted no-op and exact source recheck. No blockers or live jobs remain. Proceed to implementing the authorized cascade models and the focused comparison plan; this stage launched neither comparison nor long trial.

To reproduce the bounded profile from the frozen source, choose a fresh output directory (existing directories are rejected):

```bash
cd /home/sk57289/io_equilibrium_sim && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 docs/agent_reports/evidence/cascade_optimizer_20260921/profile_baseline.py --out /tmp/cascade_baseline_profile_recheck
```
