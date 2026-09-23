# Exact simulation acceleration — 2026-09-21

The four primary full-network modes now run slightly faster than biological time
in isolated, recorded, fixed-seed benchmarks. The original additive baseline was
0.4345 biological seconds per wall second; the optimized primary modes measured
1.073–1.081 with recording. This is approximately 2.5× the original throughput.
The subsequent four-worker production-cadence pilots achieved **1.057–1.079×
real time**, each covering 600 active biological seconds and including burn-in,
recording, checkpoints, validation and analysis. Separate CPU caches and local
NUMA memory placement were necessary. See
[the campaign report](realtime_experiment_report.md) for all measurements and
the running eight-trial campaign. Full four-hour sustained rates remain pending.

Validation status: **all 125 tests pass in 63.227 seconds**, including seven
focused native equivalence tests. Coordinator checkpoint recovery also passes
ten fault comparisons. All eight independent frozen-source comparisons pass
exactly for full scientific state, RNGs, events and sampled recordings. No long runs launched by this optimizer.

## Scope and preserved science

The user authorized significant optimization toward one biological hour per real
hour before eight four-hour biological runs: additive, binary, Abbott cascade,
and Mauk cascade, seeds 0 and 1. The baseline was the immutable source at
`results/workflow_campaign/cascade_20260921T145424Z/source`; the coordinator
verified current sources matched it before this stage. The 118-test prerequisite
suite passed in 497.367 seconds. Cleanup was a no-op for this new request.

No equations, precision, timesteps/substeps, rates, connectivity, eligibility
windows, default plasticity parameters, or random streams were changed. Previous
uncommitted scientific changes remain intact and are not attributed to this work.
No scientific data or old results were deleted. Temporary test states are not
persisted as bulky raw dumps.

## Measured changes

1. **IO integration** was the baseline bottleneck. The compiled backend fuses
   elementwise float64 arithmetic with contraction and fast math disabled, while
   calling the actual installed NumPy `exp` ufunc on population-length vectors
   and NumPy matrix multiplication for gap coupling. This preserves arithmetic
   operation order, exponential implementation, and reduction order. IO noise
   generation still occurs at the original call site and consumes the same block.
   The Python `IOPopulation.substep` remains the executable reference.
2. **PF generation** became the dominant bottleneck after IO fusion. An initial
   generic BitGenerator `next_double`-to-bool loop produced no meaningful gain
   and was removed. The retained PCG64 specialization uses public state integers
   and four interleaved exact integer recurrences, restoring original draw order
   before thresholding. It consumes exactly the original number of uniforms and
   preserves `has_uint32`/`uinteger`; the generator lock spans state read/update.
   Other generator types use the original NumPy path. This is the same generator,
   not a substitute statistical process. Formula reference:
   [NumPy 2.2.6 PCG64 implementation](https://github.com/numpy/numpy/blob/v2.2.6/numpy/random/src/pcg64/pcg64.h).
3. **Resolved cascade updates** remained sufficiently expensive to put the
   recorded cascades just below real time. The final fusion performs the same
   state/probability/switch indexing in C after existing Python eligibility and
   transition RNG draws. Event order, strict probability comparison, endpoint
   draws, null handling, expressed weight levels, and all counters remain exact.
   The Python implementation is retained as the fallback/reference. Noncontiguous
   weight views retain tuple-index-equivalent writes.

The build is `python3 -m sim.build_native`. No dependencies were downloaded.
Compiler: GCC 11.4.0; flags include `-O3 -std=c99 -ffp-contract=off
-fno-fast-math -fno-associative-math`. Python 3.10.12, NumPy 2.2.6, standalone
Eccles, CPU affinity 4, one BLAS/OpenMP thread. No extra simulation thread was
introduced. Source, binary SHA256, compiler command/version and dependency
identity are in `sim/native_build.json` and campaign provenance. An absent build
uses Python; a stale/incomplete/incompatible build errors. Both runner and
analyzer explicitly verify native backend identity. Frozen snapshots must include
C source, build helper, loader, binary and build manifest.

## Repeated timings

Each final row has two six-biological-second runs at seed 0, matched two-level
initialization, full network, original dt, no burn-in. Recording is the original
in-memory `Simulation.run` recorder. Off runs call the same `_step` dynamics.
The dedicated recording RNG makes recording observational. Within each mode all
four recordings-on/off final weight hashes were identical.

| Mode | Biological s / wall s, recording off | Biological s / wall s, recording on | Recording-on wall seconds (6 biological seconds) |
|---|---:|---:|---|
| Additive | 1.1551 | 1.0767 | 5.5929, 5.5527 |
| Binary | 1.1573 | 1.0736 | 5.5815, 5.5957 |
| Abbott cascade | 1.1627 | 1.0728 | 5.5565, 5.6294 |
| Mauk cascade | 1.1667 | 1.0815 | 5.5291, 5.5663 |

Recording cost was approximately 7–8% of runtime. Peak process RSS was at most
74,376 KiB in the final multi-case benchmark (a process high-water mark, not
independent per-case RSS). No scientific bundle bytes were written by these
in-memory timing runs; only compact logs/JSON profiles were saved. Durable file
output, compression, checkpoint and recovery overhead require the pipeline pilot.

The original additive benchmark used three biological seconds per repetition,
uniform initial weights, and produced recording-on wall times 6.9286, 6.8860,
6.8967 seconds. IO fusion alone reached 0.8525×; generic PF fusion 0.8552×
(rejected); exact interleaved PCG64 reached 1.0972× on that original configuration.
The final matched-init cascade fusion increased Abbott from 0.9647× to 1.0728×,
and Mauk from 0.9695× to 1.0815×. Small differences between successive additive
benchmarks are normal timing variability/configuration differences, not an
additional additive optimization claim.

## Exact validation and evidence

All acceptance comparisons use bitwise array/scalar equality, exact spike/event
vectors, complete RNG states, and alias preservation. No tolerances were relaxed.
The focused tests cover all four network modes, noisy IO populations of sizes
1/7/40/83, heterogeneous conductances and gap coupling, singular gating voltages,
voltage clamps, gap-off, parameter swaps, external array-reference semantics,
random and boundary 128-bit PCG states, uint32 cache preservation, zero/odd/full
network draw shapes, probabilities including endpoints, alternate generators,
and absent/corrupt backend manifests. Existing cascade tests independently check
the transition rules and probability edges.

Evidence is in `docs/agent_reports/evidence/realtime_optimization_20260921/`:

- `baseline.json`, `baseline.profile.txt`, `benchmark.py`: fresh baseline and
  repeatable benchmark against the immutable prior source.
- `native_io.json`, `native_io_pf.json`, `native_io_pcg.json` and profiles:
  successive bottleneck measurements, including the rejected PF experiment.
- `cascade_before_fusion.profile.txt`: 0.585/3.347 seconds in resolved cascade
  updates before fusion, 0.998/3.347 seconds in total plasticity.
- `all_modes_before_discrete.json`, `all_modes_benchmark.json`, logs and
  `benchmark_modes.py`: two independent timings per recording/mode combination.
- `native_tests.log`: seven focused tests pass in 6.596 seconds (subsequent full
  suite includes scalar-return type preservation as well).
- `validation_source.json`: exact source, native binary/build and test hashes.
- `full_suite.log`: **125 tests passed in 63.227 seconds, exit 0** on the final source.
- `final_confirmation.json`: final-code recorded timing and measured log-array
  payload, with complete resolved configurations and current provenance.
- `recovery.json`: coordinator gate, ten exact fault/resume comparisons; raw
  temporary state dumps were removed after passing.

A final scalar PF return-type preservation adjustment after the six-second
benchmarks affects only zero-dimensional/None shapes; the network always uses a
2D weight shape. Full-suite validation uses the final source. The coordinator's
independent preoptimization-versus-current comparisons and fault recovery are
separate gates before campaign acceptance.


## Handoff

Simulation/source edits are released to the coordinator. The optimization's
source-hash tree identity is
`961f5fc8ed633c0bcbb0770e5280cd4c4d93172a531cf411678514b1795d4a5d`
(canonical JSON of `source_hashes`, SHA256). Native source SHA256 is
`7f749bbf754755d20db2e18dce8ed9287a027348f09a2ab069c4a09b1c084dbd`;
binary SHA256 is
`967c6f0f119f0987ab86f54def1b72d16806d4bb0e66b670c2704dd44929f779`.
No runtime scratch buffers were added to checkpoint state. Complete neuron,
plasticity and RNG state remain owned by the same Python objects.

Repeat the suite:

```bash
cd /home/sk57289/io_equilibrium_sim
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -q
```

Repeat final-mode timing:

```bash
cd /home/sk57289/io_equilibrium_sim
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 docs/agent_reports/evidence/realtime_optimization_20260921/benchmark_modes.py
```

The coordinator subsequently accepted four-worker pipeline throughput, storage,
CPU scheduling and immutable-source gates and launched the user-selected eight
four-hour runs. Four original preoptimization 120-second datasets also matched
the optimized pilots exactly in complete state and every recorded array; see
`evidence/realtime_optimization_20260921/cpu_placement_equivalence.json`.
Pilot performance does not establish a four-hour speed guarantee.
Final-code three-second recorded confirmations took 2.760–2.803 wall seconds
(1.070–1.087×); NumPy payload in each in-memory log was 15.227–15.232 MB.
These additional measurements ran while the coordinator could be performing
independent checks, so the isolated repeated timings above remain the primary
speed measurements.
