# Cascade implementation handoff

Implemented on 2026-09-21 under the user's explicit authorization of Abbott and
Mauk alternatives and focused comparisons. Prior cleanup and no-op optimization
were accepted by the coordinator. No scientific campaign was launched here.

## Changes

- `config.py`: optional additive/default, binary, Abbott and Mauk selectors;
  fixed-level strengths/probabilities, matched two-level initialization and two
  independent seed overrides. No original default value changed.
- `sim/plasticity.py`: literal eight-state CbmSim d921c856 transition tables,
  strict uniform thresholds, initialized state/weight agreement, eligible versus
  state-transition versus expressed-switch counters, invariant validation and
  public resolved-event primitive. Existing ring timing and CF ownership are
  unchanged. Original additive update arithmetic moved intact into a helper.
- `sim/simulate.py`: connects options and isolated seed namespaces 701/702;
  rejects discrete plasticity with direct homeostasis. Discrete rules also reject
  nonzero weight dependence. Default additive creates no new random generator.
- `experiments/pipeline/{record,runner,validate,analyze}.py`: optional bounded
  state/switch snapshots; complete-state checkpoint validation; aligned-row,
  occupancy, weight/state and monotone-counter audits; automatic cascade summary
  including measured switch-based weight changes and range-normalized movement.
  Existing additive recording/analysis schemas stay unchanged.
- `tests/test_cascade_plasticity.py`, README: tests and public API/model contract.

The primitive is `Plasticity.apply_resolved(weights, flat_indices, directions,
uniforms=None)`. Indices must be unique valid 1D integer C-order IDs; directions
are -1/0/+1. Supplied uniforms align to every ID, including null events; they
bypass the RNG. Without supplied uniforms, only non-null events consume draws,
in supplied order, including endpoint events. Null and empty sets consume none.
The public method checks full state consistency; the network calls the same
private update kernel after deriving guaranteed unique resolved events from the
original timing ring. Batches must be chronological.

Discrete models initialize shallow 3/4 states by independent equal-probability
assignments; the additive matched option receives identical weights with the
same seed. No independently shuffled state/weight vectors. State-transition
counts include transitions that also switch expressed weight. Fixed levels are
.25/.55 and base probabilities .9/.1 unless explicitly configured otherwise.

## Focused verification

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_cascade*.py' -v`
passed 13 tests in 0.866 seconds. Log:
`evidence/cascade_implementation_20260921/focused_tests.log`.
The original `test_plasticity.py` passed all 11 tests in 0.003 seconds.
Python compilation also passed for changed source modules.

Tests cover all states/directions and exact/adjacent probability thresholds,
zero/unit base probabilities, endpoints, null/no-activity, repeated-index and
invalid-input rejection before mutation, non-contiguous weight writes, state
corruption rejection, original LTD/null timing boundaries and CF ownership,
matched initialization and RNG isolation, invalid model combinations, exact
additive arithmetic, recorder schema/counter agreement and exact pickle resume
with pending history. Scalar transition expectations do not import production
tables. One initial test-reference formula used an incorrect exponent for LTD
within low states (2**(3-state) instead of 2**(4-state)); checking literal CbmSim
cases 1/2/3 confirmed divisors 8/4/2. Corrected only that test formula; production
transitions were already correct. No tolerance was relaxed.

## Pending gates and limitations

Source ownership is released to the coordinator for independent frozen-baseline
full-state equivalence, independent transition reference, full unittest suite,
pipeline fault recovery for both cascades, source freeze, short pilots and
scientific comparisons. These gates have not run in this implementation stage.
There is no result here establishing better retention or circuit stability.

Inactive additive Plasticity objects have new configuration-only attributes
`mode`, `shape`, `low_weight`, `high_weight`, `p_ltd`, `p_ltp`; no new state arrays
or RNG. Frozen-source comparators may explicitly account for these and added
SimConfig metadata, but must compare all old scientific state and RNG exactly.

This is a transition-table port using float64 and existing simulator timing,
not CbmSim's scheduling or initialization. Native probabilities/levels are not
fitted biological measurements. Checkpoints use the existing full-object pickle
pipeline and include states, RNG, counters and pending ring; old frozen runs
remain reproducible with their own snapshots. Snapshot trajectories cannot
recover exact within-second switch times, though cumulative counts retain every
switch. Cascades and binary express two weights, so low variance alone is not
proof of memory. Direct homeostatic scaling remains unsupported for these modes.
