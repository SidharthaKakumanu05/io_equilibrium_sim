# Abbott and Mauk cascade comparison

Authorized on 2026-09-21: clean the project, implement both alternatives, and
compare them. The user selected a focused comparison before any new four-hour
network campaigns. The user separately approved deletion of the old test dumps
listed in `agent_reports/evidence/cascade_cleanup_20260921/bulky_candidates.json`,
while retaining scientific runs, reports, summaries, snapshots, and scripts.

## Model contract

The additive model remains the default. Add optional Abbott and Mauk cascade
rules using the transition kernels in the local CbmSim checkout at
`d921c8561597657bfa595f651dc4208fbe165c42`, `src/cbm_core/kernels.cu`.
Include a two-state binary control to distinguish hidden-state retention from
the effects of using only two expressed strengths.

Each cascade has eight states: 0–3 express the low weight and 4–7 the high
weight. Preserve CbmSim's transition probabilities and strict `uniform < p`
thresholds. Abbott opposing events may switch directly to the shallow opposite
state. Mauk transitions move one state at a time. The initial weights must be
derived from the actual initialized states, never shuffled independently.

For states ordered 0 through 7, the independently read kernel transitions are:

| Direction/model | Next state on a successful transition | Probability multiplier |
| --- | --- | --- |
| LTD, Abbott | 0, 0, 1, 2, 3, 3, 3, 3 | 0, 1/8, 1/4, 1/2, 1, 1/2, 1/4, 1/8 |
| LTP, Abbott | 4, 4, 4, 4, 5, 6, 7, 7 | 1/8, 1/4, 1/2, 1, 1/2, 1/4, 1/8, 0 |
| LTD, Mauk | 0, 0, 1, 2, 3, 4, 5, 6 | same LTD multipliers |
| LTP, Mauk | 1, 2, 3, 4, 5, 6, 7, 7 | same LTP multipliers |

Multiply each entry by the corresponding base LTD or LTP probability. Endpoint
self-transitions do not count as state changes. Count eligible events, hidden
state transitions, and expressed-weight switches separately. A binary control
uses only states 3 and 4, with the unattenuated directional probability.

Preserve this simulator's existing PF eligibility, CF ownership, LTD/null/LTP
windows, resolution delay, timestep, neurons, connectivity, and synaptic gains.
This ports the cascade transition rules, not CbmSim's entire scheduling scheme.
Use a separate seeded plasticity RNG; leave all existing random streams and
default additive event results bitwise unchanged. Checkpoints must preserve
hidden states, transition RNG, switches, and pending eligibility.

Do not combine the existing direct multiplicative homeostatic weight updates
with fixed-level cascade assignments. For this comparison, homeostasis is off
in all conditions. Reject an unsupported combined configuration explicitly;
define a separate effective-gain model before supporting that combination.

The native CbmSim parameters are low/high weights 0.25/0.55 and base
LTD/LTP transition probabilities 0.9/0.1. These are reference parameters, not
Purkinje-specific fitted measurements. Do not tune circuit rates to an expected
answer.

## Validation gates

1. Complete the authorized cleanup and required before/after full tests.
2. Review the existing optimizer evidence, obtain a bounded profile, and make
   no optimization change unless a measured need justifies one.
3. Exhaustively test all eight states, both update directions, probability
   boundaries, endpoints, null events, and eligibility timing against an
   independent literal interpretation of the CbmSim kernels.
4. Verify disabled-cascade/additive full scientific state and event outputs
   against the frozen pre-change source exactly. Added inactive configuration
   metadata may be excluded explicitly; scientific arrays and RNGs may not.
5. Test checkpoint interruption/recovery for both rules, including nonempty
   delayed PF history and nontrivial hidden states. Compare recordings,
   scientific state, and RNGs exactly; run the full unittest suite.

## Focused experiments

Use identical prescribed input realizations across isolated rule comparisons,
with at least three seeds and hundreds of fixed identified synapses per seed.
Measure acquisition of a two-group weight pattern, retention under unbiased
background learning, reversal, and retention of the reversed pattern. Report
both absolute pattern contrast and retention normalized to the contrast actually
acquired; undefined normalization after failed acquisition must be labeled.
Retention should cover at least 600 biological seconds in each direction.

Include the original additive rule, an additive control with the same 0.25–0.55
weight range, binary, Abbott, and Mauk. For binary and cascades, supplement the
native probabilities with a transparently derived update-size control:
`p_LTP = 0.001 / (0.55 - 0.25)` and
`p_LTD = 0.009 / (0.55 - 0.25)`.
This matches expected changes for eligible switches at the shallow crossover;
it does not match every hidden state or fit a firing rate. State this limitation.

An accelerated isolated assay may apply the production learning-event primitive
directly if its event distribution, omitted shared-CF timing correlations, and
units are explicit. It must not be mislabeled as a full-network simulation.
Validate any batched event reordering against a chronological reference.

Resolved assay protocol (declared before collecting comparison data): seeds
0, 1, 2; 512 synapses with common shallow low/high initial assignments; 120 s
background, 120 s learning, 600 s retention, 120 s reversal, 600 s reversed
retention. Per-second PF counts are Binomial(1000, 0.02), the exact marginal
of independent 1 ms Bernoulli spikes at 20 Hz. Each event independently receives
an LTD verdict with probability 0.1 during background/retention. During learning
the two equal groups use 0.02/0.18; reversal swaps these probabilities. The
population-average verdict probability remains 0.1. Uniforms are attached to
events and shared across all variants, which run the production update primitive.
Record all 512 weights/states and cumulative switches each second. This protocol
intentionally omits shared-CF verdict correlations and circuit feedback.

Follow with matched, background-only full-network checks of additive, binary,
Abbott, and Mauk: two seeds, initially 120 biological seconds per condition,
8-second burn-in, unchanged 1 ms timestep and original full network. Use common
initial weight assignments where supported, or document the initialization
difference and avoid attributing startup differences solely to the rule.
The 120-second checks assess integration, firing rates, switches, and operating
point; they do not establish four-hour network stability. Revise the bounded
duration only for an evidenced reason recorded before launch.

Use sequential one-thread pilots on standalone Eccles, existing memory/disk
ceilings and 20 GiB free reserve. Pilot each rule, freeze the tested source,
and run through the validated durable pipeline with automatic analysis.
For the eight independent network checks, permit up to four single-thread
workers after the measured pilots establish resource headroom. This stays within
the previously delegated four-CPU/four-job, 32 GiB total RAM ceiling and avoids
approximately 40 minutes of serial integration checks. Record the pilot-based
projection and final concurrency before launch; scientific settings remain the
same. Project agents and source-edit ownership remain sequential.

## Decision criteria

Report per-seed acquisition and reversal, pattern contrast and retention curves,
switch/dwell statistics, hidden-state occupancy, individual lagged displacement,
and weight-range-normalized changes. In network runs also report PKJ/CF/DCN/PF
activity and distribution/mean behavior. A narrower weight distribution, a
stable population mean, or a synapse that cannot learn is not evidence of a
successful drift remedy. Favor no variant unless the measured comparison
supports it; preserve negative results and separate mechanistic inference from
measurements.
