# Focused Abbott–Mauk comparison

Completed 2026-09-21. **Both cascade alternatives are implemented and validated, but neither native variant fixes individual-synapse drift or preserves the learned pattern in these tests.** Mauk retains the pattern longer than Abbott at the same probabilities. Slowing transitions improves Mauk's early retention, with weaker acquisition and reversal; it still loses most contrast over 600 seconds. These results do not justify changing the additive default.

The completed four-hour homeostatic run is preserved as the user's chosen historical result; it was not repeated. Its stable population mean did not imply stable individual weights. The present campaign tests a different mechanism with **homeostatic scaling off**. Direct scaling and fixed-level cascades are currently rejected in combination.

## Scope and acceptance

Campaign: `results/workflow_campaign/cascade_20260921T145424Z`. All 24 isolated conditions (eight variants × seeds 0/1/2), four 10-second pilots and eight 120-second network runs (four modes × seeds 0/1) completed. The network remains 40 IO / 320 PKJ / 80 DCN / 160,000 PF synapses, 1 ms timestep, 8 s burn-in, background activity and unchanged input/rate parameters. The short network runs assess integration and operating point, not four-hour stability or network conditioning.

The [cleanup report](cascade_cleanup_report.md) documents the user-approved deletion of 1.332 GiB of old validation dumps and bytecode, preserving scientific runs and source snapshots. The [performance gate](cascade_performance_report.md) made no optimization change. The [implementation report](cascade_implementation_report.md) documents optional binary, Abbott and Mauk modes, hidden states, independent initialization/transition RNGs, and checkpoint/recording support. The [execution report](cascade_experiment_report.md) documents all campaign gates.

Validation evidence in `evidence/cascade_validation_20260921/`: 118 tests passed; three old/new additive configurations match scientific state, events and RNG exactly; eight SIGKILL recoveries preserve hidden state, pending PF history and every recorded array; the assay batching matches chronological processing exactly. `completion_audit.json` and `assay_audit.json` verify completion, finite data, cadence, source/configuration provenance and checksums. Cross-mode network PF samples/RNG, selected identities and initial weights are paired within seed. The independent analysis here rechecked all checkpoint payload hashes, timestamps and finite arrays, all 24 assay hashes, and reproduced every acquired/reversed contrast and half-crossing. No failed or partial trial enters this report.

## Isolated learning, retention and reversal

Each seed has 512 synapses split into two groups. The protocol is 120 s background, 120 s learning, 600 s retention, 120 s reversed learning, then 600 s reversed retention. PF counts have the per-second marginal of 1 ms Bernoulli activity at 20 Hz. Independently resolved LTD verdict probability is 0.1 during background/retention and 0.02 versus 0.18 during learning, swapped during reversal. The production update primitive receives paired verdicts and uniforms. This synthetic assay omits shared-CF timing, eligibility delay and circuit feedback; biological conditioning was not simulated.

All variants start from the same shallow two-level assignment. Original additive allows [0,1]; narrow additive and all discrete variants use [0.25,0.55]. Native transition probabilities are pLTP=0.1, pLTD=0.9. At the shallow crossover their expected expressed steps are 0.03 and 0.27, **30 times** the original additive steps 0.001 and 0.009. Slowed variants use pLTP=0.0033333333 and pLTD=0.03, matching expected shallow steps. This does not match the entire state-dependent learning process or its variance.

Contrast is mean(group A) minus mean(group B), with sign reversed for reversal. Retention divides signed contrast by its value at the start of that retention phase. Negative fractions remain negative: they indicate a small reversed residual, not negative physical weights. All acquired contrasts are nonzero, so the ratios are defined. Mean ± sample SD below describes three seeds; it is not a confidence interval or a significance test.

| Variant | Acquired contrast | Reversed contrast | Initial pattern retained at 600 s | Reversed pattern retained at 600 s |
|---|---:|---:|---:|---:|
| Additive original | 0.992 ± 0.000 | 0.993 ± 0.001 | 0.503 ± 0.022 | 0.483 ± 0.022 |
| Additive narrow | 0.292 ± 0.000 | 0.293 ± 0.001 | 0.020 ± 0.032 | -0.013 ± 0.027 |
| Binary native | 0.144 ± 0.008 | 0.153 ± 0.013 | 0.057 ± 0.072 | -0.013 ± 0.049 |
| Abbott native | 0.219 ± 0.005 | 0.222 ± 0.014 | 0.027 ± 0.011 | -0.027 ± 0.060 |
| Mauk native | 0.282 ± 0.005 | 0.284 ± 0.003 | -0.013 ± 0.082 | -0.061 ± 0.020 |
| Binary slowed | 0.157 ± 0.023 | 0.157 ± 0.002 | -0.023 ± 0.064 | -0.025 ± 0.021 |
| Abbott slowed | 0.195 ± 0.014 | 0.194 ± 0.017 | -0.027 ± 0.066 | -0.063 ± 0.046 |
| Mauk slowed | 0.206 ± 0.004 | 0.159 ± 0.006 | 0.132 ± 0.068 | 0.021 ± 0.065 |

Per-seed retention values and first half-crossings (seed order 0,1,2):

| Variant | Initial 600 s fractions | Reversed 600 s fractions | Initial half-crossing (s) | Reversed half-crossing (s) |
|---|---|---|---|---|
| Additive original | 0.528, 0.494, 0.487 | 0.458, 0.494, 0.499 | >600, 585, 573 | 510, 575, 600 |
| Additive narrow | 0.048, 0.027, -0.016 | -0.038, 0.016, -0.017 | 51, 55, 57 | 55, 61, 51 |
| Binary native | 0.139, 0.024, 0.008 | 0.029, -0.000, -0.068 | 1, 1, 1 | 1, 1, 1 |
| Abbott native | 0.038, 0.026, 0.016 | 0.040, -0.043, -0.078 | 2, 3, 3 | 2, 2, 2 |
| Mauk native | 0.072, -0.020, -0.092 | -0.049, -0.049, -0.084 | 8, 10, 12 | 10, 10, 8 |
| Binary slowed | -0.058, -0.061, 0.051 | -0.000, -0.038, -0.037 | 6, 5, 7 | 6, 5, 5 |
| Abbott slowed | -0.085, -0.041, 0.046 | -0.012, -0.074, -0.102 | 19, 44, 31 | 22, 33, 26 |
| Mauk slowed | 0.109, 0.078, 0.208 | 0.094, -0.000, -0.031 | 113, 133, 133 | 135, 144, 114 |

Half-crossing is the first observed one-second sample at or below half the starting contrast, not a fitted exponential lifetime. `>600` is right-censored, not infinite retention. Complete per-seed acquisition, reversal, movement and state metrics are in the [independent summary](../../results/workflow_campaign/cascade_20260921T145424Z/comparison_analysis/summary.json).

Measured interpretation: native Mauk improves the half-crossing from Abbott's 2–3 s to 8–12 s, but both patterns are effectively lost by 600 s. Slowed Mauk reaches 113–133 s initially and 114–144 s after reversal. Its initial acquired contrast is only 0.206, versus 0.282 for native Mauk and 0.292 for narrow additive; after reversal it reaches 0.159. Thus its memory benefit comes with slower learning/relearning. Slowed binary loses half contrast in 5–7 s and slowed Abbott in 19–44 s: Mauk's hidden states add retention beyond simply slowing binary transitions. However, narrow additive already lasts 51–57 s with substantially stronger acquisition. Original-range additive retains about half its nearly maximal contrast at 600 s, illustrating how strongly allowed range affects this comparison.

Individual movement gives the same caution. During initial retention, RMS displacement divided by allowed range at 600 s is 0.307 ± 0.011 (original additive), 0.561 ± 0.010 (narrow additive), 0.708 ± 0.008 (native Abbott), 0.711 ± 0.022 (native Mauk), and 0.660 ± 0.010 (slowed Mauk). Native cascades therefore do not immobilize synapses despite their lower raw SD. By 600 s every native binary/Abbott/Mauk synapse switched at least once in both retention phases, for every seed. Slowed Mauk leaves 11.7%, 12.1%, 14.5% unswitched in initial retention and 14.1%, 11.7%, 14.1% after reversal. Its conditional median first-switch samples are 106/128/131 s initially and 109/141.5/129.5 s after reversal; native Abbott is 2/2/2 s and native Mauk 10/10/10 s initially. These are first-switch observations in one-second intervals; exact switch times and subsequent dwell-time distributions cannot be recovered from these snapshots.

[Assay trajectories and retention figure](../../results/workflow_campaign/cascade_20260921T145424Z/assay_analysis/comparison.png).

## Full-network checks

The network uses native probabilities only. The additive control shares the initial 0.25/0.55 assignments but retains [0,1] bounds. All comparisons below give seed 0 / seed 1. Time means use samples at 1–120 s; they are not continuous-time integrals. Lag movement averages squared changes over all sample pairs at that lag and all 400 fixed tracked synapses.

| Mode | Time-mean weight | Endpoint mean | Endpoint within-PKJ RMS SD | 60 s RMS movement / range | CF Hz | PKJ Hz | DCN Hz |
|---|---|---|---|---|---|---|---|
| additive | 0.345 / 0.358 | 0.352 / 0.346 | 0.195 / 0.198 | 0.103 / 0.102 | 1.025 / 1.025 | 59.872 / 60.138 | 15.206 / 14.968 |
| binary | 0.390 / 0.388 | 0.372 / 0.443 | 0.147 / 0.144 | 0.711 / 0.717 | 1.500 / 1.500 | 62.075 / 62.065 | 14.054 / 14.043 |
| abbott_cascade | 0.374 / 0.377 | 0.424 / 0.428 | 0.148 / 0.147 | 0.703 / 0.700 | 1.292 / 1.275 | 61.149 / 61.119 | 14.500 / 14.485 |
| mauk_cascade | 0.363 / 0.363 | 0.351 / 0.356 | 0.142 / 0.143 | 0.678 / 0.684 | 1.150 / 1.142 | 60.562 / 60.568 | 14.794 / 14.765 |

Across the two seeds, time-mean weight is 0.35163 ± 0.00933 for additive, 0.38907 ± 0.00186 for binary, 0.37550 ± 0.00196 for Abbott and 0.36297 ± 0.00041 for Mauk (mean ± sample SD). CF rates are respectively 1.025 ± 0, 1.500 ± 0, 1.2833 ± 0.0118 and 1.1458 ± 0.0059 Hz. Two seeds cannot establish robust uncertainty bounds.

The binary mean oscillates substantially: its last-minute range is 0.292–0.475 / 0.290–0.491, and endpoint means differ much more than time means. Temporal SD over 1–120 s is 0.052/0.057 for binary, 0.036/0.034 for Abbott, 0.022/0.019 for Mauk and 0.018/0.017 for additive. An endpoint-only comparison would miss this behavior. All modes sample identical PF input rates within each seed: 19.952 / 19.994 Hz over the 40 recorded PF identities.

Abbott makes 17.602/17.605 million expressed switches, Mauk 7.534/7.617 million, and binary 34.046/34.139 million across 160,000 synapses in 120 s. All 400 tracked synapses switch in both seeds for every discrete model. Full-state counters show 159,998 of 160,000 Mauk synapses switched in seed 0, and all 160,000 in seed 1; all synapses switched in both Abbott and binary seeds. The narrower expressed range limits discrete SD to at most 0.15; it cannot itself establish stability. Sixty-second movement is about 0.68–0.70 of that range for cascades, versus 0.10 of the original additive range. Native Mauk reduces one-second movement relative to Abbott (0.406/0.409 versus 0.571/0.572 of range) but neither preserves individual expressed values over a minute.

Endpoint low-weight fractions from states are Abbott 0.420/0.407, Mauk 0.664/0.648 and binary 0.594/0.358. All discrete weights necessarily occupy one of the two expressed levels. The generic recorder's zero floor/ceiling fractions refer to configuration bounds 0/1 and must not be mistaken for absent boundary occupancy. Eight-state occupancy arrays are retained in the summary; neither cascade has collapsed to a single hidden state. Endpoint between-PKJ SD is 0.0061–0.0101 across modes and between-CF-territory SD 0.0020–0.0081, both much smaller than within-PKJ dispersion. All IO cells within each run have the same total CF count (rate SD zero); this does not alone measure fine-time spike synchrony. Burst counts and a newly fitted synchrony mechanism are unavailable here.

Measured rates shift despite no rate retuning: CF rises from additive's 1.025 Hz to Abbott's 1.275–1.292 Hz and Mauk's 1.142–1.150 Hz. Thus native cascade integration changes the circuit operating point. The hypothesis that slower cascades would retain their isolated-assay benefit inside this feedback loop remains untested; slowed network variants and long cascade runs were outside the accepted focused scope.

[Network comparison figure](../../results/workflow_campaign/cascade_20260921T145424Z/comparison_analysis/network_comparison.png).

## Reproduction and limits

Per-run wall time was 328–359 s including burn-in and durable recording, with peak RSS 140.8–143.9 MiB. Runs used up to four single-thread workers. These are concurrent workload measurements, not a controlled implementation speed benchmark. Each raw network run occupies about 62.2–62.9 MiB; the entire new campaign is about 0.65 GiB. The assay took 76.0–76.6 s per seed for all eight variants. Recording-on/off overhead was not separately measured.

The frozen source is `source/`; `provenance/manifest.json` identifies the dirty source snapshot and the retained CbmSim reference is under `provenance/cbmsim_reference/` at commit `d921c8561597657bfa595f651dc4208fbe165c42`. This ports transition tables into this simulator's existing PF/CF scheduling, not CbmSim's complete network implementation. Source parameters are model choices, not fitted biological probabilities. The original additive path and default `weight_dependence=0` remain unchanged.

Reproduce this cross-analysis without running a simulation:

```bash
cd /home/sk57289/io_equilibrium_sim
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 results/workflow_campaign/cascade_20260921T145424Z/comparison_analysis/analyze_comparison.py --campaign /home/sk57289/io_equilibrium_sim/results/workflow_campaign/cascade_20260921T145424Z
```

The derived `ANALYSIS_COMPLETE.json` records script, summary and figure hashes. The independent script reproduces key assay statistics from raw arrays; the original assay plotting code is retained under `provenance/analysis_code/analyze_assay.py`. Raw data are untouched. All requested focused comparisons are complete; there are no pending simulations. Three assay seeds and two network seeds support descriptive comparisons only. No claim of four-hour cascade stability, biological superiority, or reliable network memory follows from this campaign. Historical missing five-hour arrays and confounded membrane-noise runs were not used as new evidence.
