# Homeostatic pilot decline and completed four-hour result

The user selected **use the completed result** on 2026-09-21. Campaign `homeostasis_20260921T034911Z`, run `homeostatic_scaling_seed0_14400s`, is complete and validated. No restart, new experiment, source change or raw-data modification was needed. This report supplements the historical `final_report.md`, which remains unchanged.

The pilot's large initial mean-weight drop is accounted for almost entirely by excess LTD under the existing additive rule. The completed four-hour trajectory does not continue that large downward population trend. Individual synapses nevertheless continue moving substantially late in the run.

## Measured attribution and mechanism

The 30-second pilot starts with mean weight 0.5 and ends at 0.3453162095, a change of −0.1546837905 (30.94% of the initial value). Across 160,000 synapses, recorded plasticity counts are 12,043,770 LTD and 83,646,262 LTP events. The frozen rule subtracts 0.009 for LTD and adds 0.001 for LTP, then clips to [0,1]. Thus the **attempted** additive contribution to the mean is

`(0.001 × 83,646,262 − 0.009 × 12,043,770) / 160,000 = −0.154672925`.

The recorded, realized homeostatic contribution is only −0.00001265334 per synapse, about 0.0082% of the observed drop. A +0.00000178788 residual remains between these terms and the observed change, consistent with additive clipping plus numerical accounting. There is no additive clipping-event counter, so event counts alone are not an exact realized-contribution ledger. Endpoint floor occupancy of zero does not rule out earlier clipping.

LTD represents 12.5862% of resolved pilot events; unclipped additive balance requires 10%, because LTD is nine times larger than LTP. Pilot CF rate is 1.2667 Hz, versus the approximate 1 Hz balance implied by the 100 ms LTD window and zero null window. CF rate is 1.6 Hz in the first 10 seconds and 1.1 Hz in each following 10-second interval. These measured counts directly explain the depression; describing it as startup settling of the feedback loop is an interpretation supported by the subsequent return toward balance, not a separate intervention test.

Frozen `source/sim/homeostasis.py` implements `d log(w_ij)/dt = (1 − r_i/60 Hz)/14,400 s`, using a 60-second exponentially smoothed postsynaptic rate and one-second updates. It regulates firing rate, with no target of returning each weight to 0.5. Pilot PKJ activity averages 61.418 Hz, so the small net downward scaling is the expected sign. A 30-second pilot is far shorter than either the four-hour adaptation constant or a full settling assessment. When activity is close to target, the homeostatic adjustment is small even over a long run.

## What the completed run shows

| Measurement | 30-second pilot | Four-hour run |
| --- | ---: | ---: |
| Final mean weight | 0.345316 | 0.343527 |
| Final within-PKJ RMS weight SD | 0.080642 | 0.259739 |
| Whole-run PKJ rate, Hz/cell | 61.4180 | 60.0064 |
| Whole-run CF rate, Hz/cell | 1.26667 | 1.00896 |
| Realized homeostatic mean-weight change | −0.00001265 | −0.00009446 |

Mean weights averaged separately over the four hours are 0.354676, 0.353540, 0.354210 and 0.352979. These measurements support a fluctuating population operating level after startup, not continued collapse or a formal proof of stationarity. The final instantaneous mean is lower than the last-hour average.

Late individual movement is substantial: the 400 fixed tracked synapses have RMS changes of 0.09866 over 60-second lags and 0.25504 over 600-second lags, using only pairs within the last hour. Their endpoint displacement from hour three to hour four is 0.34500 RMS; 76.25% move more than 0.1. Descriptive pooled correlations after centering each trajectory over the final three hours are 0.91246 at 60 seconds and 0.42410 at 600 seconds; no correlation time was fitted.

At the final snapshot, 99.8144% of total weight variance lies within PKJ cells. Between-CF-territory SD is 0.003991; between-PKJ-within-territory RMS SD is 0.010463. The shared rate-driven multiplicative factor provides no individual reference weight, while additive learning continues. These observations support persistent individual wandering despite a comparatively stable mean.

Final floor occupancy is zero and ceiling occupancy 0.020625%; across one-second snapshots the maximum population floor/ceiling fractions are 4.790625%/1.1375%. Boundary effects matter: the four-hour attempted additive mean change is −2.58062, with a +2.42424 residual after realized homeostasis is removed. It would be incorrect to treat attempted event totals as realized long-run weight loss.

Other recorded four-hour mean rates are sampled PF 19.9892 Hz and DCN 15.0656 Hz. No recorded CF ISI is below 100 ms. Nearly identical total CF counts do not establish precise synchrony; this focused attribution does not estimate synchrony from spike timing.

## Evidence, validation and limits

The analyst read `AGENTS.md`, `WORKFLOW.md`, central progress, all three preceding stage reports, and the complete original handoff. Cleanup and optimizer results are historical engineering evidence; the homeostatic runner subsequently records passing 105 tests, disabled-rule equivalence and recovery tests. The [fresh completion reconciliation](evidence/homeostasis_20260921T033702Z/restart_reconciliation_20260921.json) validates final step 14,400,000, all pinned source hashes, raw checksums, finite values, units, resolved configuration, expected counts, cadence and analysis checksums. It reports 14,401 weight snapshots and 14,400 activity bins with no recovery-boundary gaps or duplicates. Retained campaign storage is 3.129 GB; no new simulator benchmark is inferred from earlier optimizer timings.

The [read-only attribution script](evidence/homeostasis_drift_20260921/diagnose.py), [numeric diagnosis](evidence/homeostasis_drift_20260921/diagnosis.json), and [figure](evidence/homeostasis_drift_20260921/drift_attribution.png) retain the contribution calculation and intervals. Independent analyst extraction rechecked each committed recording checksum, exact one-second weight cadence, summary scalars, four hourly means, and the dispersion/trajectory/boundary statistics above. The additional decomposition uses the frozen connectivity's eight consecutive PKJ cells per CF territory; pooled correlations use 400 trajectories sampled every second from t=3,600 through 14,400 s. No invalid raw values were discarded or undefined correlations converted to zero.

Full summaries are under `results/workflow_campaign/homeostasis_20260921T034911Z/{pilot_analysis/homeostatic_scaling_pilot30s,analysis/homeostatic_scaling_seed0_14400s}/summary.json`. Both runs use seed 0; the pilot and long result do not provide independent seed replication. PF rates use 40 sampled identities. No matched homeostasis-off counterfactual was analyzed, so the near-target PKJ rate cannot be credited solely to the added rule. The historical membrane-noise confound, optional weight-dependence model, and unavailable deleted five-hour raw arrays are not evidence for this new trajectory. Additive parameters and scientific defaults were not retuned.

All requested analysis is complete with no blocker. Simulation resume is inapplicable. To regenerate only the derived diagnosis and figure:

```bash
cd /home/sk57289/io_equilibrium_sim && /usr/bin/python3 docs/agent_reports/evidence/homeostasis_drift_20260921/diagnose.py
```
