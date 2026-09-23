# Analyst report

Status: DEFERRED UNTIL THE NEW COMBINED CONDITION COMPLETES.

The original eight five-hour biological-time conditions completed successfully. A fresh coordinator audit verified all final 18,000,000-step horizons, complete-state/recording checksums, schemas, finite values, sample cadences and automatic-analysis artifacts. Evidence: `evidence/coordinator_resume_20260918/original_campaign_completion_audit.json`. Existing results and figures are preserved under `results/workflow_campaign/background_20260918T160421Z/analysis/`.

The user requested a ninth condition: IO heterogeneity CV 0.15 combined with double PF→PKJ gain (0.000274), baseline gap coupling, seed 0, and 18,000 seconds of background-only activity after the unchanged 8-second burn-in. The existing frozen pipeline is reused without source or model changes. A bounded combined-condition pilot passed output validation and automatic analysis. The additional long supervisor launched at 2026-09-20T02:00:18 UTC (PID/SID 1354293), using one CPU and an 8 GiB per-job limit; its pilot-based runtime estimate is approximately 12 hours. It audits/skips the eight completed runs and executes only the new combination. Its extension directory is `results/workflow_campaign/background_20260918T160421Z/extensions/combined_20260920T015726Z/`.

This is an interim workflow record, not an analyst-stage synthesis. The analyst has not started because all nine requested conditions must finish first. The completed single-factor conditions supply matched baseline/heterogeneity/gain controls for the combination, but one seed cannot quantify between-seed uncertainty. Final analysis must separate measured activity and weight distributions/individual trajectories from interpretations and hypotheses; stable means cannot establish stability of individual weights.

Authoritative status, full process identities, blockers, run IDs and exact resume commands are in progress.json. Do not duplicate a live run or rerun the eight completed conditions.


2026-09-21 completed focused cascade comparison: see [cascade_comparison_report.md](cascade_comparison_report.md). Both optional variants and all requested focused trials are validated; native cascades do not resolve individual-synapse instability in these tests. This pointer preserves the historical interim record above.
