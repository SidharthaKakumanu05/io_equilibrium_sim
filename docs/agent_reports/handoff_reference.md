# Claude handoff reference

Source: [HANDOFF_drift_diagnosis.rtf](../../HANDOFF_drift_diagnosis.rtf), dated
2026-09-18. This is a navigation summary of prior work, not a new replication.
Read the original, especially sections 2, 7, 8, and 9. Do not replace it with this
summary or treat its suggested experiments as authorization.

- Section 3 attributes individual PF->PKJ weight diffusion to the additive rule:
  common feedback regulates means but leaves weight redistribution unobserved.
  It reports about 99.8% within-PKJ variance in baseline conditions. These are
  historical results to verify against retained artifacts, not setup measurements.
- Sections 4–6 distinguish diffusion from IO entrainment. Smaller delta values
  delay diffusion; weight dependence can make dispersion stationary but changes
  the model and equilibrium rate. Keep `weight_dependence=0.0` as the default.
- Section 7 marks `noise_sigma=2.0` runs as confounded by calibration, warns that
  low dispersion can be boundary pileup, and explains that some gap-current
  correlations are undefined because their predictor has zero variance.
  Prefer `ev_io` / `ev_t` over ragged object-array `cf_times` for analysis.
- The 1800-second baseline and wd1 stationarity results have one seed each;
  the 600-second conditions are reported with two seeds. Do not imply broader
  replication. Do not infer individual stability from a stable mean.
- Section 8 attributes the recent `weight_dependence` addition to `config.py`,
  `sim/plasticity.py`, one wiring line in `sim/simulate.py`, README additions,
  and new diagnostic/analysis scripts. Other existing changes in
  `sim/simulate.py`, `sim/neurons.py`, `experiments/run_baseline.py`, and
  `experiments/run_calibration.py` predate that work. Preserve all of them.
- Reported artifacts were `~/drift_diag/*.npz`, logs, `rule_mc.json`, and
  `results/drift/*.png`. The handoff says `/home/sk57289/longsim` was deleted and
  only five-hour summaries/PNGs survive. The setup inspection did not find
  `~/drift_diag` or `~/longsim` at those paths; locate retained artifacts before
  relying on raw-data availability. Do not regenerate long campaigns implicitly.
- Historical tests: `python3 -m unittest discover -s tests -q`, 97 tests, about
  500 seconds. Historical bandwidth guidance: approximately 10 concurrent
  simulations rather than 64, subject to current resource authorization and
  pilot measurements. This is not permission to launch ten jobs.

Existing long-run files inspected during setup: `experiments/run_long.py`,
`sim/long_run.py`, and `sim/recording.py`. Spike streams are buffered, but slow
weight arrays remain in memory until final serialization. No complete-state/RNG
checkpoint or resume interface was found. `SpikeStream` opens files with `wb`;
reusing an output directory can destroy existing recordings. The current
`experiments/launch_long_campaign.sh` launches many detached jobs without the
new workflow gates. Do not invoke it as a ready-made resumable pipeline.
