# Simulation project instructions

Read `WORKFLOW.md`, `docs/agent_reports/progress.json`, and
`HANDOFF_drift_diagnosis.rtf` before work. The handoff records Claude's recent
diagnosis, changes, scientific constraints, and confounds; see also
`docs/agent_reports/handoff_reference.md`. Verify historical claims against
available artifacts before presenting them as new results.

The user subsequently authorized **sequential workflow execution** on 2026-09-18.
Start each stage only after its prerequisites in WORKFLOW.md pass. Long conditions
run for five simulated biological hours with background activity only; the user
delegated seeds, comparisons of gap coupling/heterogeneity/synaptic strength, and
sensible resource/storage limits on the standalone Eccles host. The resolved
decisions and conservative ceilings are in docs/agent_reports/progress.json.
Do not treat the handoff's open-work list as an instruction to run other experiments.

Use the project agents sequentially: cleaner -> tests -> optimizer -> validation
-> experiment_runner -> successful completion -> analyst. Never overlap edits to
`config.py`, `sim/`, or simulation drivers in `experiments/`. Freeze and identify
the exact code snapshot used by each running trial. Preserve pre-existing tracked
and untracked work; never reset it, silently overwrite it, or attribute it to a
new agent. Never delete existing experimental data without explicit user approval.

Preserve scientific explanations, equations, timestep, connectivity, cell types,
plasticity semantics, and RNG streams/draw order. Do not retune parameters or
change default `weight_dependence=0.0` to make rates match a paper. Claude's
optional soft-bound model and proposed delta scaling are not authorized defaults.

Tests use `python3 -m unittest discover -s tests -q`; the handoff reports roughly
500 seconds for 97 tests, not a fresh validation of today's files. Run baseline
and post-change tests for cleanup, and tests plus fixed-seed comparisons for
optimization. Stop on failures; never silently relax tolerances.

Agents must update their assigned report and `docs/agent_reports/progress.json`
with evidence, blockers, run IDs, and exact resume commands. No heavy jobs on a
login node. Check scheduler policy, resource budgets, disk, and permissions before
launch. Clarify whether 4–5 hours means wall-clock or simulated biological time,
trial count/seeds, and resource/storage limits before any long trial.

The analyst may write analysis scripts, figures, and reports, but must not modify
simulation source or raw datasets. A stable population mean is not evidence of
stable individual synapses. Separate measured results, interpretations, and
untested hypotheses. Missing data and intentionally undefined statistics must
be labeled explicitly.
