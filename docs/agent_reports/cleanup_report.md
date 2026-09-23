# cleaner report

Status: COMPLETE; ready for coordinator tests acceptance. Baseline and post-cleanup full suites both passed; the three conservative removals below are implemented. The coordinator tests gate is a separate pending review.

Authorization and provenance

The user subsequently authorized sequential workflow execution and confirmed Eccles is a standalone compute host. Read `AGENTS.md`, `WORKFLOW.md`, `progress.json`, the cleaner configuration, and the full `HANDOFF_drift_diagnosis.rtf`. Historical scientific claims and the handoff test pass are not represented as fresh validation.

The coordinator preserved the initial dirty tree in `docs/agent_reports/evidence/workflow_start_20260918T073421Z/` at Git revision `8437bde0f6828b045caecc698e258c4430b94492`. Static review evidence and hashes are in `docs/agent_reports/evidence/cleaner_static_review.json`. Existing tracked/untracked changes belong to prior work, including Claude's documented diagnosis/instrumentation, and are preserved.

Changes and removal evidence

- `config.py:2`: removed the obsolete import-line comment claiming there is no validation logic. `_StrictParams.__setattr__` rejects unknown fields and the parameter dataclasses inherit that guard. No configuration value, class or executable statement changed.
- `experiments/drift_figures.py:14`: removed the unused imported alias `variance_decomposition`. The defining function and used `analyze` import remain.
- `experiments/run_gap_weight_sweep.py:34`: removed the unused imported alias `spike_rate_hz`. The defining function and used `summarize` import remain.

Checked both full CLI modules, README entry points, Python AST name loads, and repository references across `config.py`, `sim/`, `experiments/`, and `tests/`; archived evidence copies were excluded. Neither removed alias is exported or referenced by another repository consumer. The used import from each defining module remains, preserving module initialization.

`docs/agent_reports/evidence/cleaner_checks/cleanup.patch` is the exact cleanup-only diff against the initial dirty-tree snapshot; `git diff` also contains pre-existing work. `cleanup_ast_equivalence.json` verifies identical ASTs after excluding only the two intentionally removed import aliases. All function bodies, equations, timestep, connectivity, plasticity and RNG behavior are unchanged by these edits.

Retained ambiguous material

Scientific explanations, units, derivations, numerical caveats, additive-rule diagnosis and instrumentation remain. Public analysis/channel helpers, the single-cell IO wrapper, and the unused `_step(record_pf=False)` parameter remain for caller compatibility. No defaults were retuned; `weight_dependence=0.0` remains unchanged. No experimental data was deleted or modified.

Validation

Both full suites use the exact command `python3 -m unittest discover -s tests -q`, from `/home/sk57289/io_equilibrium_sim`, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. The durable wrapper is `docs/agent_reports/evidence/cleaner_checks/run_check.py`. Per-check JSON records UTC timestamps, Python executable, command/environment, process identity (boot ID and `/proc` stat), Git revision, complete relevant source hashes before/after, wall time, peak child RSS and terminal exit status. Logs and staged/unstaged diffs are retained beside it.

- Baseline: PASS, 97 tests, exit 0, 488.669 seconds wall (unittest 488.539 seconds); source hashes unchanged during testing. Evidence: `cleaner_checks/baseline.json` and `baseline.log`.
- Post-cleanup: PASS, 97 tests, exit 0, 495.217 seconds wall (unittest 495.087 seconds); source hashes unchanged during testing. Evidence: `cleaner_checks/post_cleanup.json` and `post_cleanup.log`.
- CLI smoke checks: `python3 experiments/drift_figures.py --help` and `python3 experiments/run_gap_weight_sweep.py --help`, both exit 0. Evidence: `cleaner_checks/cli_smoke_checks.json` and corresponding help logs. No simulation or figure generation was launched by these commands.

All `cleaner_checks/` paths above are under `docs/agent_reports/evidence/`.

Final verification: current source hashes match the post-cleanup tested tree. Exactly the three listed files differ from the pre-cleanup source hashes. Evidence: `cleaner_checks/final_verification.json`. Source-tree identity (SHA-256 of sorted path/hash JSON): `9b5780224cd6da97ce773c88f498f15b02e6c34a97db23210858388d3b527472`. Baseline and post-cleanup child peak RSS were 88988 and 88964 KiB respectively; these are test-suite measurements, not simulator performance benchmarks. The timing difference is ordinary test-run variation and is not a measured optimization speedup.

No cleaner blockers remain. Hand off to the coordinator tests gate for diff/evidence review of this exact tree. Optimization and experiment stages have not been started by cleaner. No new scientific results are claimed.

Exact verification/resume command from the repository root, if a later source change requires retesting:

```bash
cd /home/sk57289/io_equilibrium_sim && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 -m unittest discover -s tests -q
```

The recorded test interpreter was `/usr/bin/python3`. Successful existing suites should be reviewed before rerunning; their durable logs and terminal status are already present.


2026-09-21 follow-up cleanup: user-approved old validation dumps and disposable bytecode removed before the cascade comparison; scientific source and runs retained. Both new 105-test suites pass. See [cascade_cleanup_report.md](cascade_cleanup_report.md) for exact authorization, deletion manifest and before/after evidence. Historical cleanup results above are unchanged.
