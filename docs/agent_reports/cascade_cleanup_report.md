# Cascade comparison cleanup — 2026-09-21

Status: COMPLETE; ready for coordinator tests-gate review. Both full suites passed, 105 tests each.

The user requested Abbott and Mauk cascade implementations/comparison, with project cleanup first. This stage is scoped to `/home/sk57289/io_equilibrium_sim`. Prior tracked/untracked changes, scientific explanations, all experiment output and previous workflow records are preserved. The prior complete progress state is saved as `evidence/cascade_cleanup_20260921/progress_before.json` and its stages were archived in the durable progress history before starting this stage. Historical `cleanup_report.md` is retained.

The initial source-tree SHA-256 is `b8fb0d8aa4815dfe5cc7d28f58a85fd31749cfea97aa7397620e070641d0ac38`. Exact file hashes, Git HEAD, dirty status and pre-existing patch are recorded under `evidence/cascade_cleanup_20260921/`. Eccles is the already-authorized standalone host; current free space was 69 GiB, above the 20 GiB reserve. One test process uses single-thread numerical libraries.

Inventory found 69 reconstructable Python bytecode files (605,792 bytes). Each has a retained source file; no historical JSON manifest refers to them. They are the only demonstrated disposable items. No ordinary temporary/backup/editor files were found. Public simulation/analysis entrypoints remain in use or are ambiguous, so no source deletion is justified. In particular, `run_long.py` remains called by the legacy campaign and PF sweep scripts. Scientific source/defaults/RNG behavior are untouched.

`bulky_candidates.json` inventories 1,429,826,775 bytes of optional historical validation artifacts, with exact paths, byte counts and hashes. The inventory distinguishes 58 benchmark/verification state dumps from three old recovery runs and fault-injection bundles. These files are still inputs to retained verification scripts and/or checkpoint manifests; they are not treated as disposable caches. The user then explicitly authorized this exact inventory: **“Delete the old test dumps; keep scientific runs.”** The approved files were deleted only after baseline tests passed and all file hashes/byte counts matched. Direct revalidation using these historical raw dumps now requires regenerating them from the retained harnesses and matching source. All raw `results/` campaigns, checkpoints, pilot arrays, manifests, logs, analyses, and frozen source snapshots are retained.

The required before/after command is `python3 -m unittest discover -s tests -q`. Baseline passed (exit 0) in 487.999 seconds; its log and terminal JSON are retained. Deletion removed 574 files totaling 1,430,432,567 bytes (1.332 GiB), comprising 505 approved old validation dump files and 69 bytecode files. Empty bytecode directories were also removed. Scientific source hashes and the pre-existing tracked diff match the baseline exactly. Post-cleanup tests passed (exit 0) in 495.599 seconds. Final source hashes and tracked diff still match the exact baseline; no scientific source/default/RNG changes occurred. Post-cleanup checks use `PYTHONDONTWRITEBYTECODE=1` to keep deleted caches from immediately returning. This environment flag affects cache writing, not model behavior.


Final evidence and handoff

- `baseline_unittest.log` / `baseline_unittest.json`: 105 tests passed; 487.999 s wall; exit 0.
- `post_cleanup_unittest.log` / `post_cleanup_unittest.json`: 105 tests passed; 495.599 s wall; exit 0.
- `baseline_source.json` / `post_cleanup_source.json` / `final_verification.json`: exact source-file hashes unchanged, and pre-existing tracked diff unchanged.
- `deletion_authorization.json` / `deletion_manifest.json`: approval, all 574 exact file paths, original SHA-256/bytes, successful deletions and removed empty cache directories.
- `bulky_candidates.json` / `artifact_reader_references.txt`: original inventory and affected historical direct readers; retained metrics/reports do not replace removed raw data.

All evidence paths above are under `docs/agent_reports/evidence/cascade_cleanup_20260921/`. The coordinator's new `docs/cascade_comparison_plan.md` is preserved. The historical cleanup report has only a dated pointer appended; historical workflow stages remain in progress history. Source ownership is released to the coordinator tests gate; no optimization or cascade implementation was performed by this cleanup stage. There are no cleaner blockers.

Exact validation command if needed after future source edits:

```bash
cd /home/sk57289/io_equilibrium_sim && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -q
```
