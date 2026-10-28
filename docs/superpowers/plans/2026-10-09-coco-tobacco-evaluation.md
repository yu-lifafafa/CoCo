# CoCo Tobacco Unified Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible CPU-tested evaluation path that calibrates 18 per-class thresholds only on validation data, evaluates test data only with a verified fixed-threshold artifact, and preserves the official legacy CoCo evaluator.

**Architecture:** Add `lib/evaluation_protocol.py` for pure metric, calibration, identity, provenance, and artifact transaction logic. Add a narrowly separated tobacco branch to `evaluate.py`; keep the existing `Evaluator` as the legacy default and replace only its unsafe config loader.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn 1.9.0, PyYAML, CSV/JSON, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-coco-tobacco-evaluation-design.md`

## Global Constraints

- Preserve all uncommitted Task 2/3/4 work and the existing 79 passing tests.
- Do not modify the 18-class mapping, Dataset split, CoCo model/losses, training selection rule, augmentations, optimizer, scheduler, or EMA formula.
- Do not run GPU inference, formal training, real validation calibration, or real test evaluation.
- Do not skip, delete, or fabricate a prediction for `images/task_16/IMG_7950.png`.
- Do not commit or push; commit steps are intentionally omitted.
- Official AP must call existing `VOCmAP` and preserve its exact `np.argsort(score)[::-1]` tie behavior.
- Test evaluation must load fixed thresholds and must not call threshold search.
- Directory replacement is rollback-protected but is not described as absolutely power-loss atomic.

## Review Focus

- Empty evaluation split: reject before metrics/artifacts rather than publish empty success; covered in Task 1.
- Duplicate `(split, file_name)` with different `image_id`: reject identity construction; covered in Task 2.
- Mixed checkpoint prefix state where only some keys start `module.`: reject rather than partially rewrite; covered in Task 3.
- Existing non-empty artifact target on Windows where target rename succeeds but staging rename fails: restore original bytes/tree; covered in Task 4.
- Threshold JSON numerically valid but produced for another best epoch or model/image size: reject before test inference; covered in Task 5.

---

### Task 1: Metric definitions and threshold calibration

**Files:**
- Create: `lib/evaluation_protocol.py`
- Create: `tests/test_evaluation_protocol.py`

**Interfaces:**
- Produces: `validate_evaluation_arrays(gt, scores)`, `compute_evaluation_metrics(gt, scores, thresholds) -> (summary, per_class, predictions)`, and `calibrate_thresholds(gt, scores, split) -> (thresholds, calibration_rows)`.
- Consumes: `VOCmAP`, `average_precision_score`, and the Task 2 ordered mapping.

- [ ] **Step 1: Write failing metric tests**

Cover shape/binary/finite/range validation, empty split rejection, official VOC2012 ordinary and tied-score equality against `VOCmAP`, separately named sklearn AP, no-positive AP=0, healthy/single/multi-label rows, macro arithmetic mean across 18 F1 values, micro aggregation, and Exact Match.

- [ ] **Step 2: Write failing threshold tests**

Cover candidates 0.001 and 0.999, closest-to-0.5 tie, equal-distance lower threshold, exact equality under `>=`, no-positive class returning 0.500 with explicit reason, no-predicted-positive metrics, and non-val split rejection.

- [ ] **Step 3: Run Task 1 RED**

Run: `python -m pytest tests/test_evaluation_protocol.py -q -p no:cacheprovider --basetemp <task-temp>`

Expected: import failure because `lib.evaluation_protocol` does not exist.

- [ ] **Step 4: Implement the minimal metric and calibration functions**

Call existing `VOCmAP` directly; call sklearn AP class-by-class; use explicit zero policies and lexicographic threshold tie resolution fixed by the spec.

- [ ] **Step 5: Run Task 1 GREEN**

Expected: all Task 1 tests pass.

### Task 2: Stable manifest identity and prediction collection

**Files:**
- Modify: `lib/evaluation_protocol.py`
- Modify: `tests/test_evaluation_protocol.py`

**Interfaces:**
- Produces: `TobaccoEvaluationDataset(manifest_path, data_root, split, transform=None, dataset_cls=TobaccoManifestDataset)` and `collect_model_outputs(model, dataloader, device) -> (identities, gt, scores)`.
- Each identity is a mapping containing `image_id`, `file_name`, and `split`, keyed uniquely by `(split, file_name)`.

- [ ] **Step 1: Write failing identity tests**

Cover missing/duplicate primary keys, record-to-manifest mismatch, audit reporting of global image-ID uniqueness without using it as key, and dataset order changes preserving identity/GT/score binding.

- [ ] **Step 2: Write failing prediction collection tests**

Use a small real PyTorch model/dataloader on CPU; assert sigmoid conversion, 18-dimensional alignment, stable batch identities, and explicit image decode failure propagation.

- [ ] **Step 3: Run Task 2 RED**

Expected: missing evaluation Dataset/collection interfaces.

- [ ] **Step 4: Implement identity wrapper and collection**

Read manifest identity once, normalize separators consistently with Task 2, wrap rather than alter `TobaccoManifestDataset`, and bind identity fields in each returned sample.

- [ ] **Step 5: Run Task 2 GREEN**

Expected: all Task 1-2 tests pass.

### Task 3: Checkpoint/config/threshold provenance validation

**Files:**
- Modify: `lib/evaluation_protocol.py`
- Modify: `tests/test_evaluation_protocol.py`

**Interfaces:**
- Produces: `verify_tobacco_evaluation_context(cfg) -> dict`, `load_verified_ema_state_dict(model, cfg) -> dict`, `build_threshold_artifact(...) -> dict`, and `load_verified_thresholds(path, context) -> np.ndarray`.
- Consumes: Task 4 `load_config`, `sha256_file`, `tobacco_label_mapping`, and `verify_best_checkpoint_integrity`.

- [ ] **Step 1: Write failing context tests**

Cover checkpoint SHA, current manifest SHA, mapping order, class count, excluded ID, model, backbone, image size, best epoch, finite best val mAP, selection split/metric, VOC label, and EMA identity mismatches.

- [ ] **Step 2: Write failing strict state-dict tests**

Cover exact keys/shapes, uniform `module.` prefix removal, mixed-prefix rejection, and strict load failure propagation.

- [ ] **Step 3: Write failing threshold provenance tests**

Cover wrong class order, wrong checkpoint/manifest hash, wrong epoch/model/image size, non-val calibration split, wrong candidate range/step/tie policy/operator, wrong threshold count, and out-of-range/non-finite thresholds.

- [ ] **Step 4: Run Task 3 RED**

Expected: missing provenance and loading interfaces.

- [ ] **Step 5: Implement the minimal identity-chain validators**

Run checkpoint pair integrity before `torch.load`; compare exact safe config and metadata fields; use `strict=True`; do not add bypass switches.

- [ ] **Step 6: Run Task 3 GREEN**

Expected: all Task 1-3 tests pass.

### Task 4: Auditable artifacts and directory transactions

**Files:**
- Modify: `lib/evaluation_protocol.py`
- Modify: `tests/test_evaluation_protocol.py`

**Interfaces:**
- Produces: `write_calibration_artifacts(target_dir, bundle, replace_fn=os.replace, ...)`, `write_evaluation_artifacts(...)`, `validate_calibration_artifacts(path, expected_context)`, `validate_evaluation_artifacts(path, expected_context)`, and `ArtifactTransactionError`.

- [ ] **Step 1: Write failing schema/recomputation tests**

Cover deterministic prediction/per-class columns, unique primary keys, split and row-count agreement, binary GT/predictions, finite score range, `pred == score >= threshold`, two AP families, metric recomputation from CSV, package versions, definitions, and full provenance.

- [ ] **Step 2: Write failing publication transaction tests**

Cover new target, existing non-empty target, staging write failure, target-to-backup rename failure, staging-to-target rename failure, post-publication validation failure, successful rollback preserving original tree byte-for-byte, rollback failure retaining evidence, and calibration/evaluation directory isolation.

- [ ] **Step 3: Run Task 4 RED**

Expected: missing artifact interfaces.

- [ ] **Step 4: Implement staged artifact writing and validation**

Use same-parent staging/backup directories and whole-directory renames compatible with Windows and Linux; reload and independently recompute before and after publication.

- [ ] **Step 5: Run Task 4 GREEN**

Expected: all Task 1-4 tests pass.

### Task 5: Explicit workflows and legacy-compatible evaluate entry

**Files:**
- Modify: `evaluate.py`
- Modify: `lib/evaluation_protocol.py`
- Modify: `tests/test_evaluation_protocol.py`

**Interfaces:**
- Produces: `run_val_calibration(cfg, output_root, ...)`, `run_test_evaluation(cfg, output_root, thresholds_path, ...)`, and `evaluate.main(argv=None)` dispatch for `legacy`, `val-calibration`, and `test-evaluation`.
- Preserves: existing `Evaluator` Dataset, prefix handling, metric meters, and console reporting.

- [ ] **Step 1: Write failing val workflow integration test**

Execute the real tobacco workflow entry on CPU with only model/device/Dataset environmental substitutions. Assert only val is constructed, context is verified before model loading, calibration runs once, and the complete calibration directory is published.

- [ ] **Step 2: Write failing test workflow integration test**

Set the calibration function to raise if called; execute the real test entry and assert thresholds are loaded/verified first, only test is constructed, calibration remains byte-identical, and only evaluation artifacts are published.

- [ ] **Step 3: Write failing dispatcher and legacy tests**

Execute `main(argv)` for both tobacco operations and legacy default. Verify safe config loading, legacy `MLDataset(cfg.test_path)`, uniform optional `module.` prefix handling, and unchanged legacy metric/console path.

- [ ] **Step 4: Run Task 5 RED**

Expected: no tobacco workflow/dispatcher exists and legacy uses unsafe YAML loading.

- [ ] **Step 5: Implement minimal workflow and CLI wiring**

Keep GPU defaults for real CLI but allow injected CPU dependencies in tests. Do not catch Dataset decode errors. Keep legacy `Evaluator` behavior unchanged except safe config loading and callable `main`.

- [ ] **Step 6: Run Task 5 GREEN**

Expected: all Task 5 integration and legacy tests pass.

### Task 6: Full verification and review

**Files:**
- Test and review only; production fixes require a new failing regression test first.

**Interfaces:**
- Consumes all Task 1-5 outputs.
- Produces final evidence and risk report.

- [ ] **Step 1: Run the Task 5 suite**

Run: `python -m pytest tests/test_evaluation_protocol.py -q -p no:cacheprovider --basetemp <task-temp>`

Expected: all new tests pass.

- [ ] **Step 2: Run the complete CPU suite**

Run: `python -m pytest tests -q -p no:cacheprovider --basetemp <task-temp>`

Expected: all existing 79 tests plus Task 5 tests pass.

- [ ] **Step 3: Run static and Git checks**

Compile changed Python files, run `git diff --check`, inspect `git status --short`, and prove no Task 2/3/4 core file changed except the approved `evaluate.py` path.

- [ ] **Step 4: Perform final protocol self-review**

Independently recompute representative saved metrics from prediction CSV, inspect the complete diff against the spec, and record any ruling or deferred minor. Report real val/test/GPU as not verified and the corrupt PNG as a blocker.
