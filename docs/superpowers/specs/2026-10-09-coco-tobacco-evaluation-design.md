# CoCo Tobacco Unified Evaluation and Calibration Design

Date: 2026-10-09

## Purpose

Add a reproducible 18-class tobacco evaluation path that calibrates per-class thresholds only on validation data and evaluates test data only with a previously saved, identity-checked threshold artifact. Preserve the official CoCo legacy evaluator and do not change the model, losses, training selection rule, dataset split, augmentations, optimizer, scheduler, or EMA formula.

This phase provides CPU-tested evaluation infrastructure. It does not claim real validation calibration, real test evaluation, CUDA, AMP, or full-model inference success.

## Baseline Evidence

- Branch: `feat/tobacco-coco-18cls`; official-derived HEAD: `342c4e35b19fc773ae71565dbe17be70d56bfcdc`.
- Task 2/3/4 baseline: 79 CPU tests pass.
- `evaluate.py` currently constructs only `MLDataset(cfg.test_path)`, loads configuration through unsafe `yaml.load`, strips an optional `module.` checkpoint prefix, and reports official VOC2012 plus legacy threshold/top-k meters.
- Task 4 provides safe `load_config()` and `verify_best_checkpoint_integrity()`.
- Local sklearn version is 1.9.0.
- The real manifest used for protocol evidence is `D:/tech_case/baseline/image_manifest_updated.csv`, SHA-256 `ce51205b537c3d1eee28b3d8112ce1edb6454c6f16cd3b441387bff2be2c56b5`, with 14,347 rows: train 10,050, val 2,866, test 1,431.
- The real manifest has zero missing `image_id`, zero duplicate global `image_id`, zero duplicate `(split, image_id)`, and zero duplicate `(split, file_name)`. Despite current global uniqueness, evaluation uses `(split, file_name)` as the primary key and records `image_id` only as audit metadata.
- The real test image `images/task_16/IMG_7950.png` is known to fail Pillow decoding. It must stop real evaluation explicitly; it is never skipped or replaced.

## File Structure

Add `lib/evaluation_protocol.py` for pure metrics, threshold calibration, identity/provenance validation, and transactional artifact writing. Add `tests/test_evaluation_protocol.py` for mathematical, workflow, integration, and fault-injection tests. Modify `evaluate.py` only to add a tobacco-specific evaluator/CLI branch and safe configuration loading while preserving the legacy `Evaluator` class behavior.

Do not modify `lib/metrics.py`, `lib/tobacco_dataset.py`, `lib/training_protocol.py`, `train.py`, or any model/loss file unless a demonstrated in-scope defect makes a minimal change necessary through a new RED test.

## Metric Definitions

All inputs are finite arrays with shape `[N, 18]`. Ground truth is binary. Raw model logits are transformed with sigmoid before metrics or artifact output.

### Official CoCo metric

`voc2012_per_class_ap` and `voc2012_map` are computed by the existing `VOCmAP(num_classes=18, year="2012", ignore_path=None)` implementation. The new module does not reproduce it with another sorter. This deliberately preserves official semantics:

1. Per class, `np.argsort(score)[::-1]` orders individual samples.
2. Equal-score ordering follows that exact NumPy call and may depend on its tie order.
3. The existing VOC2012 precision-envelope integration is used unchanged.

Tests compare the protocol result directly to `VOCmAP` for both ordinary scores and tied scores, including per-class AP and mAP. No stable sorting or grouped-tie substitution is permitted for this metric.

### Cross-model metric

`sklearn_per_class_ap` is computed class by class with sklearn `average_precision_score`; `sklearn_map` is the arithmetic mean of all 18 values. The local sklearn definition is the non-interpolated step integral `sum(delta_recall * precision)` and groups equal scores at unique thresholds. These names never appear as an unqualified `mAP`.

For a class with no positive GT, both reported AP variants are explicitly `0.0`. The class remains in all arrays and averages. This suppresses no class and avoids NaN; artifact metadata records the policy as `zero_if_no_positive_gt`.

### Thresholded metrics

Predictions use `score >= threshold` everywhere. Per-class precision, recall, and F1 use zero-division value `0.0`. Micro metrics aggregate TP/FP/FN across all samples and classes. Macro precision, recall, and F1 are the arithmetic means of the corresponding 18 per-class values. Exact Match Ratio is the fraction of samples whose complete 18-dimensional prediction equals GT. Official CF1 is not reused as macro-F1.

## Threshold Calibration

`calibrate_thresholds(gt, scores, split)` rejects any split other than `val`. For every class it evaluates exactly 999 candidates `0.001, 0.002, ..., 0.999` using `>=` and maximizes per-class F1.

Tie resolution uses this lexicographic order:

1. maximum F1;
2. minimum absolute distance to `0.5`;
3. lower threshold.

If a class has no positive GT, every candidate has F1 zero and the deterministic result is `0.500`. The class is marked with `degenerate_reason: no_positive_gt` in per-class calibration metadata.

The test workflow has no threshold-search parameter or call. It accepts only a verified threshold artifact created for `split=val`.

## Stable Sample Identity

Add `TobaccoEvaluationDataset`, a wrapper around `TobaccoManifestDataset`, without changing the base Dataset return interface.

The wrapper reads manifest identity fields and validates:

- non-empty `image_id` and `file_name`;
- uniqueness of `(split, file_name)` across the manifest;
- uniqueness of `image_id` is measured and recorded but not assumed as the primary key;
- each selected base record maps to exactly one manifest row by `(split, normalized file_name)`.

Each evaluation item adds `image_id`, `file_name`, and `split`. GT and scores stay bound to these batch-provided identities. Output sorting uses `(split, file_name)` only after the full row has already bound identity, GT, score, and prediction. Dataset or batch order changes therefore cannot realign scores against another sample.

## Checkpoint and Configuration Validation

Tobacco evaluation loads config through Task 4 `load_config()`. Before model weights are loaded it must:

1. run `verify_best_checkpoint_integrity(checkpoint, metadata)`;
2. recompute the current manifest SHA-256;
3. compare checkpoint SHA-256, manifest SHA-256, ordered 18-class mapping, excluded source ID, architecture/model, backbone, image size, number of classes, selection split, selection metric, VOC implementation label, EMA variant, best epoch, and finite best val mAP across config and checkpoint metadata;
4. require exactly the Task 4 EMA identity;
5. strip exactly one leading `module.` prefix from every key when present, matching legacy behavior;
6. load the model with `strict=True`, rejecting missing keys, extra keys, or tensor shape mismatch.

Calibration and evaluation artifact provenance copies the verified values. Any mismatch stops before inference or artifact creation.

## Explicit Workflows

### Validation calibration

`run_val_calibration(...)`:

1. verifies configuration/checkpoint context;
2. constructs only `split="val"` with the official evaluation transform;
3. collects sigmoid scores, GT, and stable identities;
4. computes both AP families and all thresholded metrics;
5. searches thresholds only through `calibrate_thresholds(..., split="val")`;
6. writes and validates the complete `calibration` artifact set.

### Test evaluation

`run_test_evaluation(...)`:

1. verifies configuration/checkpoint context;
2. loads an existing `thresholds.json` before inference;
3. validates its split, 18-class order, checkpoint SHA, manifest SHA, best epoch, model identity, candidate range, step, tie policy, and `>=` operator;
4. constructs only `split="test"`;
5. collects test scores/GT and applies the loaded thresholds;
6. writes and validates only the `evaluation` artifact set.

It never calls the threshold calibration function and never modifies the calibration directory.

## Artifact Contract

Default layout:

```text
outputs/tobacco_coco/
  calibration/
    val_predictions.csv
    val_metrics.json
    val_per_class.csv
    thresholds.json
  evaluation/
    test_predictions.csv
    test_metrics.json
    test_per_class.csv
```

Prediction CSV columns are deterministic:

- `image_id`, `file_name`, `split`;
- 18 `gt_{model_index}_{label}` fields;
- 18 `score_{model_index}_{label}` fields;
- 18 `pred_{model_index}_{label}` fields.

Per-class CSV records model index, source ID, label, both AP definitions, precision, recall, F1, support, predicted positives, and threshold.

Metrics JSON records unambiguous metric keys and definitions, zero-division/no-positive policy, threshold operator, sample/class counts, package versions, manifest/checkpoint hashes, mapping, model configuration, best epoch, and best val mAP.

Threshold JSON records the ordered mapping, 18 thresholds, per-class calibration status, validation split, `[0.001, 0.999]`, step `0.001`, F1 objective, both tie rules, `>=`, and full checkpoint/config provenance.

Artifact readers validate schema, dimensions, unique primary keys, split, row count, finite values, GT binary values, score range, prediction recomputation, threshold identity, and metric recomputation from saved CSV. JSON floating values are compared using defined numerical tolerances.

## Directory Transaction and Failure Handling

Each workflow writes all files into a unique same-parent staging directory, closes them, reloads them, and runs full artifact validation before publication.

For an existing non-empty target directory on Windows or Linux:

1. rename the current target to a unique backup directory;
2. rename staging to the target;
3. reload and validate the published target;
4. remove the backup only after success.

On handled failure, restore the prior target directory. If rollback itself fails, raise an artifact transaction error and retain staging/backup evidence. Calibration and evaluation are separate transactions, so test publication cannot replace thresholds. Tests inject staging write failure, first rename failure, second rename failure, post-publication validation failure, and rollback failure.

This is not described as absolutely atomic against power loss or process termination between directory renames. Retained provenance and artifact validation detect incomplete or mismatched outputs on later use.

## Legacy Compatibility

The existing `Evaluator` remains the default legacy path. Its `MLDataset(cfg.test_path)`, optional `module.` prefix removal, VOC2012 calculation, threshold/top-k calculations, and console output remain behaviorally unchanged. Only config loading changes to Task 4 `load_config()`.

The CLI adds explicit tobacco operations without changing the legacy default. Tests import and execute the real dispatcher with test doubles, proving tobacco calibration and test evaluation route to distinct functions. A legacy regression test proves the old Evaluator still constructs `MLDataset` from `test_path` and accepts prefixed checkpoint keys.

## Test Strategy

Create `tests/test_evaluation_protocol.py` before production code and capture RED. Tests cover:

1. exact 18-class mapping;
2. sigmoid/GT/identity alignment under reordered input;
3. official VOC2012 ordinary and tied-score equivalence;
4. sklearn AP ordinary and tied-score behavior;
5. macro versus micro definitions;
6. healthy, single-label, and multi-label rows;
7. threshold bounds;
8. closest-to-0.5 tie resolution;
9. equal-distance lower-threshold resolution;
10. equality under `>=`;
11. no-positive and no-predicted-positive behavior;
12. non-val calibration rejection;
13. test workflow calibration-function trap;
14. threshold class-order rejection;
15. checkpoint SHA mismatch;
16. manifest SHA mismatch;
17. checkpoint/JSON mismatch;
18. CSV/JSON reload and independent recomputation;
19. reordered Dataset identity alignment;
20. legacy Evaluator behavior;
21. actual tobacco evaluator/CLI CPU integration with model, Dataset, and device dependencies substituted only at environmental boundaries;
22. directory replacement and rollback fault injection;
23. all existing 79 tests.

Synthetic tests are reported only as CPU protocol/integration tests. No output from them is called real tobacco calibration or evaluation.

## Completion Criteria

- Both AP families are separately named and independently verified.
- Official VOC2012 output exactly matches existing `VOCmAP`, including tied-score ordering behavior.
- Validation alone can produce thresholds; test can only consume an existing verified threshold artifact.
- Every prediction row binds stable identity, GT, score, and fixed-threshold prediction.
- Checkpoint, manifest, mapping, model, EMA, threshold, and artifact provenance form a verified identity chain.
- Published artifact directories are complete or a handled failure restores the previous directory.
- Legacy evaluation behavior remains available.
- Existing 79 tests and all new CPU tests pass.
- Real val calibration, real test evaluation, GPU, corrupted image resolution, and formal inference are explicitly not verified.
