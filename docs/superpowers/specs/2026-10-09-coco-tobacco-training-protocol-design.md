# CoCo Tobacco Training Protocol Design

Date: 2026-10-09

## Purpose

Integrate the existing 18-class `TobaccoManifestDataset` and the healthy-sample-safe CoCo loss into the official training entry point while preserving the original CoCo model, losses, optimizer, scheduler, EMA formula, augmentations, and legacy dataset behavior. Tobacco experiments must use only manifest `train` and `val` splits during training and model selection. The manifest `test` split must not be constructed, read as image data, or used for checkpoint selection.

This phase is CPU-tested integration work. It does not claim CUDA, AMP, full-backbone, or real training verification.

## Existing Baseline and Constraints

- The branch is `feat/tobacco-coco-18cls` at official-derived HEAD `342c4e35b19fc773ae71565dbe17be70d56bfcdc`.
- Task 2 adds `TobaccoManifestDataset` and shared official transforms without replacing `MLDataset`.
- Task 3 changes only the contrastive-loss handling required for all-zero targets and device-neutral CE targets.
- The official `Trainer` currently creates `MLDataset(cfg.train_path)` and `MLDataset(cfg.test_path)`, treating `test.txt` as validation. That legacy convention remains unchanged for VOC, COCO, NUS-WIDE, and VG500.
- `evaluate.py` is outside this phase and must not change.
- No commit, push, GPU execution, dependency installation, weight download, threshold calibration, or formal training is allowed.

## Architecture

### Training protocol module

Add `lib/training_protocol.py` as a focused protocol boundary. It will contain:

1. Tobacco protocol constants and the ordered source-to-model label mapping derived from `lib.tobacco_dataset` constants.
2. `build_training_datasets(cfg, ...)`, which returns exactly `(train_dataset, val_dataset)`.
3. Manifest SHA-256 calculation and experiment protocol metadata construction.
4. Finite-metric validation and strict best-result comparison.
5. EMA checkpoint and JSON metadata save, rollback, and integrity verification helpers.
6. Small pure helpers needed to test safe logging intervals and config restoration without invoking CUDA.

It will not contain forward propagation, loss aggregation, AMP, optimizer, scheduler, EMA updates, or a replacement Trainer.

### Minimal Trainer wiring

`Trainer.__init__()` will call `build_training_datasets(cfg)` directly. Tests will execute the real constructor with narrowly substituted CUDA/model/loader dependencies and a dataset builder that fails if any `test` split is requested. This proves the real call chain, rather than only testing the helper in isolation.

For tobacco:

```python
train = TobaccoManifestDataset(
    cfg.manifest_path,
    cfg.data_root,
    split="train",
    transform=build_coco_transform(cfg, training=True),
)
val = TobaccoManifestDataset(
    cfg.manifest_path,
    cfg.data_root,
    split="val",
    transform=build_coco_transform(cfg, training=False),
)
```

No tobacco `test` dataset or loader exists in `Trainer`. The manifest parser may validate manifest schema and split declarations, but it does not resolve or open images from unselected splits. Test scores and labels are not accepted by the selection API.

For legacy datasets, the helper preserves the official calls:

```python
train = MLDataset(cfg, cfg.train_path, training=True)
val = MLDataset(cfg, cfg.test_path, training=False)
```

The legacy name `test_path` and its role as validation remain compatible.

## CLI and Configuration

`get_args()` adds optional `--manifest-path` and `--data-root` arguments. Existing defaults remain unchanged. `--img-size`, `--batch-size`, and all official arguments retain their current definitions.

`prepare_env()` branches by dataset:

- `data != tobacco`: preserve official `train.txt`, `test.txt`, `label.txt`, and optional `ignore.npy` derivation and class-count behavior.
- `data == tobacco`: require existing manifest file and data-root directory, set `num_classes=18`, set `ignore_path=None`, and do not require or synthesize TXT/label/ignore files.

The tobacco config records:

- `dataset: tobacco`
- `task: multilabel_classification`
- `num_classes: 18`
- `excluded_source_class_id: 13`
- `selection_split: val`
- `selection_metric: mAP`
- `selection_variant: ema`
- `metric_implementation: VOC2012`
- ordered 18-class mapping with source ID, model index, and label column
- manifest SHA-256
- all CLI model and training parameters

Configuration is written with `yaml.safe_dump()` and loaded with `yaml.safe_load()`. Paths are serialized as strings and mapping order is represented explicitly by a list. A loader returns an `argparse.Namespace` suitable for later model restoration. Tests cover both tobacco and representative legacy configurations.

## Validation and Selection

The validation loop remains the official loop:

1. Evaluate current model on the validation loader.
2. Evaluate EMA model on the same validation loader.
3. Compute and log both VOC2012 mAP values.
4. Validate that both values are finite; NaN or Inf raises an explicit error before selection.
5. Use only EMA validation mAP for best-checkpoint selection.

Best-state semantics:

- Initial best is `None`, so a first valid EMA mAP of `0` is saved.
- A candidate replaces the best only when `candidate > previous_best`.
- Equal values retain the earlier epoch and files.
- The ordinary model mAP is logged but cannot enter the selection function.
- Early-stopping patience is reset only by a strict EMA validation improvement.

No test metric or test dataset is represented in the selection interface.

## Checkpoint and Metadata Consistency

The EMA weight file remains the official `ema_model.state_dict()` object so existing loading semantics and key names are preserved. Metadata is stored separately in `best_checkpoint_info.json`.

Metadata includes:

- architecture/model and backbone
- number of classes and image size
- seed
- manifest SHA-256
- complete ordered 18-class mapping
- excluded source ID
- selection split, metric, implementation, and model variant
- best epoch and best validation mAP
- checkpoint path and checkpoint SHA-256

### Save protocol

Only rank 0 invokes the save helper. Within the checkpoint directory:

1. Write the candidate EMA state dict to a unique temporary file.
2. Flush/close it, calculate SHA-256, reload it on CPU, and verify that it is a state dictionary loadable into the supplied EMA model shape.
3. Build metadata from the candidate hash and write JSON to a unique temporary file; reload it and validate required values.
4. Before replacement, copy any existing final checkpoint and metadata to unique backup files.
5. Replace the checkpoint final path with the candidate and verify its final SHA-256.
6. Replace the metadata final path and verify that its recorded SHA-256, epoch, mAP, and checkpoint path match the installed checkpoint.
7. Only after complete verification remove backups and temporary files and update in-memory best epoch/mAP.

### Failure handling

If any write, replace, hash, reload, or metadata verification step raises:

- Restore both prior final files from backups when they existed.
- Remove newly installed final files when no prior pair existed.
- Verify the restored pair when possible.
- Preserve diagnostic exception context and never update in-memory best state.
- Clean temporary artifacts when safe; retain evidence if rollback itself fails.

Fault-injection tests cover candidate checkpoint write failure, metadata write failure, first replacement failure, second replacement failure, and rollback behavior with an existing best pair.

Two separate file replacements are not described as an absolutely atomic transaction. A process or machine termination between replacements cannot be fully rolled back by Python exception handling. `verify_best_checkpoint_integrity()` compares the final file SHA-256 against JSON and checks identifying fields, so a partial or externally altered pair is explicitly rejected on later use. Backup artifacts provide recovery evidence but are not silently treated as success.

## Metrics Compatibility

`VOCmAP(ignore_path=None)` means no ignore mask. A non-empty valid ignore path retains the existing `np.load()` and masking behavior. VOC2007/VOC2012 AP equations are unchanged. Tobacco training continues to request VOC2012.

## Failure Evidence and Small-Loader Safety

The tobacco exception path will not delete the experiment directory when training fails. Config, logs, temporary/backup checkpoint evidence, and traceback remain available. Legacy cleanup behavior remains unchanged.

The training log interval becomes `max(1, len(train_loader) // 6)` to prevent modulo by zero for small loaders without changing loss, optimizer, scheduler, or update frequency.

## Test Strategy

Create `tests/test_training_protocol.py` before production changes and capture RED output. Tests cover:

1. Tobacco CLI arguments.
2. Required and valid manifest/data-root checks.
3. Exactly train and val tobacco dataset construction.
4. A test-split constructor trap.
5. The real `Trainer.__init__()` call chain using the protocol builder.
6. Absence of alternate test access in simulated training/selection.
7. Fixed 18-class validation and ordered mapping.
8. Legacy `MLDataset(train_path/test_path)` compatibility.
9. First mAP zero, strict improvement, non-improvement, and tie behavior.
10. NaN/Inf rejection.
11. EMA checkpoint reload with unchanged state-dict structure.
12. Metadata epoch, score, path, mapping, and manifest hash.
13. Candidate write, JSON write, replace, and rollback fault injection.
14. Post-save and corrupted-pair SHA-256 verification.
15. Safe YAML tobacco round trip and legacy-config round trip.
16. `VOCmAP(ignore_path=None)` and unchanged non-empty ignore behavior.
17. Small-loader log interval.
18. Tobacco failure evidence retention policy.
19. Existing Dataset and contrastive tests.

Tests may substitute CUDA/model construction and DataLoader internals when invoking `Trainer.__init__`, but must execute the real constructor and assert its calls. They are CPU integration tests and are not reported as GPU training.

## Files

Expected additions:

- `lib/training_protocol.py`
- `tests/test_training_protocol.py`
- this design document
- an implementation plan under `docs/superpowers/plans/`

Expected minimal modifications:

- `train.py`
- `lib/utils.py`
- `lib/metrics.py`

Explicitly unchanged:

- `evaluate.py`
- `models/mlic.py` beyond the already present Task 3 change
- Backbone, Transformer, GatedGNN, query embeddings, classifier
- ASL and contrastive-loss equations
- optimizer, scheduler, AMP, and EMA update formulas
- Task 2 dataset mapping and validation behavior

## Completion Criteria

- All new tests exhibit RED before implementation and GREEN afterward.
- The existing 44 CPU tests remain green.
- No test split is constructed in the tobacco Trainer call chain.
- Only finite EMA validation mAP drives strict best selection.
- Checkpoint and metadata normal saves verify as a matching pair; injected failures restore the prior pair or report an integrity failure.
- Config can be safely reloaded with all tobacco protocol fields and legacy keys intact.
- Git shows no changes to `evaluate.py` or CoCo core algorithms beyond prior Task 2/3 work.
- GPU, AMP, real train/val image access, and formal training are reported as not verified.
