# CoCo Tobacco Training Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the 18-class tobacco manifest dataset to the official CoCo Trainer with strict train/val/test isolation, finite EMA-val selection, auditable checkpoint metadata, and safely reloadable configuration.

**Architecture:** Add a focused `lib/training_protocol.py` containing dataset construction and pure protocol/checkpoint helpers, then minimally wire the existing Trainer and environment preparation to it. Keep the official forward, ASL plus contrastive aggregation, AMP, optimizer, scheduler, EMA update, transforms, and legacy TXT dataset convention unchanged.

**Tech Stack:** Python 3, PyTorch, torchvision, NumPy, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-coco-tobacco-training-protocol-design.md`

## Global Constraints

- Work on `feat/tobacco-coco-18cls`; preserve all uncommitted Task 2 and Task 3 files.
- Do not modify `evaluate.py`, the tobacco 18-class mapping, CoCo architecture, ASL, contrastive math, optimizer/scheduler defaults, EMA formula, or official augmentation sequence.
- Do not run CUDA, formal training, inference, dependency installation, or downloads.
- Do not commit or push; commit steps are intentionally omitted from this plan.
- Tobacco uses only manifest `train` and `val`; legacy datasets keep official `train.txt` and `test.txt` behavior.
- Do not claim two independent file replacements are absolutely transaction-atomic; verify consistency with SHA-256 and report crash limitations.

## Review Focus

- Existing best checkpoint with missing/corrupt metadata: integrity verification must reject the pair before it is trusted; covered in Task 3.
- A failure after checkpoint replacement but before metadata replacement: prior checkpoint and metadata must both be restored; covered in Task 3.
- Relative and Windows-style manifest/data-root paths round-tripped through YAML: restored values must remain strings and resolve consistently; covered in Task 2.
- A tobacco manifest containing test records whose images are unavailable: Trainer construction must not open test images; covered in Tasks 1 and 4.
- A validation metric represented as NumPy scalar rather than Python float: finite checking and comparison must behave identically; covered in Task 3.

---

### Task 1: Dataset construction and tobacco protocol constants

**Files:**
- Create: `lib/training_protocol.py`
- Create: `tests/test_training_protocol.py`

**Interfaces:**
- Consumes: `TobaccoManifestDataset`, `SOURCE_LABEL_COLUMNS`, `MODEL_LABEL_COLUMNS`, `EXCLUDED_SOURCE_ID`, `MLDataset`, and `build_coco_transform`.
- Produces: `TOBACCO_NUM_CLASSES`, `tobacco_label_mapping() -> list[dict]`, and `build_training_datasets(cfg, tobacco_dataset_cls=TobaccoManifestDataset, legacy_dataset_cls=MLDataset) -> tuple[Dataset, Dataset]`.

- [ ] **Step 1: Write failing dataset-protocol tests**

Add tests asserting ordered source IDs/model indices, fixed 18 classes, tobacco construction calls exactly `train` and `val`, a constructor trap raises on `test`, test-image absence is not accessed, and legacy calls remain `(cfg.train_path, True)` plus `(cfg.test_path, False)`.

- [ ] **Step 2: Run the focused tests and record RED**

Run: `python -m pytest tests/test_training_protocol.py -q -p no:cacheprovider --basetemp <workspace-temp>`

Expected: collection/import failure because `lib.training_protocol` does not exist.

- [ ] **Step 3: Implement the minimal protocol constants and dataset builder**

The tobacco branch must validate `cfg.num_classes == 18`, use the shared official transforms, instantiate no other split, and return exactly `(train, val)`. The legacy branch must preserve official path and `training` arguments.

- [ ] **Step 4: Run focused tests GREEN**

Expected: all Task 1 tests pass.

### Task 2: CLI, safe configuration, and metrics compatibility

**Files:**
- Modify: `lib/utils.py`
- Modify: `lib/metrics.py`
- Modify: `tests/test_training_protocol.py`

**Interfaces:**
- Consumes: Task 1 mapping and SHA helper.
- Produces: CLI `--manifest-path`/`--data-root`; `load_config(path) -> argparse.Namespace`; tobacco-aware `prepare_env`; `VOCmAP(..., ignore_path=None)` support.

- [ ] **Step 1: Write failing CLI/config/metrics tests**

Cover parsing tobacco arguments, missing manifest/data-root, `num_classes=18`, absence of fake TXT paths, complete tobacco protocol fields, exact manifest SHA-256, safe YAML round trip for paths/model/training arguments/mapping, representative legacy config behavior, `ignore_path=None`, and unchanged real ignore-mask AP behavior.

- [ ] **Step 2: Run the focused tests and record RED**

Expected failures: unknown CLI options, official TXT checks on tobacco, missing safe loader/protocol fields, and `os.path.exists(None)` TypeError.

- [ ] **Step 3: Implement minimal environment and metric branches**

Use `yaml.safe_dump`/`yaml.safe_load`. Preserve existing official default values and legacy path derivation. For tobacco require a file manifest and directory root, calculate SHA-256, set `ignore_path=None`, and serialize only safe built-in values.

- [ ] **Step 4: Run focused tests GREEN**

Expected: all Task 1–2 tests pass.

### Task 3: EMA selection and checkpoint/metadata consistency

**Files:**
- Modify: `lib/training_protocol.py`
- Modify: `tests/test_training_protocol.py`

**Interfaces:**
- Produces: `validate_finite_map(name, value) -> float`, `is_strict_improvement(candidate, best) -> bool`, `save_best_ema_checkpoint(ema_model, cfg, epoch, val_map, replace_fn=os.replace, ...) -> dict`, `verify_best_checkpoint_integrity(checkpoint_path, metadata_path) -> dict`, and `sha256_file(path) -> str`.
- The save helper persists exactly `ema_model.state_dict()` and returns verified metadata only after both final files match.

- [ ] **Step 1: Write failing selection and save tests**

Cover first legal zero, strict improvement, lower score, tie, Python/NumPy NaN and Inf, metadata completeness, ordered mapping, manifest hash, epoch/score correspondence, unchanged EMA state-dict reload, and final checkpoint SHA match.

- [ ] **Step 2: Write failing fault-injection tests**

Inject checkpoint write failure, JSON write failure, first replace failure, second replace failure, and rollback failure boundaries. With an existing epoch-10 pair, any recoverable epoch-15 failure must leave the epoch-10 pair byte-for-byte and metadata-consistent. Corrupt or partially replaced pairs must be rejected by integrity verification.

- [ ] **Step 3: Run Task 3 tests and record RED**

Expected: missing selection/save/integrity interfaces.

- [ ] **Step 4: Implement the save and rollback protocol**

Use unique same-directory temporary and backup files, candidate reload, SHA calculation, metadata reload, ordered replacement, final verification, and exception rollback. Update no Trainer best state inside the helper. Preserve artifacts and chain errors if rollback cannot complete.

- [ ] **Step 5: Run Task 3 tests GREEN**

Expected: normal save, reload, integrity, and all fault-injection tests pass.

### Task 4: Minimal real Trainer integration

**Files:**
- Modify: `train.py`
- Modify: `tests/test_training_protocol.py`

**Interfaces:**
- Consumes: `build_training_datasets`, finite metric validation, strict comparison, checkpoint save helper.
- Preserves: `Trainer.train`, model forward, loss aggregation, AMP, optimizer, scheduler, and EMA update logic except for safe log interval calculation.

- [ ] **Step 1: Write failing real-constructor and selection tests**

Execute the actual `Trainer.__init__()` with substituted model/CUDA/writer dependencies. Assert it calls the protocol builder once, receives only train/val, creates no test loader, and initializes best score/epoch safely. Add a builder trap for any test access.

- [ ] **Step 2: Write failing run-loop tests**

Use a Trainer instance with simulated `train()` and `validation()` results to verify only finite EMA val mAP triggers saving, first zero saves, improvement replaces, lower/tie does not replace, patience uses strict improvement, and saved epoch/metric match. Do not expose a test metric argument to the selector.

- [ ] **Step 3: Write failing operational-boundary tests**

Cover `max(1, len(loader)//6)` and tobacco exception cleanup policy while preserving legacy cleanup behavior.

- [ ] **Step 4: Run Task 4 tests and record RED**

Expected: Trainer still constructs `MLDataset` directly, initializes best to zero, directly calls `torch.save`, and has unsafe log modulo/cleanup behavior.

- [ ] **Step 5: Apply the minimal Trainer wiring**

Call the protocol builder from the constructor; keep both validation metrics and return EMA mAP; reject non-finite metrics; initialize best as `None`; save only strict EMA improvements; persist verified metadata; use a safe log interval; retain tobacco failure evidence without changing legacy cleanup.

- [ ] **Step 6: Run Task 4 tests GREEN**

Expected: real-constructor, run-loop, isolation, and operational tests pass without CUDA.

### Task 5: Regression and final verification

**Files:**
- Test only; production changes only if a failure exposes an in-scope defect and a new RED test is added first.

**Interfaces:**
- Consumes all Task 1–4 outputs.
- Produces verification evidence and final diff audit.

- [ ] **Step 1: Run all Task 4 tests**

Run: `python -m pytest tests/test_training_protocol.py -q -p no:cacheprovider --basetemp <workspace-temp>`

Expected: all pass.

- [ ] **Step 2: Run the complete CPU suite**

Run: `python -m pytest tests -q -p no:cacheprovider --basetemp <workspace-temp>`

Expected: existing 44 tests plus all Task 4 tests pass.

- [ ] **Step 3: Run static and Git checks**

Run Python compilation for changed Python files, `git diff --check`, `git status --short`, and targeted diffs. Confirm `evaluate.py`, Task 2 mapping, Task 3 contrastive behavior, and CoCo architecture are unchanged by Task 4.

- [ ] **Step 4: Report verification limits**

Record GPU, AMP, real train/val images, full official dependency stack, and formal training as `NOT VERIFIED`. Report test doubles as CPU integration evidence only.
