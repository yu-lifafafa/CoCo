# CoCo Tobacco GPU Smoke and Deployment Design

Date: 2026-10-10

## Purpose

Prepare the current uncommitted CoCo 18-class tobacco implementation for transfer from Windows to a Linux RTX 4090 D server. Add a bounded, auditable GPU smoke runner that exercises the real ResNet101 CoCo model for a few optimizer steps, plus CPU-tested deployment-package tooling and Linux instructions. This phase does not run GPU computation, formal training, validation calibration, or test evaluation.

## Baseline and Scope

- Branch: `feat/tobacco-coco-18cls`.
- Official-derived baseline HEAD: `342c4e35b19fc773ae71565dbe17be70d56bfcdc`.
- Task 2-5 changes remain uncommitted and must be included in deployment.
- Expected manifest SHA-256: `ce51205b537c3d1eee28b3d8112ce1edb6454c6f16cd3b441387bff2be2c56b5`.
- Task protocol: 18 labels, source ID 13 excluded, train/val/test splits preserved.
- Server GPU, driver, CUDA runtime, Python environment, installed packages, image paths, and cached weights are not verified locally.
- No model architecture, ASL, contrastive construction, optimizer default, scheduler default, EMA formula, dataset mapping, or training-selection rule changes are authorized.

## Repository Evidence

The official repository has no `requirements.txt`; README states only `torch >= 1.12.0` and `torchvision >= 0.13.0`. The actual import and execution chain additionally requires NumPy, Pillow, PyYAML, tqdm, tensorboard, scikit-learn for Task 5, and `randaugment` for training transforms. SciPy is to be checked against the final import inventory but is not declared required without a live import path.

`train.Trainer` calls `build_training_datasets(cfg)`. The tobacco branch creates only train and val datasets. EMA checkpoint selection uses finite EMA val VOC2012 mAP and strict improvement. The healthy-label contrastive fix skips all-zero targets while retaining classification loss.

`models.factory.create_backbone()` always requests pretrained initialization. `models.resnet._resnet()` uses `torch.utils.model_zoo.load_url()` with the official PyTorch ResNet101 URL and loads all non-`fc` keys with `strict=False`. This can access the network when the corresponding Torch Hub cache file is absent.

Importing `models.factory` executes `models/__init__.py`, which unconditionally imports TResNet and can require the compiled `inplace_abn` package even when the selected backbone is ResNet101. This is a deployment compatibility risk unrelated to ResNet101 computation.

## Chosen Architecture

Use an independent smoke runner rather than modifying `Trainer.run()`. It reuses the production Dataset, official transforms, real MLIC model, loss outputs, optimizer helper, scheduler helper, and EMA implementation, but owns its bounded sample selection, step loop, result schema, and output directory. This avoids formal experiment creation, validation, best-checkpoint writes, early stopping, and cleanup behavior.

Package the current source as a deterministic full source snapshot built from all tracked repository files plus an explicit allowlist of required untracked Task 2-6A files. A full snapshot avoids losing untracked files, which a patch-only deployment would do. It extracts into a new server directory rather than overwriting an existing worktree.

## Files

Planned additions:

- `scripts/smoke_tobacco_coco.py`: CLI, preflight, sample selection, real GPU steps, and result writing.
- `tests/test_smoke_tobacco_coco.py`: CPU tests for all control and failure paths.
- `scripts/build_deployment_package.py`: deterministic source snapshot builder.
- `deploy/verify_deployment.py`: archive and extracted-tree verifier.
- `tests/test_deployment_package.py`: package inclusion, exclusion, corruption, and path-safety tests.
- `deploy/server_setup.md`: Linux setup and Task 6B commands.
- `deploy/environment.yml`: non-CUDA Python dependencies and documented PyTorch installation boundary.
- `deploy/deployment_manifest.json`: generated package metadata.
- `deploy/coco_tobacco_deploy.zip`: generated verified package.

Potential minimal modification:

- `models/__init__.py` and, only if necessary for a clear error, `models/factory.py`: isolate the optional TResNet `inplace_abn` dependency. This change occurs only after a RED test proves ResNet101 import is blocked without `inplace_abn`. With the dependency installed, legacy TResNet registration remains unchanged. Selecting unavailable TResNet must produce an explicit dependency error.

No change is planned for `train.py`, `models/mlic.py`, `models/resnet.py`, Task 2 Dataset mapping, Task 4 checkpoint selection, or Task 5 evaluation logic.

## Smoke CLI

The script accepts:

- `--manifest-path PATH` (required)
- `--data-root PATH` (required)
- `--img-size INT` (default `384`, positive and divisible by 32)
- `--batch-size INT` (default `1`, positive)
- `--device DEVICE` (default `cuda:0`)
- mutually exclusive `--amp` / `--no-amp` (default AMP enabled)
- `--seed INT` (default `42`; explicitly a smoke choice, not the official default)
- `--backbone-init {torch-cache,local-file,allow-download}` (default `torch-cache`)
- `--backbone-weights PATH` (required only for `local-file`)
- `--output-dir PATH` (required)
- `--max-steps INT` (default `1`, hard maximum `8`)
- `--scenarios` containing a non-empty subset of `healthy,single,multi,mixed`

The script fixes `model=mlic`, `arch=resnet101`, `data=tobacco`, and `num_classes=18`. It creates the remaining MLIC, ASL, optimizer, scheduler, and EMA configuration from the official defaults, except `max_epochs=1` and scheduler step count are bounded to the smoke execution. It refuses unknown scenarios, duplicate scenarios, unsupported devices, non-CUDA devices, and output paths inside `experiments/` or any formal checkpoint directory.

`--max-steps` is a global optimizer-step ceiling. Requested scenarios beyond the ceiling remain `NOT_VERIFIED`; the script never continues into an epoch. A scenario receives at most one optimizer step per invocation.

## Pretrained Backbone Modes

The smoke result distinguishes initialization evidence from computation evidence.

### `torch-cache`

Require the exact standard Torch Hub cache member implied by the official ResNet101 URL. Record its absolute path and SHA-256 before model construction. If it is absent, stop with `NOT_VERIFIED`; do not initiate a download.

### `local-file`

Require an explicit readable file. Compute its SHA-256, create an isolated Torch Hub cache under the smoke output directory, copy the file to the official ResNet101 cache filename, set `TORCH_HOME` before model construction, and confirm the cached copy hash. The model then follows the repository's unchanged `model_zoo.load_url()` and non-`fc`, `strict=False` loading path.

### `allow-download`

Use the repository's official URL behavior and clearly record that network access was allowed. After model creation, require the resulting cache file and record its SHA-256. A successful forward pass does not by itself prove pretrained loading; the cache evidence is mandatory.

There is no random-initialization success mode. The smoke script does not accept an official full CoCo checkpoint or another baseline checkpoint as a substitute for the ImageNet backbone.

## Deterministic Train Sample Selection

Selection reads only manifest rows with `split=train`. It never opens or selects val/test records. Dataset construction and image reads use `TobaccoManifestDataset` and `build_coco_transform(cfg, training=True)`.

Categories are:

- `healthy`: `record.is_healthy is True`, `source_positive_count == 0`, and the 18-dimensional target sum is zero.
- `single`: `task_positive_count == 1`.
- `multi`: `task_positive_count >= 2`.
- `mixed`: exact batch size, containing at least one sample from each of healthy, single, and multi, with remaining slots filled deterministically from the union without duplicates.

Root-knot-only records can have an 18-dimensional all-zero target but are not called healthy because their source labels are not healthy. They may not fill the healthy scenario.

Candidate records are sorted by normalized manifest `file_name`; the seed determines any stable secondary sampling permutation. No row is duplicated to reach a requested batch size. Insufficient category counts or `batch_size < 3` for mixed return `NOT_VERIFIED` with a reason.

## GPU Step

For each selected batch the runner:

1. reads every image through Pillow and verifies RGB conversion;
2. applies the official training transform and verifies `[B,3,H,W]` with configured H/W;
3. verifies float32 targets shaped `[B,18]`;
4. transfers inputs and targets to the selected CUDA device;
5. resets CUDA peak-memory statistics and synchronizes;
6. executes the real MLIC forward under the requested autocast mode;
7. verifies logits `[B,18]`, finite classification loss, finite contrastive loss, and finite total loss;
8. for healthy batches, requires contrastive loss to equal zero;
9. uses a real GradScaler when AMP is enabled, performs backward, verifies finite trainable gradients, performs optimizer step and scaler update;
10. verifies at least one tracked parameter changed;
11. advances the real scheduler and verifies its step counter;
12. updates the real EMA and verifies the tracked EMA parameter changed consistently;
13. synchronizes CUDA and records elapsed step time plus peak allocated/reserved memory.

CUDA OOM is caught only to record a failed scenario and actionable diagnostics; the runner does not automatically reduce resolution, alter architecture, or continue after corrupted CUDA state. A 384 OOM recommendation is to retry a smaller batch first. Resolution changes require a separate invocation.

## Result Contract

Write a unique JSON result and text log under the smoke output directory through temporary files followed by `os.replace`. Never create or overwrite `best_ema_model.pth`, `best_checkpoint_info.json`, training logs, or Task 5 evaluation artifacts.

Top-level and per-scenario status values are exactly:

- `PASS`: every required runtime assertion for the executed scenario passed.
- `FAIL`: execution occurred and a runtime assertion, image read, dependency, CUDA, OOM, or optimizer operation failed.
- `NOT_VERIFIED`: the scenario was not executed, including GPU unavailable, step ceiling reached, or insufficient samples.

When CUDA work did not execute, GPU model, memory, timing, loss, backward, optimizer, scheduler, and EMA success fields are null or false as appropriate. They are never populated with synthetic CPU values.

Recorded provenance includes manifest hash, ordered mapping, source IDs, selected stable identities, Python/Torch/torchvision/CUDA versions, device name, driver-visible capability, model/backbone, input size, batch size, AMP, seed, initialization mode, weight path/hash, and script/source version identifiers.

## CPU Tests

CPU tests exercise real manifest parsing and selection with temporary CSV/images, plus pure CLI, validation, package, and status logic. They do not instantiate a fake CoCo model and do not report GPU success.

Required cases include:

- CLI defaults and all supported arguments;
- missing required paths and manifest SHA mismatch;
- train-only selection, with a trap on val/test access;
- healthy, single, multi, and mixed selection;
- root-knot-only not classified as healthy;
- target dtype and 18-dimensional shape;
- configured resolution;
- exact batch-size requirements and hard step ceiling;
- unavailable CUDA gives `NOT_VERIFIED` and null GPU metrics;
- unsafe output path rejection;
- result schema differentiating PASS/FAIL/NOT_VERIFIED;
- pretrained-mode constraints and local-file hash provenance;
- ResNet101 import without `inplace_abn`, while unavailable TResNet fails explicitly;
- no call path capable of writing a formal checkpoint.

Tests for the package builder cover deterministic member ordering, expected Task 2-6A files, exclusion of data/checkpoints/caches/secrets, baseline commit metadata, archive path traversal rejection, missing/extra member rejection, byte corruption, source-tree mismatch, and ZIP SHA mismatch.

## Dependency Plan

Create a Python 3.10 Conda environment. Do not encode a speculative CUDA toolkit build into `environment.yml`. The server guide first records `nvidia-smi`, driver version, and supported CUDA, then selects a PyTorch/torchvision pair from the official PyTorch compatibility instructions.

Confirmed runtime dependencies:

- Python
- PyTorch and torchvision, mutually compatible and CUDA-enabled on the server
- NumPy
- Pillow
- PyYAML
- tqdm
- tensorboard
- scikit-learn
- `randaugment`

Conditional or audit dependencies:

- `inplace_abn`: needed only for TResNet after optional-import isolation; not required for ResNet101 smoke.
- `timm`: repository vendors its own historical `models/timm_models`; an external `timm` package is not declared unless final import tracing proves a live import.
- SciPy: not declared required unless final import tracing proves a live runtime import.

Compatibility risks to document:

- README minimum versions are not a verified modern 4090 D environment.
- `torch.cuda.amp` APIs are deprecated in current Torch but remain the official training path; server smoke verifies actual behavior rather than silently rewriting training AMP.
- `inplace_abn` is a compiled extension with high version sensitivity.
- `randaugment` package naming/API must supply `from randaugment import RandAugment`.
- Python 3.12 and NumPy 2.x may expose legacy-package incompatibilities; Python 3.10 is recommended.
- Automatic ImageNet download occurs unless smoke mode proves a cache/local file.

## Deployment Package

The builder creates a deterministic ZIP from:

1. every path returned by `git ls-files`, using current working-tree bytes;
2. an explicit required allowlist for untracked Task 2-6A source, tests, specs, plans, verifier, and server documentation.

It rejects missing required paths, symlinks, absolute/archive-traversal names, duplicate normalized names, unreadable files, and changes to the expected baseline HEAD. It never relies on `git diff` alone.

Excluded categories include `.git`, `.agents`, `.codex`, `.aws`, data/images/manifests, experiments, checkpoints/weights, outputs, caches, virtual/Conda environments, coverage files, temporary test directories, secrets, and existing deployment archives/manifests.

`deployment_manifest.json` records:

- schema version;
- creation timestamp in UTC;
- baseline commit and branch as audit information;
- dirty tracked/untracked source state summary;
- ordered archive member list;
- each member's relative path, byte size, and SHA-256;
- total member count;
- archive filename, byte size, and SHA-256;
- expected tobacco manifest SHA-256 without including the manifest itself.

`verify_deployment.py` verifies the outer ZIP hash, safe and exact member set, per-member bytes, and optionally an extracted source tree. It also verifies the target repository's baseline commit when deployment uses an official clone.

The manifest is stored beside the ZIP rather than inside it, avoiding circular self-hashing. `server_setup.md` and verifier hashes are themselves listed as archive members.

## Linux Deployment Flow

The server guide uses Bash variables rather than assumed absolute paths:

```bash
COCO_WORKDIR=
PACKAGE_DIR=
MANIFEST_PATH=
DATA_ROOT=
BACKBONE_WEIGHTS=
SMOKE_OUTPUT=
CUDA_DEVICE=0
```

Flow:

1. inspect and back up any existing worktree; refuse to overwrite uncommitted changes;
2. create a new empty CoCo directory or clone the official repository and check out the exact baseline commit;
3. verify package ZIP and metadata before extraction;
4. extract into a new staging directory and verify every extracted file;
5. create the independent Conda environment;
6. install a server-compatible PyTorch/torchvision pair only after driver inspection;
7. install and import-check the remaining dependencies;
8. verify manifest SHA and all train/val paths, without changing data;
9. verify the ResNet101 ImageNet weight source/cache/hash;
10. run bounded batch-size 1 and 2 scenarios, followed by batch-size 4 including mixed;
11. stop on dependency failure, image failure, non-finite values, OOM, gradient failure, optimizer/scaler failure, scheduler failure, or EMA failure;
12. retain smoke JSON/logs for Task 6B review; do not start formal training automatically.

## Validation and Completion

Local completion requires:

- existing 136 CPU tests still pass;
- all new smoke/package CPU tests pass;
- every Python file in the deploy snapshot compiles;
- `git diff --check` passes;
- package build succeeds twice with identical source-member hashes and deterministic ZIP bytes;
- archive and extracted-tree verification pass;
- a deliberately corrupted package fails verification;
- documented CLI flags match parser help;
- Git branch and HEAD are unchanged and no commit/push occurred.

Local completion does not change these statuses:

- server GPU smoke: `NOT RUN`;
- CUDA/AMP/4090 D compatibility: `NOT VERIFIED`;
- server train/val path completeness: `NOT VERIFIED`;
- actual ImageNet weight availability/hash: `NOT VERIFIED` until a server file is supplied;
- formal training, val calibration, and test evaluation: `NOT RUN`.

Task 6B readiness means the package, verifier, commands, and smoke runner are locally ready to execute. It does not mean the GPU or data have passed.
