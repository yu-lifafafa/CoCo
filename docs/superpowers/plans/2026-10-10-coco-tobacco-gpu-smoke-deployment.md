# CoCo Tobacco GPU Smoke and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a CPU-tested, strictly bounded real-CoCo GPU smoke runner and a deterministic, verifiable Linux deployment package for Task 6B without running GPU work or formal training locally.

**Architecture:** Add a standalone smoke runner that reuses production Dataset, transforms, MLIC, optimizer, scheduler, GradScaler, and EMA while owning deterministic train-only sampling and a hard step ceiling. Add independent deterministic package-builder/verifier tooling that snapshots all tracked files plus an explicit Task 2-6A untracked allowlist, then document a new-directory Linux deployment flow.

**Tech Stack:** Python 3, PyTorch, torchvision, NumPy, Pillow, PyYAML, CSV/JSON, zipfile, hashlib, pytest, Linux Bash documentation.

**Spec:** `docs/superpowers/specs/2026-10-10-coco-tobacco-gpu-smoke-deployment-design.md`

## Global Constraints

- Preserve all uncommitted Task 2-5 work on `feat/tobacco-coco-18cls` at baseline HEAD `342c4e35b19fc773ae71565dbe17be70d56bfcdc`.
- Do not commit, push, merge, install packages, run CUDA work, start training, or perform real val/test inference.
- Do not modify the tobacco 18-class mapping, split assignments, MLIC architecture, ASL, contrastive mathematics, optimizer/scheduler defaults, EMA formula, checkpoint selection, or Task 5 evaluation protocol.
- The smoke runner uses only manifest `train` records and never creates or reads a val/test Dataset.
- `--max-steps` is a global hard ceiling in `[1,8]`; one scenario can consume at most one step.
- A GPU-unavailable local run must report `NOT_VERIFIED` with null runtime metrics, never a synthetic PASS.
- Deployment outputs exclude data, manifests, images, checkpoints, experiments, caches, environments, secrets, and existing archives.
- Generated deployment metadata records expected manifest SHA but never packages the real manifest.
- Server paths and dependency versions remain variables until Task 6B verifies them.

## Review Focus

- A root-knot-only train row has an 18-dimensional zero target but must not be selected as an original healthy sample; Task 2 selection tests pin this.
- `batch_size < 3` cannot satisfy mixed composition and must return `NOT_VERIFIED`, not duplicate rows; Task 2 tests pin this.
- An absent cache in `torch-cache` mode must not reach `model_zoo.load_url` and accidentally download; Task 3 tests pin preflight ordering.
- A ZIP member using absolute or `..` traversal must be rejected before extraction; Task 4 tests pin archive safety.
- A server directory with uncommitted changes must not be overwritten by documented deployment commands; Task 5 documentation checks pin the clean/new-directory flow.

---

### Task 1: Baseline Audit and Optional TResNet Dependency Isolation

**Files:**
- Modify only if RED proves necessary: `models/__init__.py`
- Modify only if required for an explicit selection error: `models/factory.py`
- Create or modify: `tests/test_smoke_tobacco_coco.py`

**Interfaces:**
- Produces: ResNet101 registration/import works without external `inplace_abn`; TResNet remains registered when the dependency is available and fails explicitly when selected without it.
- Preserves: every existing backbone/model entrypoint when its dependencies are installed.

- [ ] **Step 1: Record the read-only baseline**

Run branch, HEAD, complete status, tracked/untracked inventory, `git diff --check`, Python compile, and full existing test suite. Save command results in the execution ledger, not a product file.

- [ ] **Step 2: Write the failing optional-dependency tests**

Add tests that import the real `models` package while blocking `inplace_abn`, require `resnet101` and `mlic` entrypoints to remain available, then simulate a usable `inplace_abn` module and require TResNet registration. If explicit TResNet selection behavior needs factory support, assert the error names `inplace_abn` rather than returning a generic missing-key error.

- [ ] **Step 3: Run Task 1 RED**

Run: `python -m pytest tests/test_smoke_tobacco_coco.py -q -p no:cacheprovider --basetemp <task-temp>`

Expected: import fails through unconditional TResNet `inplace_abn` loading.

- [ ] **Step 4: Implement the smallest import isolation**

Guard only the TResNet import. Re-raise unrelated import errors. Preserve ResNet/timm/MLIC imports and legacy TResNet registration when `inplace_abn` imports successfully. Do not alter any model class or computation.

- [ ] **Step 5: Run Task 1 GREEN and the existing suite**

Expected: new dependency-isolation tests pass and all existing 136 tests pass.

### Task 2: Smoke CLI, Manifest Contract, and Train-Only Scenario Selection

**Files:**
- Create: `scripts/smoke_tobacco_coco.py`
- Modify: `tests/test_smoke_tobacco_coco.py`

**Interfaces:**
- Produces: `build_parser()`, `parse_args(argv=None)`, `validate_smoke_args(args, project_root)`, `load_train_smoke_dataset(args, dataset_cls=TobaccoManifestDataset)`, `select_scenario_indices(records, scenario, batch_size, seed)`, `new_smoke_result(args)`, and `finalize_status(result)`.
- Consumes: `TobaccoManifestDataset`, `build_coco_transform`, Task 2 record fields, expected 18-class mapping, and SHA helper.

- [ ] **Step 1: Write failing CLI and safety tests**

Cover every documented argument/default, missing manifest/data root/output, image size positivity and divisibility by 32, batch size positivity, CUDA-only device syntax, AMP toggles, `max_steps` range `[1,8]`, scenario validation/deduplication, backbone-mode/path constraints, and rejection of outputs under `experiments`, checkpoint directories, or formal artifact directories.

- [ ] **Step 2: Write failing manifest and selection tests**

Build temporary train/val/test manifests and images. Assert exact manifest SHA checking, only train Dataset construction, no val/test image read, deterministic selection, exact batch sizes, no duplicate indices, healthy source semantics, root-knot-only exclusion from healthy, single/multi counts after source ID 13 removal, mixed composition, insufficient categories, and mixed batch sizes below three returning `NOT_VERIFIED` reasons.

- [ ] **Step 3: Write failing status-schema tests**

Assert top-level and scenario states accept only `PASS`, `FAIL`, `NOT_VERIFIED`; an unexecuted/GPU-unavailable result has null GPU name, losses, memory, duration, and false execution flags.

- [ ] **Step 4: Run Task 2 RED**

Expected: smoke module or declared functions are missing.

- [ ] **Step 5: Implement minimal pure control logic**

Keep imports that require CUDA/model dependencies lazy so `--help`, parsing, and CPU selection tests run without CUDA extensions. Use stable normalized file names and a seeded local RNG without mutating global sampling state.

- [ ] **Step 6: Run Task 2 GREEN**

Expected: all CLI, manifest, selection, and status tests pass; existing suite remains green.

### Task 3: Pretrained Evidence and Real Bounded GPU Step Implementation

**Files:**
- Modify: `scripts/smoke_tobacco_coco.py`
- Modify: `tests/test_smoke_tobacco_coco.py`

**Interfaces:**
- Produces: `prepare_backbone_initialization(args, output_dir) -> dict`, `build_smoke_cfg(args) -> argparse.Namespace`, `validate_batch(batch, args)`, `execute_gpu_scenario(model, batch, runtime) -> dict`, `write_smoke_result(output_dir, result)`, `run_smoke(args) -> dict`, and `main(argv=None) -> int`.
- Consumes: real `create_model`, `get_optimizer`, `get_lr_scheduler`, `ModelEma`, official transform, MLIC return keys, and scenario selections from Task 2.

- [ ] **Step 1: Write failing pretrained-mode tests**

Cover absent `torch-cache` member stopping before model construction, `local-file` hash/copy into an isolated smoke Torch Hub cache, cache-copy hash equality, explicit `allow-download` provenance, and absence of any random-init PASS path. Use file-operation doubles only; do not mock a backbone or claim pretrained computation succeeded.

- [ ] **Step 2: Write failing GPU-unavailable and preflight-order tests**

Patch only the CUDA availability boundary to false and assert `run_smoke` returns `NOT_VERIFIED` before model construction, leaves all runtime metrics null, writes no checkpoint, and respects the step ceiling. Add a trap proving cache-only failure occurs before the real model factory is called.

- [ ] **Step 3: Write failing result/output tests**

Cover atomic JSON/log replacement, unique smoke-only filenames, output provenance, no formal checkpoint names, and handled failure preserving a readable FAIL/NOT_VERIFIED result.

- [ ] **Step 4: Run Task 3 RED**

Expected: pretrained, runtime, and output functions are missing.

- [ ] **Step 5: Implement real GPU execution path without running it**

Construct fixed MLIC/ResNet101/18-class config, real model, AdamW/official configured optimizer, OneCycleLR bounded to smoke steps, ModelEma, CUDA autocast, and GradScaler. Validate shapes/dtypes/devices, finite losses/gradients, healthy CTR zero, tracked parameter change, scheduler advancement, EMA change, CUDA synchronization, time, and peak memory. Catch OOM only to record failure and stop. Never save model weights.

- [ ] **Step 6: Run Task 3 GREEN on CPU control paths**

Expected: control tests pass; no test reports a GPU scenario PASS; full suite remains green.

### Task 4: Deterministic Deployment Builder and Verifier

**Files:**
- Create: `scripts/build_deployment_package.py`
- Create: `deploy/verify_deployment.py`
- Create: `tests/test_deployment_package.py`

**Interfaces:**
- Produces: `collect_snapshot_files(repo_root, required_untracked)`, `validate_member_name(name)`, `build_package(repo_root, archive_path, manifest_path, baseline_commit)`, `load_deployment_manifest(path)`, `verify_archive(archive_path, manifest)`, `verify_extracted_tree(root, manifest)`, and CLIs for builder/verifier.
- Consumes: `git ls-files`, explicit Task 2-6A allowlist, SHA-256 helper, deterministic ZIP metadata.

- [ ] **Step 1: Write failing snapshot inclusion/exclusion tests**

Use a temporary Git repository with tracked modifications and untracked required files. Assert current working bytes are packaged, required untracked files are included, missing required files fail, and data/manifests/images/checkpoints/experiments/caches/secrets/deploy archives are excluded.

- [ ] **Step 2: Write failing deterministic-package tests**

Build twice from unchanged inputs and assert identical ordered member lists, per-file hashes, archive bytes, and archive SHA. Assert baseline commit and expected external manifest SHA are recorded.

- [ ] **Step 3: Write failing verifier/security tests**

Cover ZIP hash mismatch, missing/extra members, member-byte corruption, absolute path, drive-qualified path, `..` traversal, duplicate normalized names, symlinks, extracted-tree missing/extra/changed files, and baseline commit mismatch.

- [ ] **Step 4: Run Task 4 RED**

Expected: builder and verifier interfaces are missing.

- [ ] **Step 5: Implement deterministic package tooling**

Use fixed ZIP timestamps, permissions, compression, sorted POSIX member names, temporary output files, reload verification, and `os.replace`. Store the manifest beside the ZIP to avoid circular hashing. Do not extract unsafe members.

- [ ] **Step 6: Run Task 4 GREEN**

Expected: all package and corruption tests pass; smoke and prior tests remain green.

### Task 5: Dependency Definition and Linux Server Runbook

**Files:**
- Create: `deploy/environment.yml`
- Create: `deploy/server_setup.md`
- Modify: `tests/test_smoke_tobacco_coco.py`
- Modify: `tests/test_deployment_package.py`

**Interfaces:**
- Produces: a Python 3.10 environment boundary, dependency/import audit commands, and exact Bash commands matching the smoke/verifier CLIs.
- Consumes: parser help and package verifier interfaces from Tasks 2-4.

- [ ] **Step 1: Write documentation-contract tests**

Parse smoke and verifier help and assert every option used by the runbook exists. Assert the runbook defines all path variables, checks baseline commit, refuses dirty overwrite, verifies package before/after extraction, checks manifest SHA and train/val files, audits pretrained cache/hash, uses batch 1/2/4 bounded commands, names output locations, and contains no formal training command execution.

- [ ] **Step 2: Write environment-contract tests**

Safely load `environment.yml` and assert Python 3.10 plus confirmed non-CUDA dependencies. Assert it does not pin a speculative CUDA toolkit or silently install torch from an unverified channel. Assert server instructions require driver inspection before selecting torch/torchvision.

- [ ] **Step 3: Run Task 5 RED**

Expected: environment and runbook files are missing.

- [ ] **Step 4: Write the environment and runbook**

Use Linux Bash variables `COCO_WORKDIR`, `PACKAGE_DIR`, `MANIFEST_PATH`, `DATA_ROOT`, `BACKBONE_WEIGHTS`, `SMOKE_OUTPUT`, and `CUDA_DEVICE`. Include dependency diagnostics, optional TResNet note, cache-only/local-file/download behavior, image completeness checks, batch 1/2/4 commands, failure stop conditions, and Task 6B evidence handoff.

- [ ] **Step 5: Run Task 5 GREEN**

Expected: docs/environment contract tests pass and CLI commands parse without GPU execution.

### Task 6: Generate and Verify the Real Deployment Package

**Files:**
- Generate: `deploy/coco_tobacco_deploy.zip`
- Generate: `deploy/deployment_manifest.json`
- Test/review all Task 6A files.

**Interfaces:**
- Consumes all Tasks 1-5.
- Produces the final archive, manifest, hashes, local evidence, and Task 6B readiness report.

- [ ] **Step 1: Run the new targeted suites**

Run smoke and deployment tests with isolated temporary directories.

Expected: all new CPU tests pass with no GPU work.

- [ ] **Step 2: Run the complete CPU suite**

Run: `python -m pytest tests -q -p no:cacheprovider --basetemp <task-temp>`

Expected: the existing 136 tests plus all new tests pass.

- [ ] **Step 3: Run static checks**

Compile every Python file selected for deployment, run `git diff --check`, inspect branch/HEAD/status, and confirm no unauthorized Task 2-5 core file changed.

- [ ] **Step 4: Build the real package twice**

Build to two temporary archive/manifest pairs, compare archive SHA and bytes, then publish one verified pair to `deploy/`. Expected: deterministic equality and successful reload verification.

- [ ] **Step 5: Verify corruption detection**

Copy the published archive to a temporary location, flip one byte, and require verifier failure. Delete only the controlled corrupted temporary copy afterward.

- [ ] **Step 6: Verify an extracted tree**

Extract through the safe verifier into a temporary directory, verify exact file set and hashes, compile its Python files, and confirm the real manifest/data/checkpoints are absent.

- [ ] **Step 7: Final protocol review**

Review the diff against the spec, verify documented commands against live `--help`, record dependency and weight facts separately from server unknowns, and explicitly retain `SERVER GPU TEST: NOT RUN` and `SERVER DATA CHECK: NOT RUN`.

- [ ] **Step 8: Preserve the branch**

Do not commit or push. Remove only Task 6A test/package temporary directories. Report the complete dirty status and package paths/hashes.
