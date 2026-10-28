"""Bounded real-model GPU smoke runner for the 18-class tobacco task.

The module keeps CUDA/model imports lazy so its CLI and protocol controls can be
tested on CPU-only developer machines.  It never saves model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_MANIFEST_SHA256 = (
    "ce51205b537c3d1eee28b3d8112ce1edb6454c6f16cd3b441387bff2be2c56b5"
)
ALLOWED_SCENARIOS = ("healthy", "single", "multi", "mixed")
ALLOWED_STATUSES = ("PASS", "FAIL", "NOT_VERIFIED")
MAX_STEP_LIMIT = 8
RESNET101_WEIGHT_FILENAME = "resnet101-63fe2227.pth"
RESNET101_WEIGHT_URL = (
    "https://download.pytorch.org/models/resnet101-63fe2227.pth"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded real-CoCo tobacco GPU smoke test."
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone-init",
        choices=("torch-cache", "local-file", "allow-download"),
        default="torch-cache",
    )
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=ALLOWED_SCENARIOS,
        default=["healthy"],
    )
    parser.add_argument(
        "--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _is_forbidden_output(output_dir: Path, project_root: Path) -> bool:
    try:
        relative_parts = tuple(
            part.lower() for part in output_dir.resolve().relative_to(
                project_root.resolve()
            ).parts
        )
    except ValueError:
        return False
    if "experiments" in relative_parts or "checkpoints" in relative_parts:
        return True
    return relative_parts[:3] == ("outputs", "tobacco_coco", "evaluation") or (
        relative_parts[:3] == ("outputs", "tobacco_coco", "calibration")
    )


def validate_smoke_args(args, project_root: Path) -> dict:
    manifest_path = Path(args.manifest_path)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest file does not exist: {manifest_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root directory does not exist: {data_root}")
    if args.img_size <= 0:
        raise ValueError("image size must be positive")
    if args.img_size % 32:
        raise ValueError("image size must be divisible by 32")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if re.fullmatch(r"cuda:\d+", args.device) is None:
        raise ValueError("device must be an explicit CUDA device such as cuda:0")
    if not 1 <= args.max_steps <= MAX_STEP_LIMIT:
        raise ValueError(f"max steps must be between 1 and {MAX_STEP_LIMIT}")
    if len(args.scenarios) != len(set(args.scenarios)):
        raise ValueError("scenarios must not contain duplicate entries")
    if len(args.scenarios) > args.max_steps:
        raise ValueError("scenario count cannot exceed max steps")
    if args.backbone_init == "local-file":
        if args.backbone_weights is None:
            raise ValueError("local-file mode requires --backbone-weights")
        if not Path(args.backbone_weights).is_file():
            raise FileNotFoundError(
                f"backbone weights do not exist: {args.backbone_weights}"
            )
    elif args.backbone_weights is not None:
        raise ValueError("--backbone-weights is valid only with local-file mode")
    if _is_forbidden_output(output_dir, Path(project_root)):
        raise ValueError("smoke output cannot target formal experiment artifacts")
    actual_sha256 = _sha256_file(manifest_path)
    expected_sha256 = str(args.expected_manifest_sha256).lower()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "manifest SHA-256 mismatch: expected {}, got {}".format(
                expected_sha256, actual_sha256
            )
        )
    return {
        "manifest_sha256": actual_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "data_root": str(data_root.resolve()),
        "output_dir": str(output_dir.resolve()),
    }


def load_train_smoke_dataset(args, transform, dataset_cls=None):
    if dataset_cls is None:
        from lib.tobacco_dataset import TobaccoManifestDataset

        dataset_cls = TobaccoManifestDataset
    return dataset_cls(
        args.manifest_path,
        args.data_root,
        split="train",
        transform=transform,
    )


def _not_verified(reason: str) -> dict:
    return {"status": "NOT_VERIFIED", "indices": [], "reason": reason}


def select_scenario_indices(records, scenario: str, batch_size: int, seed: int) -> dict:
    if scenario not in ALLOWED_SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario}")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    healthy = [
        index
        for index, record in enumerate(records)
        if record.is_healthy and record.source_positive_count == 0
    ]
    single = [
        index
        for index, record in enumerate(records)
        if record.task_positive_count == 1
    ]
    multi = [
        index
        for index, record in enumerate(records)
        if record.task_positive_count >= 2
    ]
    candidates = {"healthy": healthy, "single": single, "multi": multi}
    rng = random.Random(f"{seed}:{scenario}:{batch_size}")

    if scenario != "mixed":
        pool = candidates[scenario]
        if len(pool) < batch_size:
            return _not_verified(
                f"scenario {scenario} has {len(pool)} eligible train records, "
                f"but batch size is {batch_size}"
            )
        return {
            "status": "PASS",
            "indices": rng.sample(pool, batch_size),
            "reason": None,
        }

    if batch_size < 3:
        return _not_verified("mixed scenario requires a batch size of at least 3")
    missing = [name for name, pool in candidates.items() if not pool]
    if missing:
        return _not_verified(
            "mixed scenario lacks eligible train records for: " + ", ".join(missing)
        )
    selected = [rng.choice(candidates[name]) for name in ("healthy", "single", "multi")]
    eligible = sorted(set(healthy + single + multi) - set(selected))
    remainder = batch_size - len(selected)
    if len(eligible) < remainder:
        return _not_verified(
            f"mixed scenario has only {len(selected) + len(eligible)} unique eligible "
            f"train records for batch size {batch_size}"
        )
    selected.extend(rng.sample(eligible, remainder))
    rng.shuffle(selected)
    return {"status": "PASS", "indices": selected, "reason": None}


def new_smoke_result(args) -> dict:
    return {
        "schema_version": 1,
        "status": "NOT_VERIFIED",
        "allowed_statuses": list(ALLOWED_STATUSES),
        "reason": "GPU smoke has not run",
        "gpu_executed": False,
        "gpu_name": None,
        "device": args.device,
        "image_size": args.img_size,
        "batch_size": args.batch_size,
        "amp": bool(args.amp),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "scenarios_requested": list(args.scenarios),
        "classification_loss": None,
        "contrastive_loss": None,
        "total_loss": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "duration_seconds": None,
        "scenarios": [],
    }


def finalize_status(result: dict) -> dict:
    scenario_statuses = [item["status"] for item in result.get("scenarios", [])]
    invalid = [status for status in scenario_statuses if status not in ALLOWED_STATUSES]
    if invalid:
        raise ValueError(f"invalid smoke status values: {invalid}")
    if any(status == "FAIL" for status in scenario_statuses):
        result["status"] = "FAIL"
    elif scenario_statuses and all(status == "PASS" for status in scenario_statuses):
        result["status"] = "PASS"
    else:
        result["status"] = "NOT_VERIFIED"
    return result


def _torch_cache_weight_path() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / RESNET101_WEIGHT_FILENAME


def prepare_backbone_initialization(args, output_dir) -> dict:
    """Fail closed or prepare an isolated cache before model construction."""
    output_dir = Path(output_dir)
    mode = args.backbone_init
    evidence = {
        "mode": mode,
        "architecture": "resnet101",
        "url": RESNET101_WEIGHT_URL,
        "network_allowed": mode == "allow-download",
        "source_path": None,
        "source_sha256": None,
        "resolved_path": None,
        "resolved_sha256": None,
    }
    if mode == "torch-cache":
        cached_path = _torch_cache_weight_path()
        if not cached_path.is_file():
            raise FileNotFoundError(
                "required cached ImageNet backbone weight is absent: "
                f"{cached_path}. Use local-file or allow-download explicitly."
            )
        evidence["resolved_path"] = str(cached_path.resolve())
        evidence["resolved_sha256"] = _sha256_file(cached_path)
        return evidence

    isolated_torch_home = output_dir / "torch_home"
    os.environ["TORCH_HOME"] = str(isolated_torch_home)
    destination = (
        isolated_torch_home / "hub" / "checkpoints" / RESNET101_WEIGHT_FILENAME
    )
    if mode == "local-file":
        source = Path(args.backbone_weights)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        source_sha256 = _sha256_file(source)
        destination_sha256 = _sha256_file(destination)
        if source_sha256 != destination_sha256:
            raise RuntimeError("copied backbone weight SHA-256 mismatch")
        evidence.update(
            {
                "source_path": str(source.resolve()),
                "source_sha256": source_sha256,
                "resolved_path": str(destination),
                "resolved_sha256": destination_sha256,
            }
        )
    elif mode != "allow-download":
        raise ValueError(f"unsupported backbone initialization mode: {mode}")
    return evidence


def build_smoke_cfg(args):
    """Return the fixed Task 6B real-CoCo smoke configuration."""
    return SimpleNamespace(
        model="mlic",
        arch="resnet101",
        num_classes=18,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        amp=bool(args.amp),
        num_heads=8,
        enc_layers=1,
        dec_layers=2,
        lamda=1.0,
        loss="asl",
        gamma_pos=0.0,
        gamma_neg=2.0,
        tau=1.0,
        num_steps=1,
        ema_decay=0.9997,
        optimizer="AdamW",
        lr=0.0001,
        weight_decay=0.0001,
        lr_scheduler="OneCycleLR",
        pct_start=0.2,
        max_epochs=1,
        mode="full",
    )


def cuda_is_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def build_real_smoke_model(cfg):
    from models.factory import create_model

    return create_model(cfg.model, cfg=cfg)


def validate_batch(batch, args, require_cuda=True):
    import torch

    images = batch["img"]
    targets = batch["target"]
    if images.ndim != 4 or tuple(images.shape[1:]) != (
        3,
        args.img_size,
        args.img_size,
    ):
        raise ValueError(
            f"expected image batch [B,3,{args.img_size},{args.img_size}], "
            f"got {tuple(images.shape)}"
        )
    if targets.shape != (images.shape[0], 18):
        raise ValueError(f"expected target batch [B,18], got {tuple(targets.shape)}")
    if targets.dtype != torch.float32:
        raise TypeError(f"expected float32 targets, got {targets.dtype}")
    if not torch.isfinite(images).all() or not torch.isfinite(targets).all():
        raise ValueError("batch contains non-finite values")
    if require_cuda and (not images.is_cuda or not targets.is_cuda):
        raise ValueError("smoke tensors must be on CUDA")


def _first_trainable_parameter(model):
    return next(parameter for parameter in model.parameters() if parameter.requires_grad)


def execute_gpu_scenario(model, batch, runtime) -> dict:
    """Execute exactly one real optimizer step for a selected scenario."""
    import torch

    args = runtime["args"]
    device = runtime["device"]
    optimizer = runtime["optimizer"]
    scheduler = runtime["scheduler"]
    scaler = runtime["scaler"]
    ema_model = runtime["ema_model"]
    scenario = runtime["scenario"]

    images = batch["img"].to(device, non_blocking=False)
    targets = batch["target"].to(device, non_blocking=False)
    cuda_batch = dict(batch)
    cuda_batch.update({"img": images, "target": targets})
    validate_batch(cuda_batch, args, require_cuda=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    parameter = _first_trainable_parameter(model)
    parameter_before = parameter.detach().clone()
    ema_parameter = _first_trainable_parameter(ema_model.module)
    ema_before = ema_parameter.detach().clone()
    scheduler_before = scheduler.last_epoch
    started = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=args.amp):
        output = model(images, targets)
        logits = output["logits"]
        classification_loss = output["bce"]
        contrastive_loss = output["ctr"]
        total_loss = classification_loss + runtime["cfg"].lamda * contrastive_loss
    if logits.shape != (images.shape[0], 18):
        raise ValueError(f"expected logits [B,18], got {tuple(logits.shape)}")
    for name, value in (
        ("classification loss", classification_loss),
        ("contrastive loss", contrastive_loss),
        ("total loss", total_loss),
    ):
        if value.ndim != 0 or not torch.isfinite(value):
            raise FloatingPointError(f"{name} is not a finite scalar")
    if scenario == "healthy" and not torch.allclose(
        contrastive_loss.detach(), contrastive_loss.new_zeros(()), atol=1e-7, rtol=0
    ):
        raise AssertionError("all-healthy contrastive loss must be zero")

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    gradients = [
        value.grad for value in model.parameters() if value.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("model gradients are missing or non-finite")
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    ema_model.update(model)
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - started

    if torch.equal(parameter_before, parameter.detach()):
        raise AssertionError("optimizer step did not update the tracked parameter")
    if scheduler.last_epoch <= scheduler_before:
        raise AssertionError("scheduler did not advance")
    if torch.equal(ema_before, ema_parameter.detach()):
        raise AssertionError("EMA update did not change the tracked parameter")

    return {
        "status": "PASS",
        "reason": None,
        "scenario": scenario,
        "batch_size": int(images.shape[0]),
        "image_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "target_dtype": str(targets.dtype),
        "classification_loss": float(classification_loss.detach().cpu()),
        "contrastive_loss": float(contrastive_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "backward": True,
        "gradients_finite": True,
        "optimizer_step": True,
        "scheduler_step": True,
        "ema_update": True,
        "duration_seconds": duration,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def write_smoke_result(output_dir, result) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "smoke_result.json"
    log_path = output_dir / "smoke_result.log"
    json_temporary = output_dir / "smoke_result.json.tmp"
    log_temporary = output_dir / "smoke_result.log.tmp"
    json_temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = json.loads(json_temporary.read_text(encoding="utf-8"))
    if reloaded != result:
        raise RuntimeError("smoke JSON reload verification failed")
    log_temporary.write_text(
        "status={}\nreason={}\n".format(
            result.get("status"), result.get("reason")
        ),
        encoding="utf-8",
    )
    os.replace(json_temporary, json_path)
    os.replace(log_temporary, log_path)
    return {"json": json_path, "log": log_path}


def _finish_backbone_evidence(evidence):
    if evidence["resolved_path"] is None:
        cached_path = _torch_cache_weight_path()
        if not cached_path.is_file():
            raise FileNotFoundError(
                "model construction did not produce the expected ImageNet weight cache: "
                f"{cached_path}"
            )
        evidence["resolved_path"] = str(cached_path.resolve())
        evidence["resolved_sha256"] = _sha256_file(cached_path)
    return evidence


def validate_loaded_backbone_weights(model, weight_path) -> dict:
    """Prove the loaded backbone exactly matches the supplied ImageNet tensors."""
    import torch

    try:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 1.12 compatibility
        checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("pretrained backbone file must contain a state_dict mapping")
    expected = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(key, str) and not key.startswith("fc.")
    }
    actual = model.backbone.state_dict()
    missing = sorted(set(actual) - set(expected))
    unexpected = sorted(set(expected) - set(actual))
    if missing or unexpected:
        raise ValueError(
            "pretrained backbone key mismatch: missing={}, unexpected={}".format(
                missing, unexpected
            )
        )
    for key, actual_tensor in actual.items():
        expected_tensor = expected[key]
        if not torch.is_tensor(expected_tensor):
            raise TypeError(f"pretrained value is not a tensor: {key}")
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"pretrained backbone shape mismatch for {key}: "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(actual_tensor.shape)}"
            )
        if not torch.equal(actual_tensor.detach().cpu(), expected_tensor.detach().cpu()):
            raise ValueError(f"loaded backbone tensor mismatch: {key}")
    return {
        "exact_state_dict_match": True,
        "verified_tensor_count": len(actual),
        "excluded_checkpoint_prefixes": ["fc."],
    }


def _set_runtime_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_smoke(args) -> dict:
    validation = validate_smoke_args(args, PROJECT_ROOT)
    result = new_smoke_result(args)
    result.update(validation)
    result["architecture"] = "mlic/resnet101"
    result["num_classes"] = 18
    if not cuda_is_available():
        result["reason"] = "CUDA is not available"
        write_smoke_result(args.output_dir, result)
        return result

    initialization = prepare_backbone_initialization(args, args.output_dir)
    result["backbone_initialization"] = initialization

    import torch
    from torch.utils.data._utils.collate import default_collate

    from lib.dataset import build_coco_transform
    from lib.utils import ModelEma, get_lr_scheduler, get_optimizer

    cfg = build_smoke_cfg(args)
    _set_runtime_seed(args.seed)
    transform = build_coco_transform(cfg, training=True)
    dataset = load_train_smoke_dataset(args, transform)
    selections = [
        (scenario, select_scenario_indices(dataset.records, scenario, args.batch_size, args.seed))
        for scenario in args.scenarios
    ]
    runnable = [(name, selection) for name, selection in selections if selection["status"] == "PASS"]
    if not runnable:
        result["scenarios"] = [
            {"scenario": name, **selection} for name, selection in selections
        ]
        result["reason"] = "No requested scenario could form an exact train batch"
        finalize_status(result)
        write_smoke_result(args.output_dir, result)
        return result

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = build_real_smoke_model(cfg)
    _finish_backbone_evidence(initialization)
    initialization["load_verification"] = validate_loaded_backbone_weights(
        model, initialization["resolved_path"]
    )
    model = model.to(device)
    model.train()
    ema_model = ModelEma(model, decay=cfg.ema_decay)
    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = get_lr_scheduler(optimizer, cfg, steps_per_epoch=args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    result["gpu_executed"] = True
    result["gpu_name"] = torch.cuda.get_device_name(device)

    for scenario, selection in selections:
        if len(result["scenarios"]) >= args.max_steps:
            break
        if selection["status"] != "PASS":
            result["scenarios"].append({"scenario": scenario, **selection})
            continue
        batch = default_collate([dataset[index] for index in selection["indices"]])
        runtime = {
            "args": args,
            "cfg": cfg,
            "device": device,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "ema_model": ema_model,
            "scenario": scenario,
        }
        try:
            scenario_result = execute_gpu_scenario(model, batch, runtime)
            scenario_result["indices"] = selection["indices"]
            scenario_result["file_names"] = [
                dataset.records[index].file_name for index in selection["indices"]
            ]
            result["scenarios"].append(scenario_result)
        except torch.cuda.OutOfMemoryError as error:
            result["scenarios"].append(
                {"scenario": scenario, "status": "FAIL", "reason": f"CUDA OOM: {error}"}
            )
            break
        except Exception as error:
            result["scenarios"].append(
                {"scenario": scenario, "status": "FAIL", "reason": repr(error)}
            )
            break

    finalize_status(result)
    passed = [item for item in result["scenarios"] if item["status"] == "PASS"]
    if passed:
        for key in ("classification_loss", "contrastive_loss", "total_loss"):
            result[key] = sum(item[key] for item in passed) / len(passed)
        result["duration_seconds"] = sum(item["duration_seconds"] for item in passed)
        result["peak_allocated_mib"] = max(item["peak_allocated_mib"] for item in passed)
        result["peak_reserved_mib"] = max(item["peak_reserved_mib"] for item in passed)
    result["reason"] = None if result["status"] == "PASS" else result.get("reason")
    write_smoke_result(args.output_dir, result)
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run_smoke(args)
    except Exception as error:
        result = new_smoke_result(args)
        result.update({"status": "FAIL", "reason": repr(error)})
        try:
            write_smoke_result(args.output_dir, result)
        except Exception:
            pass
        return 1
    return {"PASS": 0, "FAIL": 1, "NOT_VERIFIED": 3}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
