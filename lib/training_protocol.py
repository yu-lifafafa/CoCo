import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch

from lib.dataset import MLDataset, build_coco_transform
from lib.tobacco_dataset import (
    EXCLUDED_SOURCE_ID,
    MODEL_LABEL_COLUMNS,
    SOURCE_LABEL_COLUMNS,
    TobaccoManifestDataset,
)


TOBACCO_NUM_CLASSES = 18


class CheckpointTransactionError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_finite_map(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite, got {!r}".format(name, value))
    return value


def is_strict_improvement(candidate, best):
    candidate = validate_finite_map("candidate mAP", candidate)
    if best is None:
        return True
    best = validate_finite_map("best mAP", best)
    return candidate > best


def training_log_interval(loader_length):
    return max(1, int(loader_length) // 6)


def should_clear_failed_experiment(cfg):
    return cfg.data != "tobacco"


def _temporary_path(directory, prefix, suffix):
    descriptor, path = tempfile.mkstemp(dir=str(directory), prefix=prefix, suffix=suffix)
    os.close(descriptor)
    return Path(path)


def _write_checkpoint(state_dict, path):
    with open(path, "wb") as handle:
        torch.save(state_dict, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def _load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_saved_state_dict(saved, expected):
    if not isinstance(saved, dict):
        raise ValueError("EMA checkpoint must contain a state_dict mapping")
    if list(saved) != list(expected):
        raise ValueError("EMA checkpoint state_dict keys do not match")
    for key in expected:
        if not hasattr(saved[key], "shape") or saved[key].shape != expected[key].shape:
            raise ValueError("EMA checkpoint tensor shape mismatch for {}".format(key))


def _checkpoint_metadata(cfg, epoch, val_map, checkpoint_sha):
    return {
        "architecture": cfg.model,
        "backbone": cfg.arch,
        "num_classes": cfg.num_classes,
        "image_size": cfg.img_size,
        "seed": cfg.seed,
        "manifest_sha256": cfg.manifest_sha256,
        "label_mapping": cfg.label_mapping,
        "excluded_source_class_id": cfg.excluded_source_class_id,
        "selection_split": cfg.selection_split,
        "selection_metric": cfg.selection_metric,
        "metric_implementation": cfg.metric_implementation,
        "model_variant": "EMA",
        "best_epoch": int(epoch),
        "best_val_mAP": float(val_map),
        "checkpoint_path": cfg.ckpt_ema_best_path,
        "checkpoint_sha256": checkpoint_sha,
    }


def verify_best_checkpoint_integrity(checkpoint_path, metadata_path):
    checkpoint_path = Path(checkpoint_path)
    metadata_path = Path(metadata_path)
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("checkpoint and metadata must both exist")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {
        "architecture",
        "backbone",
        "num_classes",
        "image_size",
        "seed",
        "manifest_sha256",
        "label_mapping",
        "excluded_source_class_id",
        "selection_split",
        "selection_metric",
        "metric_implementation",
        "model_variant",
        "best_epoch",
        "best_val_mAP",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError("checkpoint metadata missing fields: {}".format(", ".join(missing)))
    expected_path = Path(metadata["checkpoint_path"])
    if expected_path.resolve() != checkpoint_path.resolve():
        raise ValueError("checkpoint metadata path does not match checkpoint")
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match metadata")
    validate_finite_map("best val mAP", metadata["best_val_mAP"])
    _load_state_dict(checkpoint_path)
    return metadata


def save_best_ema_checkpoint(
    ema_model,
    cfg,
    epoch,
    val_map,
    checkpoint_writer=_write_checkpoint,
    metadata_writer=_write_metadata,
    replace_fn=os.replace,
):
    val_map = validate_finite_map("EMA val mAP", val_map)
    checkpoint_path = Path(cfg.ckpt_ema_best_path)
    metadata_path = Path(cfg.ckpt_best_info_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_exists = checkpoint_path.is_file()
    metadata_exists = metadata_path.is_file()
    if checkpoint_exists != metadata_exists:
        raise ValueError("existing checkpoint and metadata are inconsistent")
    if checkpoint_exists:
        verify_best_checkpoint_integrity(checkpoint_path, metadata_path)

    candidate_checkpoint = _temporary_path(
        checkpoint_path.parent, ".best_ema_candidate_", ".pth.tmp"
    )
    candidate_metadata = _temporary_path(
        metadata_path.parent, ".best_info_candidate_", ".json.tmp"
    )
    backup_checkpoint = None
    backup_metadata = None
    preserve_artifacts = False
    replacement_started = False
    expected_state = ema_model.state_dict()
    try:
        checkpoint_writer(expected_state, candidate_checkpoint)
        saved_state = _load_state_dict(candidate_checkpoint)
        _validate_saved_state_dict(saved_state, expected_state)
        checkpoint_sha = sha256_file(candidate_checkpoint)
        metadata = _checkpoint_metadata(cfg, epoch, val_map, checkpoint_sha)
        metadata_writer(metadata, candidate_metadata)
        with candidate_metadata.open("r", encoding="utf-8") as handle:
            if json.load(handle) != metadata:
                raise ValueError("candidate checkpoint metadata failed verification")

        if checkpoint_exists:
            backup_checkpoint = _temporary_path(
                checkpoint_path.parent, ".best_ema_backup_", ".pth.bak"
            )
            backup_metadata = _temporary_path(
                metadata_path.parent, ".best_info_backup_", ".json.bak"
            )
            shutil.copy2(checkpoint_path, backup_checkpoint)
            shutil.copy2(metadata_path, backup_metadata)

        replacement_started = True
        replace_fn(str(candidate_checkpoint), str(checkpoint_path))
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise ValueError("installed checkpoint SHA-256 verification failed")
        replace_fn(str(candidate_metadata), str(metadata_path))
        installed_metadata = verify_best_checkpoint_integrity(
            checkpoint_path, metadata_path
        )
        if (
            installed_metadata["best_epoch"] != int(epoch)
            or installed_metadata["best_val_mAP"] != val_map
        ):
            raise ValueError("installed checkpoint metadata identifies another result")
        return installed_metadata
    except Exception as error:
        rollback_errors = []
        if replacement_started:
            if checkpoint_exists:
                for backup, final in (
                    (backup_checkpoint, checkpoint_path),
                    (backup_metadata, metadata_path),
                ):
                    try:
                        replace_fn(str(backup), str(final))
                    except Exception as rollback_error:
                        rollback_errors.append(rollback_error)
                if not rollback_errors:
                    try:
                        verify_best_checkpoint_integrity(checkpoint_path, metadata_path)
                    except Exception as rollback_error:
                        rollback_errors.append(rollback_error)
            else:
                for final in (checkpoint_path, metadata_path):
                    try:
                        if final.exists():
                            final.unlink()
                    except Exception as rollback_error:
                        rollback_errors.append(rollback_error)
        if rollback_errors:
            preserve_artifacts = True
            raise CheckpointTransactionError(
                "checkpoint update failed and rollback could not restore a verified pair: {}".format(
                    rollback_errors
                )
            ) from error
        raise
    finally:
        if not preserve_artifacts:
            for path in (
                candidate_checkpoint,
                candidate_metadata,
                backup_checkpoint,
                backup_metadata,
            ):
                if path is not None and path.exists():
                    path.unlink()


def tobacco_label_mapping():
    mapping = []
    model_index = 0
    for source_id, label in enumerate(SOURCE_LABEL_COLUMNS):
        if source_id == EXCLUDED_SOURCE_ID:
            continue
        mapping.append(
            {
                "model_index": model_index,
                "source_id": source_id,
                "label": label,
            }
        )
        model_index += 1
    if tuple(entry["label"] for entry in mapping) != MODEL_LABEL_COLUMNS:
        raise RuntimeError("tobacco label mapping is inconsistent")
    return mapping


def build_training_datasets(
    cfg,
    tobacco_dataset_cls=TobaccoManifestDataset,
    legacy_dataset_cls=MLDataset,
):
    if cfg.data == "tobacco":
        if cfg.num_classes != TOBACCO_NUM_CLASSES:
            raise ValueError(
                "tobacco num_classes must be {}, got {}".format(
                    TOBACCO_NUM_CLASSES, cfg.num_classes
                )
            )
        train_dataset = tobacco_dataset_cls(
            cfg.manifest_path,
            cfg.data_root,
            split="train",
            transform=build_coco_transform(cfg, training=True),
        )
        val_dataset = tobacco_dataset_cls(
            cfg.manifest_path,
            cfg.data_root,
            split="val",
            transform=build_coco_transform(cfg, training=False),
        )
        return train_dataset, val_dataset

    return (
        legacy_dataset_cls(cfg, cfg.train_path, training=True),
        legacy_dataset_cls(cfg, cfg.test_path, training=False),
    )
