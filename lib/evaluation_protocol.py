import csv
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import uuid

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset

from lib.metrics import VOCmAP
from lib.tobacco_dataset import (
    EXCLUDED_SOURCE_ID,
    MODEL_LABEL_COLUMNS,
    SOURCE_LABEL_COLUMNS,
    TobaccoManifestDataset,
)


TOBACCO_NUM_CLASSES = 18
THRESHOLD_MIN = 0.001
THRESHOLD_MAX = 0.999
THRESHOLD_STEP = 0.001
THRESHOLD_OPERATOR = ">="


class ArtifactTransactionError(RuntimeError):
    pass


def evaluation_label_mapping():
    mapping = []
    for source_id, label in enumerate(SOURCE_LABEL_COLUMNS):
        if source_id == EXCLUDED_SOURCE_ID:
            continue
        mapping.append(
            {
                "model_index": len(mapping),
                "source_id": source_id,
                "label": label,
            }
        )
    if tuple(row["label"] for row in mapping) != MODEL_LABEL_COLUMNS:
        raise RuntimeError("tobacco evaluation label mapping is inconsistent")
    return mapping


def _normalize_file_name(value):
    return "/".join(value.replace("\\", "/").split("/"))


class TobaccoEvaluationDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        data_root,
        split,
        transform=None,
        dataset_cls=TobaccoManifestDataset,
    ):
        with open(manifest_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_id", "file_name", "split"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "manifest is missing evaluation identity columns: {}".format(
                        ", ".join(sorted(missing))
                    )
                )
            rows = list(reader)

        identities = {}
        image_ids = []
        for line_number, row in enumerate(rows, start=2):
            image_id = row["image_id"]
            file_name = row["file_name"]
            row_split = row["split"]
            if not image_id or image_id != image_id.strip():
                raise ValueError("line {} image_id must be non-empty".format(line_number))
            if not file_name or file_name != file_name.strip():
                raise ValueError("line {} file_name must be non-empty".format(line_number))
            key = (row_split, _normalize_file_name(file_name))
            if key in identities:
                raise ValueError(
                    "line {} is a duplicate split/file_name identity: {}/{}".format(
                        line_number, row_split, file_name
                    )
                )
            identities[key] = {
                "image_id": image_id,
                "file_name": file_name,
                "split": row_split,
            }
            image_ids.append(image_id)

        counts = Counter(image_ids)
        duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
        self.identity_audit = {
            "row_count": len(rows),
            "global_image_id_unique": duplicate_count == 0,
            "duplicate_image_id_count": duplicate_count,
        }
        self.dataset = dataset_cls(
            manifest_path, data_root, split=split, transform=transform
        )
        self.identities = []
        for record in self.dataset.records:
            key = (split, _normalize_file_name(record.file_name))
            if key not in identities:
                raise ValueError(
                    "Dataset record has no matching manifest identity: {}/{}".format(
                        split, record.file_name
                    )
                )
            self.identities.append(identities[key])
        if len(self.identities) != len(self.dataset):
            raise ValueError("Dataset and manifest identity counts do not match")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample.update(self.identities[index])
        return sample


@torch.no_grad()
def collect_model_outputs(model, dataloader, device):
    model.eval()
    identities = []
    targets = []
    scores = []
    for batch in dataloader:
        images = batch["img"].to(device)
        output = model(images)
        logits = output["logits"]
        batch_scores = torch.sigmoid(logits).detach().cpu().numpy()
        batch_targets = batch["target"].detach().cpu().numpy()
        if batch_scores.shape != batch_targets.shape:
            raise ValueError("model scores and GT must have the same shape")
        for image_id, file_name, split in zip(
            batch["image_id"], batch["file_name"], batch["split"]
        ):
            identities.append(
                {"image_id": image_id, "file_name": file_name, "split": split}
            )
        targets.append(batch_targets)
        scores.append(batch_scores)
    if not targets:
        raise ValueError("evaluation dataloader must not be empty")
    gt = np.concatenate(targets, axis=0)
    score_array = np.concatenate(scores, axis=0)
    validate_evaluation_arrays(gt, score_array)
    if len(identities) != len(gt):
        raise ValueError("identity, GT, and score row counts do not match")
    keys = [(row["split"], _normalize_file_name(row["file_name"])) for row in identities]
    if len(keys) != len(set(keys)):
        raise ValueError("collected predictions contain duplicate split/file_name identities")
    return identities, gt, score_array


def verify_tobacco_evaluation_context(cfg):
    from lib.training_protocol import sha256_file, verify_best_checkpoint_integrity

    if getattr(cfg, "data", None) != "tobacco":
        raise ValueError("tobacco evaluation requires data='tobacco'")
    metadata = verify_best_checkpoint_integrity(
        cfg.ckpt_ema_best_path, cfg.ckpt_best_info_path
    )
    actual_manifest_sha = sha256_file(cfg.manifest_path)
    if actual_manifest_sha != cfg.manifest_sha256:
        raise ValueError("current manifest SHA-256 does not match configuration")

    comparisons = (
        ("architecture", cfg.model, metadata["architecture"]),
        ("backbone", cfg.arch, metadata["backbone"]),
        ("num_classes", cfg.num_classes, metadata["num_classes"]),
        ("image_size", cfg.img_size, metadata["image_size"]),
        ("seed", cfg.seed, metadata["seed"]),
        ("manifest SHA-256", actual_manifest_sha, metadata["manifest_sha256"]),
        ("label mapping", cfg.label_mapping, metadata["label_mapping"]),
        (
            "excluded source class",
            cfg.excluded_source_class_id,
            metadata["excluded_source_class_id"],
        ),
        ("selection_split", cfg.selection_split, metadata["selection_split"]),
        ("selection_metric", cfg.selection_metric, metadata["selection_metric"]),
        (
            "metric_implementation",
            cfg.metric_implementation,
            metadata["metric_implementation"],
        ),
    )
    for name, configured, saved in comparisons:
        if configured != saved:
            raise ValueError("{} does not match checkpoint metadata".format(name))
    if cfg.num_classes != TOBACCO_NUM_CLASSES:
        raise ValueError("num_classes must be 18")
    if cfg.label_mapping != evaluation_label_mapping():
        raise ValueError("label mapping does not match the ordered tobacco contract")
    if cfg.excluded_source_class_id != EXCLUDED_SOURCE_ID:
        raise ValueError("excluded source class must be 13")
    if cfg.selection_split != "val" or cfg.selection_metric != "mAP":
        raise ValueError("selection_split/selection_metric must be val/mAP")
    if cfg.metric_implementation != "VOC2012":
        raise ValueError("metric_implementation must be VOC2012")
    if str(cfg.selection_variant).lower() != "ema" or metadata["model_variant"] != "EMA":
        raise ValueError("evaluation checkpoint must be the selected EMA model")
    if not isinstance(metadata["best_epoch"], int) or metadata["best_epoch"] < 0:
        raise ValueError("best epoch must be a non-negative integer")
    if not math.isfinite(float(metadata["best_val_mAP"])):
        raise ValueError("best val mAP must be finite")

    return {
        "architecture": metadata["architecture"],
        "backbone": metadata["backbone"],
        "num_classes": metadata["num_classes"],
        "image_size": metadata["image_size"],
        "seed": metadata["seed"],
        "manifest_sha256": metadata["manifest_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "label_mapping": metadata["label_mapping"],
        "excluded_source_class_id": metadata["excluded_source_class_id"],
        "selection_split": metadata["selection_split"],
        "selection_metric": metadata["selection_metric"],
        "metric_implementation": metadata["metric_implementation"],
        "model_variant": metadata["model_variant"],
        "best_epoch": metadata["best_epoch"],
        "best_val_mAP": float(metadata["best_val_mAP"]),
        "checkpoint_path": str(Path(cfg.ckpt_ema_best_path)),
    }


def _load_torch_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_verified_ema_state_dict(model, cfg, context=None):
    if context is None:
        context = verify_tobacco_evaluation_context(cfg)
    state_dict = _load_torch_state_dict(cfg.ckpt_ema_best_path)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("EMA checkpoint must contain a non-empty state_dict")
    prefixed = [key.startswith("module.") for key in state_dict]
    if any(prefixed) and not all(prefixed):
        raise ValueError("mixed module. prefix state is not allowed")
    if all(prefixed):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return context


def _validate_calibration_rows(rows, thresholds, mapping):
    if not isinstance(rows, list) or len(rows) != TOBACCO_NUM_CLASSES:
        raise ValueError("threshold calibration metadata must contain 18 rows")
    for index, (row, entry) in enumerate(zip(rows, mapping)):
        for key in ("model_index", "source_id", "label"):
            if row.get(key) != entry[key]:
                raise ValueError(
                    "threshold calibration row {} has inconsistent {}".format(
                        index, key
                    )
                )
        try:
            row_threshold = float(row["threshold"])
            best_f1 = float(row["best_f1"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "threshold calibration row {} has invalid numeric fields".format(index)
            ) from error
        if not math.isclose(
            row_threshold, float(thresholds[index]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "threshold calibration row {} disagrees with thresholds".format(index)
            )
        if not math.isfinite(best_f1) or best_f1 < 0.0 or best_f1 > 1.0:
            raise ValueError(
                "threshold calibration row {} best_f1 must be in [0, 1]".format(index)
            )
        if row.get("degenerate_reason") not in (None, "no_positive_gt"):
            raise ValueError(
                "threshold calibration row {} has an unknown degenerate reason".format(
                    index
                )
            )


def build_threshold_artifact(thresholds, calibration_rows, context):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (TOBACCO_NUM_CLASSES,):
        raise ValueError("thresholds must contain exactly 18 values")
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < THRESHOLD_MIN) or np.any(
        thresholds > THRESHOLD_MAX
    ):
        raise ValueError("thresholds must be finite values in [0.001, 0.999]")
    _validate_calibration_rows(
        calibration_rows, thresholds, context["label_mapping"]
    )
    return {
        "schema_version": 1,
        "calibration_split": "val",
        "operator": THRESHOLD_OPERATOR,
        "thresholds": thresholds.tolist(),
        "label_mapping": deepcopy(context["label_mapping"]),
        "calibration": deepcopy(calibration_rows),
        "search": {
            "minimum": THRESHOLD_MIN,
            "maximum": THRESHOLD_MAX,
            "step": THRESHOLD_STEP,
            "objective": "per_class_f1",
            "tie_break": ["closest_to_0.5", "lower_threshold"],
        },
        **{
            key: context[key]
            for key in (
                "architecture",
                "backbone",
                "num_classes",
                "image_size",
                "seed",
                "manifest_sha256",
                "checkpoint_sha256",
                "excluded_source_class_id",
                "selection_split",
                "selection_metric",
                "metric_implementation",
                "model_variant",
                "best_epoch",
                "best_val_mAP",
            )
        },
    }


def load_verified_thresholds(path, context):
    with open(path, "r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    return _verify_threshold_artifact(artifact, context)


def _verify_threshold_artifact(artifact, context):
    if artifact.get("calibration_split") != "val":
        raise ValueError("thresholds must be calibrated on val")
    if artifact.get("operator") != THRESHOLD_OPERATOR:
        raise ValueError("threshold operator must be >=")
    if artifact.get("label_mapping") != context["label_mapping"]:
        raise ValueError("threshold label mapping does not match evaluation context")
    search = artifact.get("search")
    expected_search = {
        "minimum": THRESHOLD_MIN,
        "maximum": THRESHOLD_MAX,
        "step": THRESHOLD_STEP,
        "objective": "per_class_f1",
        "tie_break": ["closest_to_0.5", "lower_threshold"],
    }
    if search != expected_search:
        raise ValueError("threshold search protocol does not match")
    field_labels = {
        "architecture": "architecture",
        "backbone": "backbone",
        "num_classes": "num_classes",
        "image_size": "image_size",
        "seed": "seed",
        "manifest_sha256": "manifest",
        "checkpoint_sha256": "checkpoint",
        "excluded_source_class_id": "excluded source class",
        "selection_split": "selection split",
        "selection_metric": "selection metric",
        "metric_implementation": "metric implementation",
        "model_variant": "model variant",
        "best_epoch": "epoch",
        "best_val_mAP": "best val mAP",
    }
    for key, label in field_labels.items():
        if artifact.get(key) != context.get(key):
            raise ValueError("threshold {} does not match evaluation context".format(label))
    thresholds = np.asarray(artifact.get("thresholds"), dtype=np.float64)
    if thresholds.shape != (TOBACCO_NUM_CLASSES,):
        raise ValueError("threshold artifact must contain 18 thresholds")
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < THRESHOLD_MIN) or np.any(
        thresholds > THRESHOLD_MAX
    ):
        raise ValueError("threshold values must be finite and within search range")
    _validate_calibration_rows(
        artifact.get("calibration"), thresholds, context["label_mapping"]
    )
    return thresholds


def _prediction_columns():
    mapping = evaluation_label_mapping()
    columns = ["image_id", "file_name", "split"]
    for prefix in ("gt", "score", "pred"):
        columns.extend(
            "{}_{}_{}".format(prefix, row["model_index"], row["label"])
            for row in mapping
        )
    return columns


def _per_class_columns():
    return [
        "model_index",
        "source_id",
        "label",
        "voc2012_ap",
        "sklearn_ap",
        "precision",
        "recall",
        "f1",
        "support",
        "predicted_positives",
        "threshold",
    ]


def _json_dump(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def _csv_write(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_bundle(bundle, expected_split):
    gt, scores = validate_evaluation_arrays(bundle["gt"], bundle["scores"])
    thresholds = np.asarray(bundle["thresholds"], dtype=np.float64)
    summary, per_class, predictions = compute_evaluation_metrics(
        gt, scores, thresholds
    )
    identities = list(bundle["identities"])
    if len(identities) != len(gt):
        raise ValueError("identity, GT, and score row counts do not match")
    keys = []
    for identity in identities:
        if identity.get("split") != expected_split:
            raise ValueError("artifact identity split does not match target split")
        key = (identity["split"], _normalize_file_name(identity["file_name"]))
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError("artifact identities contain duplicate split/file_name")
    np.testing.assert_array_equal(predictions, bundle["predictions"])
    for key, value in summary.items():
        if not math.isclose(value, float(bundle["summary"][key]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("bundle metric {} is not recomputable".format(key))
    return identities, gt, scores, thresholds, summary, per_class, predictions


def _metrics_document(split, bundle, summary, thresholds):
    return {
        "schema_version": 1,
        "split": split,
        "sample_count": len(bundle["identities"]),
        "num_classes": TOBACCO_NUM_CLASSES,
        "summary": summary,
        "thresholds": thresholds.tolist(),
        "threshold_operator": THRESHOLD_OPERATOR,
        "ap_no_positive_policy": "zero_if_no_positive_gt",
        "zero_division_policy": 0.0,
        "definitions": {
            "voc2012_map": "mean of repository VOCmAP(year='2012') per-class AP",
            "sklearn_map": "arithmetic mean of 18 sklearn average_precision_score AP values",
            "macro_f1": "arithmetic mean of 18 per-class F1 values",
            "micro_f1": "F1 from globally aggregated TP, FP, and FN",
            "exact_match_ratio": "fraction of exactly matching 18-label vectors",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
        },
        "threshold_provenance": deepcopy(bundle["threshold_artifact"]),
        "provenance": deepcopy(bundle["context"]),
    }


def _write_artifact_files(stage, bundle, split, include_thresholds):
    identities, gt, scores, thresholds, summary, per_class, predictions = (
        _validate_bundle(bundle, split)
    )
    order = sorted(
        range(len(identities)),
        key=lambda index: _evaluation_identity_key(identities[index]),
    )
    mapping = evaluation_label_mapping()
    prediction_rows = []
    for index in order:
        row = dict(identities[index])
        for class_index, entry in enumerate(mapping):
            suffix = "{}_{}".format(class_index, entry["label"])
            row["gt_" + suffix] = int(gt[index, class_index])
            row["score_" + suffix] = format(float(scores[index, class_index]), ".17g")
            row["pred_" + suffix] = int(predictions[index, class_index])
        prediction_rows.append(row)
    prefix = "val" if split == "val" else "test"
    _csv_write(stage / (prefix + "_predictions.csv"), _prediction_columns(), prediction_rows)
    _csv_write(stage / (prefix + "_per_class.csv"), _per_class_columns(), per_class)
    _json_dump(
        _metrics_document(split, bundle, summary, thresholds),
        stage / (prefix + "_metrics.json"),
    )
    if include_thresholds:
        _json_dump(bundle["threshold_artifact"], stage / "thresholds.json")


def _compare_context(saved, expected):
    for key in (
        "architecture",
        "backbone",
        "num_classes",
        "image_size",
        "seed",
        "manifest_sha256",
        "checkpoint_sha256",
        "label_mapping",
        "excluded_source_class_id",
        "selection_split",
        "selection_metric",
        "metric_implementation",
        "model_variant",
        "best_epoch",
        "best_val_mAP",
    ):
        if saved.get(key) != expected.get(key):
            raise ValueError("artifact provenance mismatch for {}".format(key))


def _validate_artifacts(path, expected_context, split, include_thresholds):
    path = Path(path)
    prefix = "val" if split == "val" else "test"
    expected_names = {
        prefix + "_predictions.csv",
        prefix + "_metrics.json",
        prefix + "_per_class.csv",
    }
    if include_thresholds:
        expected_names.add("thresholds.json")
    actual_names = {item.name for item in path.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ValueError("artifact directory contains an incomplete or unexpected file set")
    with open(path / (prefix + "_metrics.json"), "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if metrics.get("split") != split:
        raise ValueError("metrics split does not match artifact directory")
    if metrics.get("num_classes") != TOBACCO_NUM_CLASSES:
        raise ValueError("metrics must contain 18 classes")
    _compare_context(metrics.get("provenance", {}), expected_context)
    threshold_provenance = metrics.get("threshold_provenance", {})
    if include_thresholds:
        threshold_path = path / "thresholds.json"
        with open(threshold_path, "r", encoding="utf-8") as handle:
            if json.load(handle) != threshold_provenance:
                raise ValueError("threshold file and metrics provenance differ")
    thresholds = _verify_threshold_artifact(threshold_provenance, expected_context)

    with open(path / (prefix + "_predictions.csv"), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _prediction_columns():
            raise ValueError("prediction CSV columns do not match the evaluation schema")
        rows = list(reader)
    if not rows or len(rows) != metrics.get("sample_count"):
        raise ValueError("prediction row count does not match metrics")
    keys = [(row["split"], _normalize_file_name(row["file_name"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("prediction CSV contains duplicate split/file_name identities")
    if any(row["split"] != split for row in rows):
        raise ValueError("prediction CSV contains another split")
    mapping = evaluation_label_mapping()
    gt = np.asarray(
        [[int(row["gt_{}_{}".format(i, entry["label"])]) for i, entry in enumerate(mapping)] for row in rows]
    )
    scores = np.asarray(
        [[float(row["score_{}_{}".format(i, entry["label"])]) for i, entry in enumerate(mapping)] for row in rows]
    )
    saved_predictions = np.asarray(
        [[int(row["pred_{}_{}".format(i, entry["label"])]) for i, entry in enumerate(mapping)] for row in rows]
    )
    summary, per_class, predictions = compute_evaluation_metrics(gt, scores, thresholds)
    if not np.array_equal(saved_predictions, predictions):
        raise ValueError("saved predictions do not match recomputed prediction using >=")
    for key, value in summary.items():
        if not math.isclose(value, float(metrics["summary"][key]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("saved metric {} cannot be recomputed".format(key))
    with open(path / (prefix + "_per_class.csv"), "r", encoding="utf-8", newline="") as handle:
        per_class_rows = list(csv.DictReader(handle))
    if len(per_class_rows) != TOBACCO_NUM_CLASSES:
        raise ValueError("per-class artifact must contain 18 rows")
    for saved, recomputed in zip(per_class_rows, per_class):
        if int(saved["model_index"]) != recomputed["model_index"]:
            raise ValueError("per-class rows are out of order")
        for key in ("voc2012_ap", "sklearn_ap", "precision", "recall", "f1", "threshold"):
            if not math.isclose(float(saved[key]), recomputed[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("per-class {} cannot be recomputed".format(key))
    return {**metrics, "summary": summary, "per_class": per_class}


def validate_calibration_artifacts(path, expected_context):
    return _validate_artifacts(path, expected_context, "val", True)


def validate_evaluation_artifacts(path, expected_context):
    return _validate_artifacts(path, expected_context, "test", False)


def _publish_artifacts(
    target_dir,
    bundle,
    files_writer,
    validate_fn,
    replace_fn=os.replace,
):
    target = Path(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=".{}.staging.".format(target.name), dir=str(target.parent))
    )
    backup = target.parent / ".{}.backup.{}".format(target.name, uuid.uuid4().hex)
    target_existed = target.exists()
    moved_old = False
    preserve_evidence = False
    try:
        files_writer(stage, bundle)
        validate_fn(stage, bundle["context"])
        if target_existed:
            replace_fn(str(target), str(backup))
            moved_old = True
        try:
            replace_fn(str(stage), str(target))
            validate_fn(target, bundle["context"])
        except Exception as publish_error:
            rollback_errors = []
            if target.exists():
                try:
                    shutil.rmtree(target)
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if moved_old:
                try:
                    replace_fn(str(backup), str(target))
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                preserve_evidence = True
                raise ArtifactTransactionError(
                    "artifact publication failed and rollback could not restore the prior directory: {}".format(
                        rollback_errors
                    )
                ) from publish_error
            raise
        if moved_old and backup.exists():
            shutil.rmtree(backup)
    finally:
        if not preserve_evidence:
            if stage.exists():
                shutil.rmtree(stage)
            if backup.exists():
                shutil.rmtree(backup)


def _write_calibration_files(stage, bundle):
    _write_artifact_files(Path(stage), bundle, "val", True)


def _write_evaluation_files(stage, bundle):
    _write_artifact_files(Path(stage), bundle, "test", False)


def write_calibration_artifacts(
    target_dir,
    bundle,
    replace_fn=os.replace,
    files_writer=_write_calibration_files,
    validate_fn=validate_calibration_artifacts,
):
    _publish_artifacts(
        target_dir, bundle, files_writer, validate_fn, replace_fn=replace_fn
    )


def write_evaluation_artifacts(
    target_dir,
    bundle,
    replace_fn=os.replace,
    files_writer=_write_evaluation_files,
    validate_fn=validate_evaluation_artifacts,
):
    _publish_artifacts(
        target_dir, bundle, files_writer, validate_fn, replace_fn=replace_fn
    )


def _default_model_factory(cfg):
    from models.factory import create_model

    return create_model(cfg.model, cfg=cfg)


def _default_transform_factory(cfg, training):
    from lib.dataset import build_coco_transform

    return build_coco_transform(cfg, training=training)


def _evaluation_loader(dataset, cfg):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, "num_workers", 4),
    )


def _evaluation_identity_key(identity):
    return identity["split"], _normalize_file_name(identity["file_name"])


def _canonicalize_evaluation_rows(identities, gt, scores):
    identities = list(identities)
    gt, scores = validate_evaluation_arrays(gt, scores)
    if len(identities) != len(gt):
        raise ValueError("identity, GT, and score row counts do not match")
    order = sorted(
        range(len(identities)),
        key=lambda index: _evaluation_identity_key(identities[index]),
    )
    return (
        [identities[index] for index in order],
        gt[order],
        scores[order],
    )


def _build_bundle(identities, gt, scores, thresholds, context, threshold_artifact):
    identities, gt, scores = _canonicalize_evaluation_rows(
        identities, gt, scores
    )
    summary, per_class, predictions = compute_evaluation_metrics(
        gt, scores, thresholds
    )
    return {
        "identities": identities,
        "gt": gt,
        "scores": scores,
        "predictions": predictions,
        "thresholds": thresholds,
        "summary": summary,
        "per_class": per_class,
        "context": context,
        "threshold_artifact": threshold_artifact,
    }


def run_val_calibration(
    cfg,
    output_root,
    *,
    model_factory=_default_model_factory,
    dataset_cls=TobaccoEvaluationDataset,
    transform_factory=_default_transform_factory,
    device=None,
    context_verifier=verify_tobacco_evaluation_context,
    state_loader=load_verified_ema_state_dict,
    collector=collect_model_outputs,
    calibrator=None,
    artifact_writer=write_calibration_artifacts,
):
    context = context_verifier(cfg)
    if device is None:
        device = torch.device(getattr(cfg, "evaluation_device", "cuda"))
    model = model_factory(cfg).to(device)
    state_loader(model, cfg, context=context)
    dataset = dataset_cls(
        cfg.manifest_path,
        cfg.data_root,
        split="val",
        transform=transform_factory(cfg, training=False),
    )
    dataloader = _evaluation_loader(dataset, cfg)
    identities, gt, scores = collector(model, dataloader, device)
    if calibrator is None:
        calibrator = calibrate_thresholds
    thresholds, calibration_rows = calibrator(gt, scores, split="val")
    threshold_artifact = build_threshold_artifact(
        thresholds, calibration_rows, context
    )
    bundle = _build_bundle(
        identities, gt, scores, thresholds, context, threshold_artifact
    )
    target = Path(output_root) / "calibration"
    artifact_writer(target, bundle)
    return {"split": "val", "output_dir": str(target), "context": context}


def run_test_evaluation(
    cfg,
    output_root,
    thresholds_path,
    *,
    model_factory=_default_model_factory,
    dataset_cls=TobaccoEvaluationDataset,
    transform_factory=_default_transform_factory,
    device=None,
    context_verifier=verify_tobacco_evaluation_context,
    state_loader=load_verified_ema_state_dict,
    collector=collect_model_outputs,
    artifact_writer=write_evaluation_artifacts,
):
    context = context_verifier(cfg)
    thresholds = load_verified_thresholds(thresholds_path, context)
    with open(thresholds_path, "r", encoding="utf-8") as handle:
        threshold_artifact = json.load(handle)
    if device is None:
        device = torch.device(getattr(cfg, "evaluation_device", "cuda"))
    model = model_factory(cfg).to(device)
    state_loader(model, cfg, context=context)
    dataset = dataset_cls(
        cfg.manifest_path,
        cfg.data_root,
        split="test",
        transform=transform_factory(cfg, training=False),
    )
    dataloader = _evaluation_loader(dataset, cfg)
    identities, gt, scores = collector(model, dataloader, device)
    bundle = _build_bundle(
        identities, gt, scores, thresholds, context, threshold_artifact
    )
    target = Path(output_root) / "evaluation"
    artifact_writer(target, bundle)
    return {"split": "test", "output_dir": str(target), "context": context}


def validate_evaluation_arrays(gt, scores):
    gt = np.asarray(gt)
    scores = np.asarray(scores, dtype=np.float64)
    if gt.ndim != 2 or scores.ndim != 2:
        raise ValueError("GT and scores must be two-dimensional")
    if gt.shape != scores.shape:
        raise ValueError("GT and scores must have the same shape")
    if gt.shape[0] == 0:
        raise ValueError("evaluation split must not be empty")
    if gt.shape[1] != TOBACCO_NUM_CLASSES:
        raise ValueError("evaluation arrays must contain 18 classes")
    if not np.all(np.isin(gt, (0, 1))):
        raise ValueError("GT must contain binary 0/1 labels")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("scores must be in [0, 1]")
    return gt.astype(np.int64, copy=False), scores


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision, recall):
    return _safe_ratio(2.0 * precision * recall, precision + recall)


def compute_evaluation_metrics(gt, scores, thresholds):
    gt, scores = validate_evaluation_arrays(gt, scores)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (TOBACCO_NUM_CLASSES,):
        raise ValueError("thresholds must have shape [18]")
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0) or np.any(
        thresholds > 1
    ):
        raise ValueError("thresholds must be finite values in [0, 1]")

    predictions = (scores >= thresholds.reshape(1, -1)).astype(np.int64)
    voc_ap, voc_map = VOCmAP(
        TOBACCO_NUM_CLASSES, year="2012", ignore_path=None
    ).compute(scores=scores, targets=gt)

    mapping = evaluation_label_mapping()
    per_class = []
    total_tp = total_fp = total_fn = 0
    for index, entry in enumerate(mapping):
        target = gt[:, index]
        predicted = predictions[:, index]
        tp = int(np.sum((target == 1) & (predicted == 1)))
        fp = int(np.sum((target == 0) & (predicted == 1)))
        fn = int(np.sum((target == 1) & (predicted == 0)))
        support = int(np.sum(target))
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _f1(precision, recall)
        sklearn_ap = (
            float(average_precision_score(target, scores[:, index]))
            if support
            else 0.0
        )
        per_class.append(
            {
                **entry,
                "voc2012_ap": float(voc_ap[index]),
                "sklearn_ap": sklearn_ap,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted_positives": int(np.sum(predicted)),
                "threshold": float(thresholds[index]),
            }
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = _safe_ratio(total_tp, total_tp + total_fp)
    micro_recall = _safe_ratio(total_tp, total_tp + total_fn)
    summary = {
        "voc2012_map": float(voc_map),
        "sklearn_map": float(np.mean([row["sklearn_ap"] for row in per_class])),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _f1(micro_precision, micro_recall),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "exact_match_ratio": float(np.mean(np.all(predictions == gt, axis=1))),
    }
    return summary, per_class, predictions


def select_best_threshold(candidates, f1_values):
    candidates = np.asarray(candidates, dtype=np.float64)
    f1_values = np.asarray(f1_values, dtype=np.float64)
    if candidates.ndim != 1 or candidates.shape != f1_values.shape or not len(candidates):
        raise ValueError("threshold candidates and F1 values must be equal non-empty vectors")
    best = np.max(f1_values)
    tied = candidates[np.isclose(f1_values, best, rtol=0.0, atol=1e-12)]
    return float(min(tied, key=lambda value: (abs(float(value) - 0.5), float(value))))


def calibrate_thresholds(gt, scores, split):
    if split != "val":
        raise ValueError("threshold calibration is permitted only for split='val'")
    gt, scores = validate_evaluation_arrays(gt, scores)
    candidates = np.arange(1, 1000, dtype=np.float64) / 1000.0
    thresholds = np.empty(TOBACCO_NUM_CLASSES, dtype=np.float64)
    rows = []
    for index, entry in enumerate(evaluation_label_mapping()):
        target = gt[:, index]
        if not np.any(target == 1):
            threshold = 0.5
            best_f1 = 0.0
            reason = "no_positive_gt"
        else:
            predicted = scores[:, index, None] >= candidates[None, :]
            tp = np.sum(predicted & (target[:, None] == 1), axis=0)
            fp = np.sum(predicted & (target[:, None] == 0), axis=0)
            fn = np.sum((~predicted) & (target[:, None] == 1), axis=0)
            denominator = 2 * tp + fp + fn
            f1_values = np.divide(
                2 * tp,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator != 0,
            )
            threshold = select_best_threshold(candidates, f1_values)
            best_f1 = float(f1_values[np.where(candidates == threshold)[0][0]])
            reason = None
        thresholds[index] = threshold
        rows.append(
            {
                **entry,
                "threshold": float(threshold),
                "best_f1": best_f1,
                "degenerate_reason": reason,
            }
        )
    return thresholds, rows
