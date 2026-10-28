import csv
import hashlib
import json
from pathlib import Path
from argparse import Namespace
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Sampler

if "randaugment" not in sys.modules:
    randaugment_stub = types.ModuleType("randaugment")

    class _RandAugment:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, image):
            return image

    randaugment_stub.RandAugment = _RandAugment
    sys.modules["randaugment"] = randaugment_stub

from lib.evaluation_protocol import (
    ArtifactTransactionError,
    TobaccoEvaluationDataset,
    build_threshold_artifact,
    calibrate_thresholds,
    collect_model_outputs,
    compute_evaluation_metrics,
    evaluation_label_mapping,
    load_verified_ema_state_dict,
    load_verified_thresholds,
    run_test_evaluation,
    run_val_calibration,
    select_best_threshold,
    validate_evaluation_arrays,
    validate_calibration_artifacts,
    validate_evaluation_artifacts,
    verify_tobacco_evaluation_context,
    write_calibration_artifacts,
    write_evaluation_artifacts,
)
from lib.metrics import VOCmAP
from lib.tobacco_dataset import SOURCE_LABEL_COLUMNS


NUM_CLASSES = 18


def _manifest_row(file_name, split="val", image_id="image-1", positives=(0,)):
    row = {
        "image_id": image_id,
        "file_name": file_name,
        "split": split,
        "is_healthy": "0" if positives else "1",
        "positive_label_count": str(len(positives)),
    }
    row.update({column: "0" for column in SOURCE_LABEL_COLUMNS})
    for source_id in positives:
        row[SOURCE_LABEL_COLUMNS[source_id]] = "1"
    return row


def _write_manifest(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_image(data_root, split, file_name, red):
    path = data_root / split / Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(red, 0, 0)).save(path)
    return path


def _to_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_fixture(tmp_path, state_dict=None):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("identity fixture", encoding="utf-8")
    checkpoint = tmp_path / "best_ema_model.pth"
    if state_dict is None:
        state_dict = torch.nn.Linear(2, 2).state_dict()
    torch.save(state_dict, checkpoint)
    metadata_path = tmp_path / "best_checkpoint_info.json"
    metadata = {
        "architecture": "mlic",
        "backbone": "resnet101",
        "num_classes": 18,
        "image_size": 384,
        "seed": 7,
        "manifest_sha256": _sha256(manifest),
        "label_mapping": evaluation_label_mapping(),
        "excluded_source_class_id": 13,
        "selection_split": "val",
        "selection_metric": "mAP",
        "metric_implementation": "VOC2012",
        "model_variant": "EMA",
        "best_epoch": 10,
        "best_val_mAP": 0.25,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    cfg = Namespace(
        data="tobacco",
        model="mlic",
        arch="resnet101",
        num_classes=18,
        img_size=384,
        seed=7,
        manifest_path=str(manifest),
        manifest_sha256=_sha256(manifest),
        label_mapping=evaluation_label_mapping(),
        excluded_source_class_id=13,
        selection_split="val",
        selection_metric="mAP",
        selection_variant="ema",
        metric_implementation="VOC2012",
        ckpt_ema_best_path=str(checkpoint),
        ckpt_best_info_path=str(metadata_path),
    )
    return cfg, metadata


def _fixture_arrays():
    gt = np.zeros((5, NUM_CLASSES), dtype=np.int64)
    gt[0, 0] = 1
    gt[1, (0, 1)] = 1
    gt[3, 1] = 1
    scores = np.full((5, NUM_CLASSES), 0.05, dtype=np.float64)
    scores[:, 0] = [0.9, 0.8, 0.7, 0.1, 0.2]
    scores[:, 1] = [0.4, 0.6, 0.3, 0.9, 0.2]
    return gt, scores


@pytest.mark.parametrize(
    "gt,scores,message",
    [
        (np.zeros((0, NUM_CLASSES)), np.zeros((0, NUM_CLASSES)), "empty"),
        (np.zeros((2, 17)), np.zeros((2, 17)), "18"),
        (np.zeros((2, NUM_CLASSES)), np.zeros((3, NUM_CLASSES)), "same shape"),
        (np.full((2, NUM_CLASSES), 2), np.zeros((2, NUM_CLASSES)), "binary"),
        (np.zeros((2, NUM_CLASSES)), np.full((2, NUM_CLASSES), np.nan), "finite"),
        (np.zeros((2, NUM_CLASSES)), np.full((2, NUM_CLASSES), 1.1), "[0, 1]"),
    ],
)
def test_evaluation_array_validation_rejects_invalid_contract(gt, scores, message):
    with pytest.raises(ValueError, match=message):
        validate_evaluation_arrays(gt, scores)


def test_official_voc2012_matches_repository_meter_including_tied_scores():
    gt, scores = _fixture_arrays()
    scores[0:3, 0] = 0.8
    thresholds = np.full(NUM_CLASSES, 0.5)

    expected_ap, expected_map = VOCmAP(NUM_CLASSES, year="2012").compute(
        scores=scores, targets=gt
    )
    summary, per_class, _ = compute_evaluation_metrics(gt, scores, thresholds)

    np.testing.assert_allclose(
        [row["voc2012_ap"] for row in per_class], expected_ap, atol=0, rtol=0
    )
    assert summary["voc2012_map"] == pytest.approx(expected_map, abs=0, rel=0)


def test_sklearn_ap_is_separately_named_and_no_positive_classes_are_zero():
    gt, scores = _fixture_arrays()
    summary, per_class, _ = compute_evaluation_metrics(
        gt, scores, np.full(NUM_CLASSES, 0.5)
    )

    assert per_class[0]["sklearn_ap"] == pytest.approx(
        average_precision_score(gt[:, 0], scores[:, 0])
    )
    assert per_class[2]["sklearn_ap"] == 0.0
    assert per_class[2]["voc2012_ap"] == 0.0
    assert summary["sklearn_map"] == pytest.approx(
        np.mean([row["sklearn_ap"] for row in per_class])
    )
    assert "map" not in summary


def test_sklearn_ap_matches_sklearn_grouped_tie_semantics():
    gt, scores = _fixture_arrays()
    scores[:3, 0] = 0.8
    _, per_class, _ = compute_evaluation_metrics(
        gt, scores, np.full(NUM_CLASSES, 0.5)
    )
    assert per_class[0]["sklearn_ap"] == pytest.approx(
        average_precision_score(gt[:, 0], scores[:, 0]), abs=0, rel=0
    )


def test_thresholded_metrics_use_per_class_macro_micro_and_exact_match():
    gt = np.zeros((3, NUM_CLASSES), dtype=np.int64)
    gt[0, 0] = 1
    gt[1, (0, 1)] = 1
    scores = np.zeros((3, NUM_CLASSES), dtype=np.float64)
    scores[0, 0] = 0.5  # equality proves >=
    scores[1, 0] = 0.4
    scores[1, 1] = 0.9
    summary, per_class, predictions = compute_evaluation_metrics(
        gt, scores, np.full(NUM_CLASSES, 0.5)
    )

    assert predictions[0, 0] == 1
    assert per_class[0]["f1"] == pytest.approx(2.0 / 3.0)
    assert per_class[1]["f1"] == 1.0
    assert summary["macro_f1"] == pytest.approx((2.0 / 3.0 + 1.0) / NUM_CLASSES)
    assert summary["micro_precision"] == 1.0
    assert summary["micro_recall"] == pytest.approx(2.0 / 3.0)
    assert summary["micro_f1"] == pytest.approx(0.8)
    assert summary["exact_match_ratio"] == pytest.approx(2.0 / 3.0)


def test_threshold_search_includes_both_boundaries():
    gt = np.zeros((2, NUM_CLASSES), dtype=np.int64)
    gt[0, :] = 1
    low_scores = np.vstack((np.full(NUM_CLASSES, 0.001), np.zeros(NUM_CLASSES)))
    high_scores = np.vstack((np.ones(NUM_CLASSES), np.full(NUM_CLASSES, 0.998)))

    low, _ = calibrate_thresholds(gt, low_scores, split="val")
    high, _ = calibrate_thresholds(gt, high_scores, split="val")

    np.testing.assert_allclose(low, 0.001)
    np.testing.assert_allclose(high, 0.999)


def test_threshold_ties_choose_closest_to_half_then_lower():
    candidates = np.array([0.4, 0.6, 0.7])
    f1 = np.array([0.9, 0.9, 0.9])
    assert select_best_threshold(candidates, f1) == pytest.approx(0.4)

    candidates = np.array([0.2, 0.4, 0.5, 0.8])
    f1 = np.array([1.0, 1.0, 1.0, 1.0])
    assert select_best_threshold(candidates, f1) == pytest.approx(0.5)


def test_calibration_is_val_only_and_no_positive_class_is_explicit():
    gt, scores = _fixture_arrays()
    with pytest.raises(ValueError, match="val"):
        calibrate_thresholds(gt, scores, split="test")

    thresholds, rows = calibrate_thresholds(gt, scores, split="val")
    assert thresholds.shape == (NUM_CLASSES,)
    assert thresholds[2] == pytest.approx(0.5)
    assert rows[2]["degenerate_reason"] == "no_positive_gt"
    assert rows[2]["best_f1"] == 0.0


def test_no_predicted_positive_is_finite_zero_not_nan():
    gt = np.zeros((2, NUM_CLASSES), dtype=np.int64)
    gt[0, 0] = 1
    scores = np.zeros((2, NUM_CLASSES), dtype=np.float64)
    summary, per_class, _ = compute_evaluation_metrics(
        gt, scores, np.full(NUM_CLASSES, 0.999)
    )
    assert per_class[0]["precision"] == 0.0
    assert per_class[0]["recall"] == 0.0
    assert per_class[0]["f1"] == 0.0
    assert all(np.isfinite(value) for value in summary.values())


def test_evaluation_mapping_is_exact_ordered_18_class_contract():
    mapping = evaluation_label_mapping()
    assert [row["model_index"] for row in mapping] == list(range(18))
    assert [row["source_id"] for row in mapping] == list(range(13)) + list(
        range(14, 19)
    )


def test_evaluation_dataset_rejects_duplicate_primary_key(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        _manifest_row("images/a.png", image_id="one"),
        _manifest_row("images/a.png", image_id="two"),
    ]
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="duplicate.*split/file_name"):
        TobaccoEvaluationDataset(manifest, tmp_path, "val")


def test_evaluation_dataset_binds_manifest_identity_and_audits_image_ids(tmp_path):
    data_root = tmp_path / "data"
    _write_image(data_root, "val", "images/a.png", 32)
    _write_image(data_root, "val", "images/b.png", 224)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _manifest_row("images/a.png", image_id="shared", positives=(0,)),
            _manifest_row("images/b.png", image_id="shared", positives=(1,)),
        ],
    )

    dataset = TobaccoEvaluationDataset(manifest, data_root, "val", _to_tensor)

    assert dataset.identity_audit == {
        "row_count": 2,
        "global_image_id_unique": False,
        "duplicate_image_id_count": 1,
    }
    assert dataset[1]["image_id"] == "shared"
    assert dataset[1]["file_name"] == "images/b.png"
    assert dataset[1]["split"] == "val"


class _ReverseSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(reversed(range(len(self.data_source))))

    def __len__(self):
        return len(self.data_source)


class _PixelLogitModel(torch.nn.Module):
    def forward(self, images, targets=None):
        base = images[:, 0, 0, 0]
        logits = torch.zeros((images.shape[0], NUM_CLASSES), dtype=images.dtype)
        logits[:, 0] = base
        return {"logits": logits}


def test_collect_model_outputs_keeps_reordered_identity_gt_score_alignment(tmp_path):
    data_root = tmp_path / "data"
    _write_image(data_root, "val", "images/low.png", 0)
    _write_image(data_root, "val", "images/high.png", 255)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _manifest_row("images/low.png", image_id="low", positives=(0,)),
            _manifest_row("images/high.png", image_id="high", positives=(1,)),
        ],
    )
    dataset = TobaccoEvaluationDataset(manifest, data_root, "val", _to_tensor)
    loader = DataLoader(dataset, batch_size=2, sampler=_ReverseSampler(dataset))

    identities, gt, scores = collect_model_outputs(
        _PixelLogitModel(), loader, torch.device("cpu")
    )

    assert [row["image_id"] for row in identities] == ["high", "low"]
    assert gt[0, 1] == 1 and gt[1, 0] == 1
    assert scores[0, 0] == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())
    assert scores[1, 0] == pytest.approx(0.5)


def test_evaluation_dataset_does_not_hide_image_decode_failure(tmp_path):
    data_root = tmp_path / "data"
    bad_path = data_root / "test" / "images" / "task_16" / "IMG_7950.png"
    bad_path.parent.mkdir(parents=True)
    bad_path.write_bytes(b"not a png")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _manifest_row(
                "images/task_16/IMG_7950.png",
                split="test",
                image_id="broken",
                positives=(16,),
            )
        ],
    )
    dataset = TobaccoEvaluationDataset(manifest, data_root, "test", _to_tensor)

    with pytest.raises(UnidentifiedImageError):
        dataset[0]


def test_verified_context_checks_complete_checkpoint_identity_chain(tmp_path):
    cfg, metadata = _checkpoint_fixture(tmp_path)

    context = verify_tobacco_evaluation_context(cfg)

    assert context["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert context["manifest_sha256"] == metadata["manifest_sha256"]
    assert context["best_epoch"] == 10
    assert context["best_val_mAP"] == 0.25
    assert context["model_variant"] == "EMA"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("manifest_sha256", "0" * 64, "manifest"),
        ("label_mapping", [], "mapping"),
        ("num_classes", 19, "num_classes"),
        ("excluded_source_class_id", 12, "excluded"),
        ("model", "other", "architecture"),
        ("arch", "other", "backbone"),
        ("img_size", 224, "image_size"),
        ("selection_split", "test", "selection_split"),
        ("selection_variant", "raw", "EMA"),
    ],
)
def test_verified_context_rejects_config_metadata_mismatch(
    tmp_path, field, value, message
):
    cfg, _ = _checkpoint_fixture(tmp_path)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=message):
        verify_tobacco_evaluation_context(cfg)


def test_verified_context_rejects_changed_manifest_and_checkpoint_bytes(tmp_path):
    cfg, _ = _checkpoint_fixture(tmp_path)
    Path(cfg.manifest_path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA"):
        verify_tobacco_evaluation_context(cfg)

    cfg, _ = _checkpoint_fixture(tmp_path)
    with Path(cfg.ckpt_ema_best_path).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        verify_tobacco_evaluation_context(cfg)


def test_load_verified_state_dict_accepts_uniform_module_prefix_strictly(tmp_path):
    source = torch.nn.Linear(2, 2)
    prefixed = {"module." + key: value for key, value in source.state_dict().items()}
    cfg, _ = _checkpoint_fixture(tmp_path, prefixed)
    target = torch.nn.Linear(2, 2)

    context = load_verified_ema_state_dict(target, cfg)

    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)
    assert context["model_variant"] == "EMA"


def test_load_verified_state_dict_rejects_mixed_prefix_and_shape_mismatch(tmp_path):
    source = torch.nn.Linear(2, 2).state_dict()
    mixed = {"module.weight": source["weight"], "bias": source["bias"]}
    cfg, _ = _checkpoint_fixture(tmp_path, mixed)
    with pytest.raises(ValueError, match="mixed.*module"):
        load_verified_ema_state_dict(torch.nn.Linear(2, 2), cfg)

    cfg, _ = _checkpoint_fixture(tmp_path, torch.nn.Linear(3, 2).state_dict())
    with pytest.raises(RuntimeError):
        load_verified_ema_state_dict(torch.nn.Linear(2, 2), cfg)


def test_threshold_artifact_round_trip_validates_full_provenance(tmp_path):
    cfg, _ = _checkpoint_fixture(tmp_path)
    context = verify_tobacco_evaluation_context(cfg)
    thresholds = np.linspace(0.1, 0.9, NUM_CLASSES)
    rows = [
        {
            **entry,
            "threshold": float(thresholds[index]),
            "best_f1": 0.5,
            "degenerate_reason": None,
        }
        for index, entry in enumerate(evaluation_label_mapping())
    ]
    artifact = build_threshold_artifact(thresholds, rows, context)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    restored = load_verified_thresholds(path, context)

    np.testing.assert_allclose(restored, thresholds)
    assert artifact["calibration_split"] == "val"
    assert artifact["operator"] == ">="
    assert artifact["search"] == {
        "minimum": 0.001,
        "maximum": 0.999,
        "step": 0.001,
        "objective": "per_class_f1",
        "tie_break": ["closest_to_0.5", "lower_threshold"],
    }


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda value: value.update({"checkpoint_sha256": "f" * 64}), "checkpoint"),
        (lambda value: value.update({"manifest_sha256": "e" * 64}), "manifest"),
        (lambda value: value.update({"best_epoch": 11}), "epoch"),
        (lambda value: value.update({"image_size": 224}), "image_size"),
        (lambda value: value.update({"calibration_split": "test"}), "val"),
        (lambda value: value.update({"operator": ">"}), "operator"),
        (lambda value: value["label_mapping"].reverse(), "mapping"),
        (lambda value: value.update({"thresholds": [0.5]}), "18"),
    ],
)
def test_threshold_artifact_rejects_identity_or_protocol_mismatch(
    tmp_path, mutation, message
):
    cfg, _ = _checkpoint_fixture(tmp_path)
    context = verify_tobacco_evaluation_context(cfg)
    artifact = build_threshold_artifact(
        np.full(NUM_CLASSES, 0.5),
        [
            {
                **entry,
                "threshold": 0.5,
                "best_f1": 0.0,
                "degenerate_reason": None,
            }
            for entry in evaluation_label_mapping()
        ],
        context,
    )
    mutation(artifact)
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_verified_thresholds(path, context)


def test_threshold_artifact_rejects_calibration_row_mismatch(tmp_path):
    cfg, _ = _checkpoint_fixture(tmp_path)
    context = verify_tobacco_evaluation_context(cfg)
    thresholds = np.full(NUM_CLASSES, 0.5)
    rows = [
        {
            **entry,
            "threshold": 0.5,
            "best_f1": 0.0,
            "degenerate_reason": None,
        }
        for entry in evaluation_label_mapping()
    ]
    artifact = build_threshold_artifact(thresholds, rows, context)
    artifact["calibration"][0]["source_id"] = 99
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        load_verified_thresholds(path, context)


def _artifact_bundle(tmp_path, split="val"):
    cfg, _ = _checkpoint_fixture(tmp_path)
    context = verify_tobacco_evaluation_context(cfg)
    gt = np.zeros((3, NUM_CLASSES), dtype=np.int64)
    gt[0, 0] = 1
    gt[1, (0, 1)] = 1
    scores = np.full((3, NUM_CLASSES), 0.1, dtype=np.float64)
    scores[0, 0] = 0.9
    scores[1, 0] = 0.4
    scores[1, 1] = 0.8
    thresholds = np.full(NUM_CLASSES, 0.5)
    summary, per_class, predictions = compute_evaluation_metrics(
        gt, scores, thresholds
    )
    identities = [
        {
            "image_id": "{}-{}".format(split, index),
            "file_name": "images/{}.png".format(index),
            "split": split,
        }
        for index in range(3)
    ]
    calibration_rows = [
        {
            **entry,
            "threshold": 0.5,
            "best_f1": 0.0,
            "degenerate_reason": None,
        }
        for entry in evaluation_label_mapping()
    ]
    threshold_artifact = build_threshold_artifact(
        thresholds, calibration_rows, context
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


def test_calibration_artifacts_round_trip_and_recompute_from_csv(tmp_path):
    bundle = _artifact_bundle(tmp_path)
    target = tmp_path / "outputs" / "calibration"

    write_calibration_artifacts(target, bundle)
    validated = validate_calibration_artifacts(target, bundle["context"])

    assert sorted(path.name for path in target.iterdir()) == [
        "thresholds.json",
        "val_metrics.json",
        "val_per_class.csv",
        "val_predictions.csv",
    ]
    assert validated["summary"] == pytest.approx(bundle["summary"])
    with (target / "val_predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["split"] == "val"
    assert rows[0]["gt_0_y00_c02_weather_fleck"] == "1"
    assert float(rows[0]["score_0_y00_c02_weather_fleck"]) == pytest.approx(0.9)
    assert rows[0]["pred_0_y00_c02_weather_fleck"] == "1"


def test_evaluation_artifacts_round_trip_without_threshold_file(tmp_path):
    bundle = _artifact_bundle(tmp_path, split="test")
    target = tmp_path / "outputs" / "evaluation"

    write_evaluation_artifacts(target, bundle)
    validated = validate_evaluation_artifacts(target, bundle["context"])

    assert sorted(path.name for path in target.iterdir()) == [
        "test_metrics.json",
        "test_per_class.csv",
        "test_predictions.csv",
    ]
    assert validated["sample_count"] == 3


def test_evaluation_artifact_validation_is_read_only(tmp_path, monkeypatch):
    bundle = _artifact_bundle(tmp_path, split="test")
    target = tmp_path / "outputs" / "evaluation"
    write_evaluation_artifacts(target, bundle)
    before = _tree_bytes(target)

    def forbidden_tempfile(*args, **kwargs):
        raise AssertionError("artifact validation attempted to create a temporary file")

    monkeypatch.setattr("lib.evaluation_protocol.tempfile.mkstemp", forbidden_tempfile)
    validate_evaluation_artifacts(target, bundle["context"])

    assert _tree_bytes(target) == before


def test_artifact_validation_rejects_identity_and_prediction_tampering(tmp_path):
    bundle = _artifact_bundle(tmp_path)
    target = tmp_path / "calibration"
    write_calibration_artifacts(target, bundle)
    path = target / "val_predictions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    fieldnames = list(rows[0])
    rows[1]["file_name"] = rows[0]["file_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="duplicate"):
        validate_calibration_artifacts(target, bundle["context"])

    write_calibration_artifacts(target, bundle)
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0]["pred_0_y00_c02_weather_fleck"] = "0"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="recomputed prediction"):
        validate_calibration_artifacts(target, bundle["context"])


def _tree_bytes(path):
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def test_existing_nonempty_artifact_directory_is_replaced_as_a_whole(tmp_path):
    bundle = _artifact_bundle(tmp_path)
    target = tmp_path / "calibration"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    write_calibration_artifacts(target, bundle)

    assert not (target / "old.txt").exists()
    validate_calibration_artifacts(target, bundle["context"])


@pytest.mark.parametrize("fail_call", [1, 2])
def test_directory_replace_failure_rolls_back_original_tree(tmp_path, fail_call):
    bundle = _artifact_bundle(tmp_path)
    target = tmp_path / "calibration"
    target.mkdir()
    (target / "old.txt").write_text("old bytes", encoding="utf-8")
    before = _tree_bytes(target)
    calls = []

    def failing_replace(source, destination):
        calls.append((source, destination))
        if len(calls) == fail_call:
            raise OSError("injected replace failure")
        return Path(source).replace(destination)

    with pytest.raises(OSError, match="injected"):
        write_calibration_artifacts(target, bundle, replace_fn=failing_replace)

    assert _tree_bytes(target) == before


def test_post_publish_validation_failure_restores_original_tree(tmp_path):
    bundle = _artifact_bundle(tmp_path)
    target = tmp_path / "calibration"
    target.mkdir()
    (target / "old.txt").write_text("old bytes", encoding="utf-8")
    before = _tree_bytes(target)

    def validator(path, context):
        if Path(path) == target:
            raise ValueError("injected published validation failure")
        return validate_calibration_artifacts(path, context)

    with pytest.raises(ValueError, match="published validation"):
        write_calibration_artifacts(target, bundle, validate_fn=validator)

    assert _tree_bytes(target) == before


def test_rollback_failure_raises_transaction_error_and_retains_backup(tmp_path):
    bundle = _artifact_bundle(tmp_path)
    target = tmp_path / "calibration"
    target.mkdir()
    (target / "old.txt").write_text("old bytes", encoding="utf-8")
    calls = []

    def failing_replace(source, destination):
        calls.append((source, destination))
        if len(calls) in (2, 3):
            raise OSError("injected")
        return Path(source).replace(destination)

    with pytest.raises(ArtifactTransactionError, match="rollback"):
        write_calibration_artifacts(target, bundle, replace_fn=failing_replace)

    backups = list(tmp_path.glob(".calibration.backup.*"))
    assert backups and (backups[0] / "old.txt").read_text(encoding="utf-8") == "old bytes"


def test_staging_write_failure_does_not_touch_existing_target(tmp_path):
    bundle = _artifact_bundle(tmp_path)
    target = tmp_path / "calibration"
    target.mkdir()
    (target / "old.txt").write_text("old bytes", encoding="utf-8")
    before = _tree_bytes(target)

    def fail_writer(stage, value):
        raise OSError("injected staging write failure")

    with pytest.raises(OSError, match="staging"):
        write_calibration_artifacts(target, bundle, files_writer=fail_writer)

    assert _tree_bytes(target) == before


class _TinyEvaluationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, NUM_CLASSES)

    def forward(self, images, targets=None):
        return {"logits": self.linear(images)}


def _workflow_fixture(tmp_path, constructed_splits):
    model = _TinyEvaluationModel()
    cfg, _ = _checkpoint_fixture(tmp_path, model.state_dict())
    cfg.data_root = str(tmp_path / "unused-data")
    cfg.batch_size = 2
    cfg.num_workers = 0

    class SyntheticEvaluationDataset(torch.utils.data.Dataset):
        def __init__(self, manifest_path, data_root, split, transform=None):
            constructed_splits.append(split)
            self.split = split

        def __len__(self):
            return 3

        def __getitem__(self, index):
            target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            if index == 0:
                target[0] = 1
            elif index == 1:
                target[1] = 1
            return {
                "img": torch.tensor([float(index)], dtype=torch.float32),
                "target": target,
                "img_path": "unused",
                "image_id": "{}-{}".format(self.split, index),
                "file_name": "images/{}.png".format(index),
                "split": self.split,
            }

    return cfg, model, SyntheticEvaluationDataset


def test_real_val_workflow_verifies_context_then_uses_only_val(tmp_path):
    constructed = []
    cfg, source_model, dataset_cls = _workflow_fixture(tmp_path, constructed)
    events = []

    def model_factory(value):
        events.append("model_factory")
        return _TinyEvaluationModel()

    def verifier(value):
        events.append("verify")
        return verify_tobacco_evaluation_context(value)

    output_root = tmp_path / "outputs"
    result = run_val_calibration(
        cfg,
        output_root,
        model_factory=model_factory,
        dataset_cls=dataset_cls,
        transform_factory=lambda value, training: None,
        device=torch.device("cpu"),
        context_verifier=verifier,
    )

    assert events[0] == "verify"
    assert constructed == ["val"]
    assert result["split"] == "val"
    validate_calibration_artifacts(output_root / "calibration", result["context"])


def test_val_calibration_canonicalizes_rows_before_official_voc_tie_scoring(tmp_path):
    constructed = []
    cfg, _, dataset_cls = _workflow_fixture(tmp_path, constructed)
    identities = [
        {
            "image_id": "val-{}".format(index),
            "file_name": "images/{}.png".format(file_number),
            "split": "val",
        }
        for index, file_number in enumerate((0, 1, 2, 3, 5, 4))
    ]
    gt = np.zeros((6, NUM_CLASSES), dtype=np.int64)
    gt[5, 0] = 1
    scores = np.full((6, NUM_CLASSES), np.float32(0.12345679), dtype=np.float32)

    result = run_val_calibration(
        cfg,
        tmp_path / "outputs",
        model_factory=lambda value: _TinyEvaluationModel(),
        dataset_cls=dataset_cls,
        transform_factory=lambda value, training: None,
        device=torch.device("cpu"),
        collector=lambda model, loader, device: (identities, gt, scores),
    )

    artifact_dir = Path(result["output_dir"])
    validated = validate_calibration_artifacts(artifact_dir, result["context"])
    with (artifact_dir / "val_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [row["file_name"] for row in rows] == [
        "images/0.png",
        "images/1.png",
        "images/2.png",
        "images/3.png",
        "images/4.png",
        "images/5.png",
    ]
    reloaded_gt = np.asarray(
        [
            [
                int(row["gt_{}_{}".format(i, entry["label"])])
                for i, entry in enumerate(evaluation_label_mapping())
            ]
            for row in rows
        ],
        dtype=np.int64,
    )
    reloaded_scores = np.asarray(
        [
            [
                float(row["score_{}_{}".format(i, entry["label"])])
                for i, entry in enumerate(evaluation_label_mapping())
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    canonical_order = np.asarray([0, 1, 2, 3, 5, 4])
    assert scores.dtype == np.float32
    assert scores.shape == (6, NUM_CLASSES)
    assert reloaded_scores.dtype == np.float64
    assert reloaded_scores.shape == scores.shape
    assert gt.dtype == np.int64
    assert reloaded_gt.dtype == np.int64
    np.testing.assert_array_equal(reloaded_gt, gt[canonical_order])
    np.testing.assert_array_equal(
        reloaded_scores, scores[canonical_order].astype(np.float64)
    )
    expected_ap, expected_map = VOCmAP(NUM_CLASSES, year="2012").compute(
        scores=reloaded_scores, targets=reloaded_gt
    )
    original_ap, original_map = VOCmAP(NUM_CLASSES, year="2012").compute(
        scores=scores.astype(np.float64), targets=gt
    )
    assert original_map == pytest.approx(1.0 / NUM_CLASSES, abs=0, rel=0)
    assert expected_map == pytest.approx(0.5 / NUM_CLASSES, abs=0, rel=0)
    np.testing.assert_array_equal(
        np.abs(original_ap - expected_ap),
        np.asarray([0.5] + [0.0] * (NUM_CLASSES - 1)),
    )
    np.testing.assert_allclose(
        [row["voc2012_ap"] for row in validated["per_class"]],
        expected_ap,
        atol=0,
        rtol=0,
    )
    assert validated["summary"]["voc2012_map"] == pytest.approx(
        expected_map, abs=0, rel=0
    )


def test_real_test_workflow_loads_fixed_thresholds_and_never_calibrates(
    tmp_path, monkeypatch
):
    constructed = []
    cfg, source_model, dataset_cls = _workflow_fixture(tmp_path, constructed)
    output_root = tmp_path / "outputs"
    common = dict(
        model_factory=lambda value: _TinyEvaluationModel(),
        dataset_cls=dataset_cls,
        transform_factory=lambda value, training: None,
        device=torch.device("cpu"),
    )
    run_val_calibration(cfg, output_root, **common)
    calibration_before = _tree_bytes(output_root / "calibration")
    constructed.clear()

    def forbidden_calibration(*args, **kwargs):
        raise AssertionError("test evaluation attempted threshold calibration")

    monkeypatch.setattr(
        "lib.evaluation_protocol.calibrate_thresholds", forbidden_calibration
    )
    result = run_test_evaluation(
        cfg,
        output_root,
        output_root / "calibration" / "thresholds.json",
        **common,
    )

    assert constructed == ["test"]
    assert result["split"] == "test"
    assert _tree_bytes(output_root / "calibration") == calibration_before
    validate_evaluation_artifacts(output_root / "evaluation", result["context"])


def test_test_workflow_rejects_thresholds_before_dataset_construction(tmp_path):
    constructed = []
    cfg, _, dataset_cls = _workflow_fixture(tmp_path, constructed)
    output_root = tmp_path / "outputs"
    threshold_path = tmp_path / "bad-thresholds.json"
    threshold_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="val"):
        run_test_evaluation(
            cfg,
            output_root,
            threshold_path,
            model_factory=lambda value: _TinyEvaluationModel(),
            dataset_cls=dataset_cls,
            transform_factory=lambda value, training: None,
            device=torch.device("cpu"),
        )
    assert constructed == []


def _import_evaluate(monkeypatch):
    inplace_stub = types.ModuleType("inplace_abn")
    inplace_stub.InPlaceABN = torch.nn.BatchNorm2d
    inplace_stub.ABN = torch.nn.BatchNorm2d
    monkeypatch.setitem(sys.modules, "inplace_abn", inplace_stub)
    sys.modules.pop("evaluate", None)
    import importlib

    return importlib.import_module("evaluate")


def test_evaluate_main_dispatches_explicit_tobacco_operations_safely(
    tmp_path, monkeypatch
):
    evaluate = _import_evaluate(monkeypatch)
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    (exp_dir / "config.yaml").write_text(
        "data: tobacco\nbatch_size: 8\nthreshold: 0.5\n", encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(
        evaluate,
        "run_val_calibration",
        lambda cfg, output_root: calls.append(("val", cfg.data, Path(output_root))),
    )
    monkeypatch.setattr(
        evaluate,
        "run_test_evaluation",
        lambda cfg, output_root, thresholds_path: calls.append(
            ("test", cfg.data, Path(thresholds_path))
        ),
    )

    evaluate.main(
        [
            "--exp-dir",
            str(exp_dir),
            "--operation",
            "val-calibration",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    evaluate.main(
        [
            "--exp-dir",
            str(exp_dir),
            "--operation",
            "test-evaluation",
            "--output-root",
            str(tmp_path / "out"),
            "--thresholds-path",
            str(tmp_path / "thresholds.json"),
        ]
    )
    assert [call[0] for call in calls] == ["val", "test"]


def test_legacy_evaluator_still_uses_test_path_and_module_prefix(tmp_path, monkeypatch):
    evaluate = _import_evaluate(monkeypatch)
    seen = {}

    class LegacyDataset:
        labels = ["a"]

        def __init__(self, cfg, path, training):
            seen["dataset"] = (path, training)

        def __len__(self):
            return 1

    class LegacyModel:
        def cuda(self):
            return self

    monkeypatch.setattr(evaluate, "MLDataset", LegacyDataset)
    monkeypatch.setattr(evaluate, "create_model", lambda *args, **kwargs: LegacyModel())
    monkeypatch.setattr(evaluate, "DataLoader", lambda *args, **kwargs: [])
    cfg = Namespace(
        test_path="legacy/test.txt",
        batch_size=2,
        model="mlic",
        num_classes=1,
        ignore_path=str(tmp_path / "missing-ignore.npy"),
        threshold=0.5,
        topk=1,
    )

    evaluate.Evaluator(cfg)
    normalized = evaluate._normalize_legacy_state_dict(
        {"module.weight": torch.tensor([1.0]), "module.bias": torch.tensor([0.0])}
    )

    assert seen["dataset"] == ("legacy/test.txt", False)
    assert list(normalized) == ["weight", "bias"]
