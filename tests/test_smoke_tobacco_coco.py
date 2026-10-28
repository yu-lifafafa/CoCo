import csv
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from lib.tobacco_dataset import SOURCE_LABEL_COLUMNS, TobaccoManifestDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_clean_python(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_resnet101_and_mlic_import_without_optional_inplace_abn():
    result = _run_clean_python(
        """
        import builtins

        original_import = builtins.__import__

        def block_inplace_abn(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "inplace_abn" or name.startswith("inplace_abn.")):
                raise ModuleNotFoundError("blocked optional dependency", name="inplace_abn")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = block_inplace_abn

        import models
        from models.factory import backbone_entrypoint, model_entrypoint

        assert callable(backbone_entrypoint("resnet101")[0])
        assert callable(model_entrypoint("mlic"))
        try:
            backbone_entrypoint("tresnet_m")
        except RuntimeError as error:
            assert "inplace_abn" in str(error)
        else:
            raise AssertionError("Unavailable TResNet did not fail explicitly")
        """
    )
    assert result.returncode == 0, result.stderr


def test_tresnet_entrypoints_remain_registered_when_dependency_is_available():
    result = _run_clean_python(
        """
        import sys
        import types
        import torch.nn as nn

        fake = types.ModuleType("inplace_abn")
        fake.InPlaceABN = nn.BatchNorm2d
        fake.ABN = nn.BatchNorm2d
        sys.modules["inplace_abn"] = fake

        import models
        from models.factory import backbone_entrypoint

        for name in (
            "tresnet_m",
            "tresnet_l",
            "tresnet_xl",
            "tresnet_lv2",
            "tresnet_m21k",
            "tresnet_l21k",
        ):
            assert callable(backbone_entrypoint(name)[0])
        """
    )
    assert result.returncode == 0, result.stderr


def _manifest_row(file_name, split, positives, identity):
    row = {
        "image_id": identity,
        "file_name": file_name,
        "split": split,
        "source_file": "source.json",
        "source_image_id": identity,
        "original_file_name": Path(file_name).name,
        "is_healthy": "0" if positives else "1",
        "positive_label_count": str(len(positives)),
    }
    row.update({column: "0" for column in SOURCE_LABEL_COLUMNS})
    for source_id in positives:
        row[SOURCE_LABEL_COLUMNS[source_id]] = "1"
    return row


def _write_manifest_fixture(tmp_path, rows):
    manifest_path = tmp_path / "image_manifest_updated.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data_root = tmp_path / "data"
    for row in rows:
        image_path = data_root / row["split"] / Path(*row["file_name"].split("/"))
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(20, 30, 40)).save(image_path)
    return manifest_path, data_root


def _smoke_args(tmp_path, manifest_path, data_root, **overrides):
    values = {
        "manifest_path": manifest_path,
        "data_root": data_root,
        "img_size": 384,
        "batch_size": 4,
        "device": "cuda:0",
        "amp": True,
        "seed": 42,
        "backbone_init": "torch-cache",
        "backbone_weights": None,
        "output_dir": tmp_path / "smoke-output",
        "max_steps": 4,
        "scenarios": ["healthy", "single", "multi", "mixed"],
        "expected_manifest_sha256": (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_path.is_file()
            else "0" * 64
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _selection_fixture(tmp_path):
    rows = [
        _manifest_row("images/healthy.jpg", "train", (), "healthy"),
        _manifest_row("images/root-only.jpg", "train", (13,), "root-only"),
        _manifest_row("images/single-a.jpg", "train", (1,), "single-a"),
        _manifest_row("images/single-b.jpg", "train", (14,), "single-b"),
        _manifest_row("images/multi-a.jpg", "train", (0, 2), "multi-a"),
        _manifest_row("images/multi-b.jpg", "train", (3, 13, 18), "multi-b"),
        _manifest_row("images/val.jpg", "val", (4,), "val"),
        _manifest_row("images/test.jpg", "test", (5,), "test"),
    ]
    manifest_path, data_root = _write_manifest_fixture(tmp_path, rows)
    dataset = TobaccoManifestDataset(manifest_path, data_root, "train")
    return manifest_path, data_root, dataset


def test_smoke_cli_defaults_and_required_arguments(tmp_path):
    from scripts.smoke_tobacco_coco import parse_args

    manifest_path = tmp_path / "manifest.csv"
    data_root = tmp_path / "data"
    output_dir = tmp_path / "out"
    args = parse_args(
        [
            "--manifest-path",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.img_size == 384
    assert args.batch_size == 1
    assert args.device == "cuda:0"
    assert args.amp is True
    assert args.seed == 42
    assert args.backbone_init == "torch-cache"
    assert args.max_steps == 1
    assert args.scenarios == ["healthy"]
    with pytest.raises(SystemExit):
        parse_args([])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"img_size": 383}, "divisible by 32"),
        ({"img_size": 0}, "positive"),
        ({"batch_size": 0}, "batch size"),
        ({"device": "cpu"}, "CUDA device"),
        ({"max_steps": 0}, "max steps"),
        ({"max_steps": 9}, "max steps"),
        ({"scenarios": ["healthy", "healthy"]}, "duplicate"),
    ],
)
def test_validate_smoke_args_rejects_invalid_control_values(
    tmp_path, override, message
):
    from scripts.smoke_tobacco_coco import validate_smoke_args

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(tmp_path, manifest_path, data_root, **override)
    with pytest.raises(ValueError, match=message):
        validate_smoke_args(args, PROJECT_ROOT)


def test_validate_smoke_args_checks_paths_hash_and_initialization_mode(tmp_path):
    from scripts.smoke_tobacco_coco import validate_smoke_args

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    valid = _smoke_args(tmp_path, manifest_path, data_root)
    validated = validate_smoke_args(valid, PROJECT_ROOT)
    assert validated["manifest_sha256"] == valid.expected_manifest_sha256

    cases = [
        _smoke_args(tmp_path, tmp_path / "missing.csv", data_root),
        _smoke_args(tmp_path, manifest_path, tmp_path / "missing-data"),
        _smoke_args(
            tmp_path,
            manifest_path,
            data_root,
            expected_manifest_sha256="0" * 64,
        ),
        _smoke_args(
            tmp_path,
            manifest_path,
            data_root,
            backbone_init="local-file",
            backbone_weights=None,
        ),
    ]
    for args in cases:
        with pytest.raises((ValueError, FileNotFoundError)):
            validate_smoke_args(args, PROJECT_ROOT)


@pytest.mark.parametrize(
    "unsafe_output",
    [
        Path("experiments") / "mlic_tobacco" / "exp1",
        Path("checkpoints") / "smoke",
        Path("outputs") / "tobacco_coco" / "evaluation",
    ],
)
def test_validate_smoke_args_rejects_formal_artifact_destinations(
    tmp_path, unsafe_output
):
    from scripts.smoke_tobacco_coco import validate_smoke_args

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(
        tmp_path,
        manifest_path,
        data_root,
        output_dir=PROJECT_ROOT / unsafe_output,
    )
    with pytest.raises(ValueError, match="smoke output"):
        validate_smoke_args(args, PROJECT_ROOT)


def test_load_train_smoke_dataset_constructs_only_train(tmp_path):
    from scripts.smoke_tobacco_coco import load_train_smoke_dataset

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    calls = []

    class DatasetSpy:
        def __init__(self, manifest_path, data_root, split, transform):
            calls.append((Path(manifest_path), Path(data_root), split, transform))
            if split != "train":
                raise AssertionError("Smoke must never construct val/test")

    transform = object()
    args = _smoke_args(tmp_path, manifest_path, data_root)
    dataset = load_train_smoke_dataset(args, transform, dataset_cls=DatasetSpy)
    assert isinstance(dataset, DatasetSpy)
    assert calls == [(manifest_path, data_root, "train", transform)]


@pytest.mark.parametrize("scenario", ["healthy", "single", "multi", "mixed"])
def test_scenario_selection_is_deterministic_exact_and_train_only(tmp_path, scenario):
    from scripts.smoke_tobacco_coco import select_scenario_indices

    _, _, dataset = _selection_fixture(tmp_path)
    batch_size = 3 if scenario == "mixed" else 1
    first = select_scenario_indices(dataset.records, scenario, batch_size, seed=42)
    second = select_scenario_indices(dataset.records, scenario, batch_size, seed=42)
    assert first == second
    assert first["status"] == "PASS"
    assert len(first["indices"]) == batch_size
    assert len(set(first["indices"])) == batch_size
    selected = [dataset.records[index] for index in first["indices"]]
    if scenario == "healthy":
        assert all(record.is_healthy for record in selected)
        assert all(record.source_positive_count == 0 for record in selected)
    elif scenario == "single":
        assert all(record.task_positive_count == 1 for record in selected)
    elif scenario == "multi":
        assert all(record.task_positive_count >= 2 for record in selected)
    else:
        assert any(record.is_healthy for record in selected)
        assert any(record.task_positive_count == 1 for record in selected)
        assert any(record.task_positive_count >= 2 for record in selected)


def test_root_knot_only_zero_task_target_is_not_selected_as_healthy(tmp_path):
    from scripts.smoke_tobacco_coco import select_scenario_indices

    _, _, dataset = _selection_fixture(tmp_path)
    result = select_scenario_indices(dataset.records, "healthy", 1, seed=42)
    selected = dataset.records[result["indices"][0]]
    assert selected.file_name == "images/healthy.jpg"
    assert selected.is_healthy is True


def test_insufficient_or_too_small_mixed_batches_are_not_verified(tmp_path):
    from scripts.smoke_tobacco_coco import select_scenario_indices

    _, _, dataset = _selection_fixture(tmp_path)
    too_small = select_scenario_indices(dataset.records, "mixed", 2, seed=42)
    too_many_healthy = select_scenario_indices(
        dataset.records, "healthy", 2, seed=42
    )
    assert too_small["status"] == "NOT_VERIFIED"
    assert "at least 3" in too_small["reason"]
    assert too_many_healthy["status"] == "NOT_VERIFIED"
    assert too_many_healthy["indices"] == []


def test_new_unexecuted_smoke_result_never_fakes_runtime_evidence(tmp_path):
    from scripts.smoke_tobacco_coco import new_smoke_result

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(tmp_path, manifest_path, data_root)
    result = new_smoke_result(args)
    assert result["status"] == "NOT_VERIFIED"
    assert result["gpu_executed"] is False
    assert result["gpu_name"] is None
    assert result["peak_allocated_mib"] is None
    assert result["peak_reserved_mib"] is None
    assert result["duration_seconds"] is None
    assert result["classification_loss"] is None
    assert result["contrastive_loss"] is None
    assert result["total_loss"] is None
    assert set(result["allowed_statuses"]) == {"PASS", "FAIL", "NOT_VERIFIED"}


def test_torch_cache_mode_fails_closed_when_resnet101_weights_are_absent(
    tmp_path, monkeypatch
):
    from scripts.smoke_tobacco_coco import prepare_backbone_initialization

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(tmp_path, manifest_path, data_root)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty-torch-home"))
    with pytest.raises(FileNotFoundError, match="resnet101-63fe2227.pth"):
        prepare_backbone_initialization(args, args.output_dir)


def test_local_file_mode_copies_weights_to_isolated_cache_with_same_sha(tmp_path):
    from scripts.smoke_tobacco_coco import prepare_backbone_initialization

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    source_weights = tmp_path / "resnet101-63fe2227.pth"
    source_weights.write_bytes(b"synthetic-weight-evidence")
    args = _smoke_args(
        tmp_path,
        manifest_path,
        data_root,
        backbone_init="local-file",
        backbone_weights=source_weights,
    )
    evidence = prepare_backbone_initialization(args, args.output_dir)
    copied = Path(evidence["resolved_path"])
    assert copied == (
        args.output_dir / "torch_home" / "hub" / "checkpoints" / source_weights.name
    )
    assert copied.read_bytes() == source_weights.read_bytes()
    assert evidence["source_sha256"] == evidence["resolved_sha256"]
    assert evidence["mode"] == "local-file"
    assert os.environ["TORCH_HOME"] == str(args.output_dir / "torch_home")


def test_allow_download_mode_records_network_provenance_without_fake_hash(tmp_path):
    from scripts.smoke_tobacco_coco import prepare_backbone_initialization

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(
        tmp_path, manifest_path, data_root, backbone_init="allow-download"
    )
    evidence = prepare_backbone_initialization(args, args.output_dir)
    assert evidence["mode"] == "allow-download"
    assert evidence["network_allowed"] is True
    assert evidence["resolved_path"] is None
    assert evidence["resolved_sha256"] is None


def test_cache_failure_occurs_before_real_model_factory_is_called(tmp_path, monkeypatch):
    import scripts.smoke_tobacco_coco as smoke

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(tmp_path, manifest_path, data_root)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "empty-torch-home"))
    monkeypatch.setattr(smoke, "cuda_is_available", lambda: True)
    called = []

    def trap(*args, **kwargs):
        called.append(True)
        raise AssertionError("model factory must not run before cache preflight")

    monkeypatch.setattr(smoke, "build_real_smoke_model", trap)
    with pytest.raises(FileNotFoundError, match="resnet101-63fe2227.pth"):
        smoke.run_smoke(args)
    assert called == []


def test_gpu_unavailable_run_is_not_verified_and_writes_null_runtime_metrics(
    tmp_path, monkeypatch
):
    import scripts.smoke_tobacco_coco as smoke

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(tmp_path, manifest_path, data_root)
    monkeypatch.setattr(smoke, "cuda_is_available", lambda: False)
    result = smoke.run_smoke(args)
    assert result["status"] == "NOT_VERIFIED"
    assert result["gpu_executed"] is False
    assert result["reason"] == "CUDA is not available"
    for key in (
        "gpu_name",
        "classification_loss",
        "contrastive_loss",
        "total_loss",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "duration_seconds",
    ):
        assert result[key] is None
    saved = json.loads((args.output_dir / "smoke_result.json").read_text("utf-8"))
    assert saved == result
    assert not list(args.output_dir.rglob("*.pth"))


def test_write_smoke_result_uses_smoke_only_atomic_outputs(tmp_path):
    from scripts.smoke_tobacco_coco import write_smoke_result

    output_dir = tmp_path / "smoke"
    result = {"status": "NOT_VERIFIED", "reason": "unit-test"}
    paths = write_smoke_result(output_dir, result)
    assert paths["json"].name == "smoke_result.json"
    assert paths["log"].name == "smoke_result.log"
    assert json.loads(paths["json"].read_text("utf-8")) == result
    assert "NOT_VERIFIED" in paths["log"].read_text("utf-8")
    assert not any("checkpoint" in path.name.lower() for path in output_dir.iterdir())
    assert not list(output_dir.glob("*.tmp"))


def test_build_smoke_cfg_is_fixed_to_real_resnet101_and_18_classes(tmp_path):
    from scripts.smoke_tobacco_coco import build_smoke_cfg

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    args = _smoke_args(tmp_path, manifest_path, data_root, img_size=384)
    cfg = build_smoke_cfg(args)
    assert cfg.model == "mlic"
    assert cfg.arch == "resnet101"
    assert cfg.num_classes == 18
    assert cfg.img_size == 384
    assert cfg.loss == "asl"
    assert cfg.lamda == 1.0
    assert cfg.optimizer == "AdamW"
    assert cfg.lr_scheduler == "OneCycleLR"


def test_main_gpu_unavailable_returns_distinct_not_verified_exit(tmp_path, monkeypatch):
    import scripts.smoke_tobacco_coco as smoke

    manifest_path, data_root, _ = _selection_fixture(tmp_path)
    output_dir = tmp_path / "smoke-main"
    monkeypatch.setattr(smoke, "cuda_is_available", lambda: False)
    exit_code = smoke.main(
        [
            "--manifest-path",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--expected-manifest-sha256",
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        ]
    )
    assert exit_code == 3
    assert json.loads((output_dir / "smoke_result.json").read_text("utf-8"))[
        "status"
    ] == "NOT_VERIFIED"


def test_loaded_backbone_must_exactly_match_pretrained_state_dict(tmp_path):
    from scripts.smoke_tobacco_coco import validate_loaded_backbone_weights

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)

    model = Wrapper()
    weight_path = tmp_path / "weights.pth"
    torch.save(model.backbone.state_dict(), weight_path)
    evidence = validate_loaded_backbone_weights(model, weight_path)
    assert evidence["verified_tensor_count"] == 2
    assert evidence["exact_state_dict_match"] is True

    incomplete = tmp_path / "incomplete.pth"
    torch.save({"weight": model.backbone.weight.detach().clone()}, incomplete)
    with pytest.raises(ValueError, match="key mismatch"):
        validate_loaded_backbone_weights(model, incomplete)
