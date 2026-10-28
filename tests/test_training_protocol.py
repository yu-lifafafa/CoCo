import csv
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import types

from PIL import Image
import numpy as np
import pytest
import torch
import torch.nn as nn


if "randaugment" not in sys.modules:
    randaugment_stub = types.ModuleType("randaugment")

    class RandAugment:
        def __call__(self, image):
            return image

    randaugment_stub.RandAugment = RandAugment
    sys.modules["randaugment"] = randaugment_stub


from lib.tobacco_dataset import SOURCE_LABEL_COLUMNS
from lib.training_protocol import (
    CheckpointTransactionError,
    TOBACCO_NUM_CLASSES,
    build_training_datasets,
    is_strict_improvement,
    save_best_ema_checkpoint,
    sha256_file,
    tobacco_label_mapping,
    training_log_interval,
    validate_finite_map,
    verify_best_checkpoint_integrity,
    should_clear_failed_experiment,
)
from lib.metrics import VOCmAP
from lib.utils import get_args, load_config, prepare_env


def _manifest_row(file_name, split, positives=()):
    row = {
        "file_name": file_name,
        "split": split,
        "is_healthy": "0" if positives else "1",
        "positive_label_count": str(len(positives)),
    }
    row.update({column: "0" for column in SOURCE_LABEL_COLUMNS})
    for source_id in positives:
        row[SOURCE_LABEL_COLUMNS[source_id]] = "1"
    return row


def _write_manifest_fixture(tmp_path):
    rows = [
        _manifest_row("images/task_01/train.jpg", "train", (0,)),
        _manifest_row("images/task_01/val.jpg", "val", (14,)),
        _manifest_row("images/task_01/missing-test.jpg", "test", (18,)),
    ]
    manifest_path = tmp_path / "image_manifest_updated.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    data_root = tmp_path / "data"
    for row in rows[:2]:
        image_path = data_root / row["split"] / Path(*row["file_name"].split("/"))
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_path)
    return manifest_path, data_root


def test_tobacco_label_mapping_is_ordered_and_excludes_source_13():
    mapping = tobacco_label_mapping()

    assert TOBACCO_NUM_CLASSES == 18
    assert [entry["model_index"] for entry in mapping] == list(range(18))
    assert [entry["source_id"] for entry in mapping] == list(range(13)) + list(
        range(14, 19)
    )
    assert [entry["label"] for entry in mapping] == list(
        SOURCE_LABEL_COLUMNS[:13] + SOURCE_LABEL_COLUMNS[14:]
    )


def test_tobacco_builder_constructs_exactly_train_and_val(monkeypatch):
    calls = []

    class RecordingDataset:
        def __init__(self, manifest_path, data_root, split, transform=None):
            if split == "test":
                raise AssertionError("test split must not be constructed")
            calls.append((manifest_path, data_root, split, transform))

    monkeypatch.setattr(
        "lib.training_protocol.build_coco_transform",
        lambda cfg, training: "train-transform" if training else "val-transform",
    )
    cfg = SimpleNamespace(
        data="tobacco",
        manifest_path="manifest.csv",
        data_root="data-root",
        num_classes=18,
    )

    train_dataset, val_dataset = build_training_datasets(
        cfg, tobacco_dataset_cls=RecordingDataset
    )

    assert isinstance(train_dataset, RecordingDataset)
    assert isinstance(val_dataset, RecordingDataset)
    assert calls == [
        ("manifest.csv", "data-root", "train", "train-transform"),
        ("manifest.csv", "data-root", "val", "val-transform"),
    ]


def test_tobacco_builder_does_not_open_missing_test_image(tmp_path):
    manifest_path, data_root = _write_manifest_fixture(tmp_path)
    cfg = SimpleNamespace(
        data="tobacco",
        manifest_path=str(manifest_path),
        data_root=str(data_root),
        num_classes=18,
        img_size=8,
    )

    train_dataset, val_dataset = build_training_datasets(cfg)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert train_dataset.split == "train"
    assert val_dataset.split == "val"
    assert not (data_root / "test" / "images" / "task_01" / "missing-test.jpg").exists()


def test_tobacco_builder_rejects_non_18_class_configuration():
    cfg = SimpleNamespace(
        data="tobacco",
        manifest_path="manifest.csv",
        data_root="data-root",
        num_classes=19,
    )

    with pytest.raises(ValueError, match="18"):
        build_training_datasets(cfg, tobacco_dataset_cls=object)


def test_legacy_builder_preserves_official_train_and_test_txt_calls():
    calls = []

    class RecordingLegacyDataset:
        def __init__(self, cfg, data_path, training=False):
            calls.append((cfg, data_path, training))

    cfg = SimpleNamespace(
        data="voc2007", train_path="data/voc2007/train.txt", test_path="data/voc2007/test.txt"
    )

    train_dataset, val_dataset = build_training_datasets(
        cfg, legacy_dataset_cls=RecordingLegacyDataset
    )

    assert isinstance(train_dataset, RecordingLegacyDataset)
    assert isinstance(val_dataset, RecordingLegacyDataset)
    assert calls == [
        (cfg, "data/voc2007/train.txt", True),
        (cfg, "data/voc2007/test.txt", False),
    ]


def test_cli_parses_tobacco_manifest_and_data_root():
    parser = get_args(parsing=False)

    args = parser.parse_args(
        [
            "--data",
            "tobacco",
            "--manifest-path",
            "manifests/image_manifest_updated.csv",
            "--data-root",
            "dataset",
            "--img-size",
            "384",
            "--batch-size",
            "16",
        ]
    )

    assert args.data == "tobacco"
    assert args.manifest_path == "manifests/image_manifest_updated.csv"
    assert args.data_root == "dataset"
    assert args.img_size == 384
    assert args.batch_size == 16


@pytest.mark.parametrize(
    "missing, expected",
    [("manifest", "manifest"), ("data_root", "data root")],
)
def test_prepare_env_requires_tobacco_manifest_and_data_root(
    tmp_path, monkeypatch, missing, expected
):
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("file_name,split\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    if missing == "manifest":
        manifest_path = tmp_path / "missing.csv"
    else:
        data_root = tmp_path / "missing-data"
    args = get_args(parsing=False).parse_args(
        [
            "--data",
            "tobacco",
            "--manifest-path",
            str(manifest_path),
            "--data-root",
            str(data_root),
        ]
    )

    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        prepare_env(args, ["train.py"])


def test_tobacco_config_is_safe_and_fully_recoverable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_path, data_root = _write_manifest_fixture(tmp_path)
    args = get_args(parsing=False).parse_args(
        [
            "--data",
            "tobacco",
            "--manifest-path",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--arch",
            "resnet101",
            "--img-size",
            "384",
            "--batch-size",
            "8",
            "--seed",
            "17",
        ]
    )

    cfg = prepare_env(args, ["train.py", "--data", "tobacco"])
    restored = load_config(Path(cfg.exp_dir) / "config.yaml")
    expected_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert cfg.num_classes == restored.num_classes == 18
    assert restored.dataset == "tobacco"
    assert restored.task == "multilabel_classification"
    assert restored.excluded_source_class_id == 13
    assert restored.selection_split == "val"
    assert restored.selection_metric == "mAP"
    assert restored.selection_variant == "ema"
    assert restored.metric_implementation == "VOC2012"
    assert restored.manifest_sha256 == expected_sha
    assert restored.manifest_path == str(manifest_path)
    assert restored.data_root == str(data_root)
    assert restored.arch == "resnet101"
    assert restored.img_size == 384
    assert restored.batch_size == 8
    assert restored.seed == 17
    assert restored.ignore_path is None
    assert not hasattr(restored, "train_path")
    assert not hasattr(restored, "test_path")
    assert not hasattr(restored, "label_path")
    assert restored.label_mapping == tobacco_label_mapping()


def test_legacy_config_paths_and_round_trip_remain_compatible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "voc2007"
    data_dir.mkdir(parents=True)
    (data_dir / "train.txt").write_text("train.jpg\ta\n", encoding="utf-8")
    (data_dir / "test.txt").write_text("val.jpg\ta\n", encoding="utf-8")
    (data_dir / "label.txt").write_text("a\nb\n", encoding="utf-8")
    args = get_args(parsing=False).parse_args(["--data", "voc2007"])

    cfg = prepare_env(args, ["train.py", "--data", "voc2007"])
    restored = load_config(Path(cfg.exp_dir) / "config.yaml")

    assert restored.data == "voc2007"
    assert restored.train_path == str(Path("data") / "voc2007" / "train.txt")
    assert restored.test_path == str(Path("data") / "voc2007" / "test.txt")
    assert restored.label_path == str(Path("data") / "voc2007" / "label.txt")
    assert restored.ignore_path == str(Path("data") / "voc2007" / "ignore.npy")
    assert restored.num_classes == 2
    assert restored.img_size == 448


def test_voc_map_accepts_none_ignore_path():
    metric = VOCmAP(2, year="2012", ignore_path=None)
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    targets = np.array([[1, 0], [0, 1]])

    aps, mean_ap = metric.compute(scores=scores, targets=targets)

    assert aps.tolist() == pytest.approx([1.0, 1.0])
    assert mean_ap == pytest.approx(1.0)


def test_voc_map_nonempty_ignore_mask_behavior_is_unchanged(tmp_path):
    ignore_path = tmp_path / "ignore.npy"
    np.save(ignore_path, np.array([[0], [1], [0]], dtype=np.int64))
    scores = np.array([[0.9], [0.8], [0.1]])
    targets = np.array([[1], [1], [0]])
    with_ignore = VOCmAP(1, year="2012", ignore_path=str(ignore_path))
    without_ignored_row = VOCmAP(1, year="2012", ignore_path=None)

    actual = with_ignore.compute(scores=scores, targets=targets)
    expected = without_ignored_row.compute(
        scores=scores[[0, 2]], targets=targets[[0, 2]]
    )

    assert actual[0].tolist() == pytest.approx(expected[0].tolist())
    assert actual[1] == pytest.approx(expected[1])


class TinyEma(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Linear(3, 2)


def _checkpoint_cfg(tmp_path, manifest_sha="a" * 64):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        model="mlic",
        arch="resnet101",
        num_classes=18,
        img_size=384,
        seed=17,
        manifest_sha256=manifest_sha,
        label_mapping=tobacco_label_mapping(),
        excluded_source_class_id=13,
        selection_split="val",
        selection_metric="mAP",
        metric_implementation="VOC2012",
        selection_variant="ema",
        ckpt_ema_best_path=str(checkpoint_dir / "best_ema_model.pth"),
        ckpt_best_info_path=str(checkpoint_dir / "best_checkpoint_info.json"),
    )


def _state_dict_equal(left, right):
    return list(left) == list(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), np.nan])
def test_nonfinite_validation_map_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        validate_finite_map("ema val mAP", value)


def test_strict_improvement_saves_first_zero_and_rejects_ties_or_lower_values():
    assert is_strict_improvement(0.0, None)
    assert is_strict_improvement(np.float64(0.51), 0.5)
    assert not is_strict_improvement(0.5, 0.5)
    assert not is_strict_improvement(0.49, 0.5)


def test_ema_checkpoint_and_metadata_form_a_reloadable_verified_pair(tmp_path):
    torch.manual_seed(201)
    ema_model = TinyEma()
    cfg = _checkpoint_cfg(tmp_path, manifest_sha="b" * 64)
    expected_state = copy.deepcopy(ema_model.state_dict())

    metadata = save_best_ema_checkpoint(ema_model, cfg, epoch=10, val_map=0.625)
    verified = verify_best_checkpoint_integrity(
        cfg.ckpt_ema_best_path, cfg.ckpt_best_info_path
    )
    restored_state = torch.load(cfg.ckpt_ema_best_path, map_location="cpu")

    assert _state_dict_equal(restored_state, expected_state)
    assert metadata == verified
    assert metadata["architecture"] == "mlic"
    assert metadata["backbone"] == "resnet101"
    assert metadata["num_classes"] == 18
    assert metadata["image_size"] == 384
    assert metadata["seed"] == 17
    assert metadata["manifest_sha256"] == "b" * 64
    assert metadata["label_mapping"] == tobacco_label_mapping()
    assert metadata["excluded_source_class_id"] == 13
    assert metadata["selection_split"] == "val"
    assert metadata["selection_metric"] == "mAP"
    assert metadata["metric_implementation"] == "VOC2012"
    assert metadata["model_variant"] == "EMA"
    assert metadata["best_epoch"] == 10
    assert metadata["best_val_mAP"] == pytest.approx(0.625)
    assert metadata["checkpoint_path"] == cfg.ckpt_ema_best_path
    assert metadata["checkpoint_sha256"] == sha256_file(cfg.ckpt_ema_best_path)


@pytest.mark.parametrize("failure_stage", ["checkpoint_write", "metadata_write", "first_replace", "second_replace"])
def test_failed_candidate_save_restores_existing_checkpoint_pair(tmp_path, failure_stage):
    torch.manual_seed(202)
    ema_model = TinyEma()
    cfg = _checkpoint_cfg(tmp_path)
    original_metadata = save_best_ema_checkpoint(
        ema_model, cfg, epoch=10, val_map=0.4
    )
    original_checkpoint = Path(cfg.ckpt_ema_best_path).read_bytes()
    original_json = Path(cfg.ckpt_best_info_path).read_bytes()
    with torch.no_grad():
        ema_model.module.weight.add_(5)

    def checkpoint_writer(state, path):
        if failure_stage == "checkpoint_write":
            raise OSError("injected checkpoint write failure")
        with open(path, "wb") as handle:
            torch.save(state, handle)

    def metadata_writer(payload, path):
        if failure_stage == "metadata_write":
            raise OSError("injected metadata write failure")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    replace_calls = []

    def replace(source, destination):
        import os

        replace_calls.append((source, destination))
        call_number = len(replace_calls)
        if failure_stage == "first_replace" and call_number == 1:
            raise OSError("injected first replace failure")
        if failure_stage == "second_replace" and call_number == 2:
            raise OSError("injected second replace failure")
        os.replace(source, destination)

    with pytest.raises(OSError, match="injected"):
        save_best_ema_checkpoint(
            ema_model,
            cfg,
            epoch=15,
            val_map=0.6,
            checkpoint_writer=checkpoint_writer,
            metadata_writer=metadata_writer,
            replace_fn=replace,
        )

    assert Path(cfg.ckpt_ema_best_path).read_bytes() == original_checkpoint
    assert Path(cfg.ckpt_best_info_path).read_bytes() == original_json
    assert verify_best_checkpoint_integrity(
        cfg.ckpt_ema_best_path, cfg.ckpt_best_info_path
    ) == original_metadata


def test_rollback_failure_is_reported_and_integrity_check_rejects_partial_pair(tmp_path):
    torch.manual_seed(203)
    ema_model = TinyEma()
    cfg = _checkpoint_cfg(tmp_path)
    save_best_ema_checkpoint(ema_model, cfg, epoch=10, val_map=0.4)
    with torch.no_grad():
        ema_model.module.bias.add_(7)
    replace_count = 0

    def replace_with_failed_rollback(source, destination):
        import os

        nonlocal replace_count
        replace_count += 1
        if replace_count in (2, 3):
            raise OSError("injected replace or rollback failure")
        os.replace(source, destination)

    with pytest.raises(CheckpointTransactionError, match="rollback"):
        save_best_ema_checkpoint(
            ema_model,
            cfg,
            epoch=15,
            val_map=0.7,
            replace_fn=replace_with_failed_rollback,
        )

    with pytest.raises(ValueError, match="SHA-256"):
        verify_best_checkpoint_integrity(
            cfg.ckpt_ema_best_path, cfg.ckpt_best_info_path
        )


def test_integrity_check_rejects_checkpoint_corruption(tmp_path):
    ema_model = TinyEma()
    cfg = _checkpoint_cfg(tmp_path)
    save_best_ema_checkpoint(ema_model, cfg, epoch=1, val_map=0.0)

    with open(cfg.ckpt_ema_best_path, "ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_best_checkpoint_integrity(
            cfg.ckpt_ema_best_path, cfg.ckpt_best_info_path
        )


def _load_train_module(monkeypatch):
    inplace_abn_stub = types.ModuleType("inplace_abn")
    inplace_abn_stub.InPlaceABN = nn.BatchNorm2d
    inplace_abn_stub.ABN = nn.BatchNorm2d
    monkeypatch.setitem(sys.modules, "inplace_abn", inplace_abn_stub)
    tensorboard_stub = types.ModuleType("torch.utils.tensorboard")

    class ImportOnlySummaryWriter:
        pass

    tensorboard_stub.SummaryWriter = ImportOnlySummaryWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard_stub)
    sys.modules.pop("train", None)
    import train

    return train


def _trainer_cfg(tmp_path):
    cfg = _checkpoint_cfg(tmp_path)
    cfg.data = "tobacco"
    cfg.batch_size = 4
    cfg.pretrained_model = None
    cfg.ema_decay = 0.9997
    cfg.ignore_path = None
    cfg.model = "mlic"
    cfg.exp_dir = str(tmp_path / "experiment")
    cfg.log_path = str(tmp_path / "experiment" / "train.log")
    cfg.max_epochs = 4
    cfg.estop = False
    cfg.lamda = 1.0
    cfg.amp = False
    return cfg


def test_real_trainer_constructor_uses_protocol_builder_and_has_no_test_loader(
    tmp_path, monkeypatch
):
    train_module = _load_train_module(monkeypatch)
    cfg = _trainer_cfg(tmp_path)
    builder_calls = []
    loader_datasets = []

    class FakeDataset:
        def __init__(self, split):
            if split == "test":
                raise AssertionError("test dataset constructed")
            self.split = split
            self.transform = "{}-transform".format(split)

        def __len__(self):
            return 8

    def dataset_builder(received_cfg):
        builder_calls.append(received_cfg)
        return FakeDataset("train"), FakeDataset("val")

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            loader_datasets.append(dataset.split)
            self.dataset = dataset

        def __len__(self):
            return 2

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def cuda(self, rank):
            return self

    class NullWriter:
        def __init__(self, *args, **kwargs):
            pass

    null_logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "build_training_datasets", dataset_builder)
    monkeypatch.setattr(train_module, "DataLoader", FakeLoader)
    monkeypatch.setattr(train_module.torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(train_module, "create_model", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(train_module, "get_optimizer", lambda *args: object())
    monkeypatch.setattr(train_module, "get_lr_scheduler", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_module, "get_loss_fn", lambda cfg: object())
    monkeypatch.setattr(train_module, "SummaryWriter", NullWriter)
    monkeypatch.setattr(train_module, "get_logger", lambda *args: null_logger)

    trainer = train_module.Trainer(cfg, world_size=1, rank=0)

    assert builder_calls == [cfg]
    assert loader_datasets == ["train", "val"]
    assert trainer.train_loader.dataset.split == "train"
    assert trainer.val_loader.dataset.split == "val"
    assert not hasattr(trainer, "test_loader")
    assert trainer.best_mAP is None
    assert trainer.best_epoch is None


def test_trainer_run_selects_only_strict_ema_val_improvements(tmp_path, monkeypatch):
    train_module = _load_train_module(monkeypatch)
    trainer = object.__new__(train_module.Trainer)
    trainer.cfg = _trainer_cfg(tmp_path)
    trainer.ema_model = object()
    trainer.distributed = False
    trainer.notdist_or_rank0 = True
    trainer.best_mAP = None
    trainer.best_epoch = None
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    trainer.train = lambda epoch: None
    validation_results = iter([0.0, 0.0, 0.5, 0.4])
    trainer.validation = lambda epoch: next(validation_results)
    saves = []

    def record_save(ema_model, cfg, epoch, val_map):
        saves.append((ema_model, epoch, val_map))
        return {"best_epoch": epoch, "best_val_mAP": val_map}

    monkeypatch.setattr(train_module, "save_best_ema_checkpoint", record_save)

    trainer.run()

    assert [(epoch, score) for _, epoch, score in saves] == [(0, 0.0), (2, 0.5)]
    assert all(model is trainer.ema_model for model, _, _ in saves)
    assert trainer.best_mAP == 0.5
    assert trainer.best_epoch == 2


def test_trainer_run_rejects_nonfinite_ema_validation_map(tmp_path, monkeypatch):
    train_module = _load_train_module(monkeypatch)
    trainer = object.__new__(train_module.Trainer)
    trainer.cfg = _trainer_cfg(tmp_path)
    trainer.ema_model = object()
    trainer.distributed = False
    trainer.notdist_or_rank0 = True
    trainer.best_mAP = None
    trainer.best_epoch = None
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    trainer.train = lambda epoch: None
    trainer.validation = lambda epoch: float("nan")

    with pytest.raises(ValueError, match="finite"):
        trainer.run()


def test_legacy_trainer_run_preserves_official_ema_state_dict_save(tmp_path, monkeypatch):
    train_module = _load_train_module(monkeypatch)
    trainer = object.__new__(train_module.Trainer)
    trainer.cfg = _trainer_cfg(tmp_path)
    trainer.cfg.data = "voc2007"
    trainer.cfg.max_epochs = 1
    trainer.ema_model = TinyEma()
    trainer.distributed = False
    trainer.notdist_or_rank0 = True
    trainer.best_mAP = None
    trainer.best_epoch = None
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    trainer.train = lambda epoch: None
    trainer.validation = lambda epoch: 0.0
    saved = []
    monkeypatch.setattr(
        train_module,
        "save_best_ema_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tobacco metadata saver used for legacy dataset")
        ),
    )
    monkeypatch.setattr(
        train_module.torch,
        "save",
        lambda state, path: saved.append((state, path)),
    )

    trainer.run()

    assert len(saved) == 1
    assert saved[0][1] == trainer.cfg.ckpt_ema_best_path
    assert _state_dict_equal(saved[0][0], trainer.ema_model.state_dict())
    assert trainer.best_mAP == 0.0
    assert trainer.best_epoch == 0


def test_validation_rejects_nonfinite_current_or_ema_map(tmp_path, monkeypatch):
    train_module = _load_train_module(monkeypatch)

    class EvalOnly:
        def eval(self):
            return self

    class Metric:
        def __init__(self, value):
            self.value = value

        def reset(self):
            pass

        def compute(self):
            return np.zeros(18), self.value

    trainer = object.__new__(train_module.Trainer)
    trainer.model = EvalOnly()
    trainer.ema_model = EvalOnly()
    trainer.voc_mAP = Metric(float("inf"))
    trainer.voc_ema_mAP = Metric(0.5)
    trainer.val_loader = []
    trainer.distributed = False
    trainer.notdist_or_rank0 = True
    trainer.global_step = 0
    trainer.best_mAP = None
    trainer.writer = SimpleNamespace(add_scalar=lambda *args: None)
    trainer.logger = SimpleNamespace(info=lambda *args: None)

    with pytest.raises(ValueError, match="finite"):
        trainer.validation(epoch=0)


@pytest.mark.parametrize("loader_length, expected", [(0, 1), (1, 1), (5, 1), (6, 1), (18, 3)])
def test_training_log_interval_never_allows_modulo_zero(loader_length, expected):
    assert training_log_interval(loader_length) == expected


def test_failed_experiment_cleanup_preserves_tobacco_evidence():
    assert not should_clear_failed_experiment(SimpleNamespace(data="tobacco"))
    assert should_clear_failed_experiment(SimpleNamespace(data="voc2007"))
