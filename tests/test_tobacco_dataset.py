import csv
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.tobacco_dataset import TobaccoManifestDataset


SOURCE_LABEL_COLUMNS = (
    "y00_c02_weather_fleck",
    "y01_c03_potato_virus_y",
    "y02_c04_tobacco_mosaic_virus",
    "y03_c05_cucumber_mosaic_virus",
    "y04_c06_alternaria_leaf_spot",
    "y05_c07_black_shank",
    "y06_c08_bacterial_wilt",
    "y07_c09_hollow_stalk",
    "y08_c10_black_root_rot",
    "y09_c11_wildfire",
    "y10_c12_potassium_deficiency",
    "y11_c13_magnesium_deficiency",
    "y12_c14_powdery_mildew",
    "y13_c15_root_knot_nematodes",
    "y14_c16_anthracnose",
    "y15_c17_frogeye_leaf_spot",
    "y16_c18_angular_leaf_spot",
    "y17_c19_sunburn",
    "y18_c20_herbicide_phytotoxicity",
)


def _row(
    file_name="images/task_01/sample.jpg",
    split="train",
    positives=(),
    image_id="image-1",
    source_image_id="source-image-1",
    original_file_name="sample.jpg",
):
    row = {
        "image_id": image_id,
        "file_name": file_name,
        "split": split,
        "split_seed": "20260803",
        "group_id": "",
        "task_folder": "task_01",
        "source_file": "instances_default.json",
        "source_image_id": source_image_id,
        "original_file_name": original_file_name,
        "width": "12",
        "height": "10",
        "is_healthy": "0" if positives else "1",
        "positive_label_count": str(len(positives)),
    }
    row.update({column: "0" for column in SOURCE_LABEL_COLUMNS})
    for source_id in positives:
        row[SOURCE_LABEL_COLUMNS[source_id]] = "1"
    return row


def _write_fixture(tmp_path: Path, rows, create_images=True, image_mode="RGB"):
    manifest_path = tmp_path / "image_manifest_updated.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    data_root = tmp_path / "data"
    if create_images:
        for row in rows:
            image_path = data_root / row["split"] / Path(
                *row["file_name"].replace("\\", "/").split("/")
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            color = 12 if image_mode == "L" else (12, 34, 56)
            Image.new(image_mode, (12, 10), color=color).save(image_path)
    return manifest_path, data_root


def _load_official_dataset_module(monkeypatch):
    class RandAugment:
        def __call__(self, image):
            return image

    fake_module = SimpleNamespace(RandAugment=RandAugment)
    monkeypatch.setitem(sys.modules, "randaugment", fake_module)
    sys.modules.pop("lib.dataset", None)
    return importlib.import_module("lib.dataset")


def test_excludes_source_13_and_maps_source_12_and_14_contiguously(tmp_path):
    manifest_path, data_root = _write_fixture(
        tmp_path, [_row(positives=(12, 13, 14))]
    )

    dataset = TobaccoManifestDataset(manifest_path, data_root, "train")
    target = dataset[0]["target"]

    assert target.dtype == np.float32
    assert target.shape == (18,)
    assert target[12] == 1.0
    assert target[13] == 1.0
    assert target.sum() == 2.0


def test_single_and_multiple_disease_targets_are_multi_hot(tmp_path):
    rows = [
        _row(positives=(2,), image_id="single", source_image_id="single"),
        _row(
            file_name="images/task_02/multi.jpg",
            positives=(0, 5, 18),
            image_id="multi",
            source_image_id="multi",
            original_file_name="multi.jpg",
        ),
    ]
    manifest_path, data_root = _write_fixture(tmp_path, rows)

    dataset = TobaccoManifestDataset(manifest_path, data_root, "train")

    np.testing.assert_array_equal(
        dataset[0]["target"],
        np.array([0, 0, 1] + [0] * 15, dtype=np.float32),
    )
    assert np.flatnonzero(dataset[1]["target"]).tolist() == [0, 5, 17]


def test_root_knot_cooccurrence_keeps_other_diseases_and_record(tmp_path):
    manifest_path, data_root = _write_fixture(
        tmp_path, [_row(positives=(1, 13, 16))]
    )

    dataset = TobaccoManifestDataset(manifest_path, data_root, "train")

    assert len(dataset) == 1
    assert np.flatnonzero(dataset[0]["target"]).tolist() == [1, 15]
    assert dataset.records[0].source_positive_count == 3
    assert dataset.records[0].task_positive_count == 2


def test_root_knot_only_uses_raw_positive_count_but_zero_task_target(tmp_path):
    manifest_path, data_root = _write_fixture(tmp_path, [_row(positives=(13,))])

    dataset = TobaccoManifestDataset(manifest_path, data_root, "train")

    assert dataset.records[0].source_positive_count == 1
    assert dataset.records[0].task_positive_count == 0
    assert dataset[0]["target"].sum() == 0.0


def test_healthy_sample_is_preserved_with_zero_target(tmp_path):
    manifest_path, data_root = _write_fixture(
        tmp_path, [_row(file_name="images/healthy_x.jpg")]
    )

    dataset = TobaccoManifestDataset(manifest_path, data_root, "train")
    item = dataset[0]

    assert set(item) == {"img", "target", "img_path"}
    assert item["target"].dtype == np.float32
    assert item["target"].shape == (18,)
    assert item["target"].sum() == 0.0
    assert Path(item["img_path"]) == data_root / "train" / "images" / "healthy_x.jpg"


def test_task_subdirectory_and_windows_separators_resolve_under_split(tmp_path):
    row = _row(file_name=r"images\task_09\sample.jpg")
    manifest_path, data_root = _write_fixture(tmp_path, [row])

    item = TobaccoManifestDataset(manifest_path, data_root, "train")[0]

    assert Path(item["img_path"]) == (
        data_root / "train" / "images" / "task_09" / "sample.jpg"
    )


def test_dataset_filters_only_requested_manifest_split(tmp_path):
    rows = [
        _row(split="train", image_id="train", source_image_id="train"),
        _row(
            split="val",
            image_id="val",
            source_image_id="val",
            file_name="images/task_01/val.jpg",
            original_file_name="val.jpg",
        ),
        _row(
            split="test",
            image_id="test",
            source_image_id="test",
            file_name="images/task_01/test.jpg",
            original_file_name="test.jpg",
        ),
    ]
    manifest_path, data_root = _write_fixture(tmp_path, rows)

    train = TobaccoManifestDataset(manifest_path, data_root, "train")
    val = TobaccoManifestDataset(manifest_path, data_root, "val")
    test = TobaccoManifestDataset(manifest_path, data_root, "test")

    assert len(train) == len(val) == len(test) == 1
    assert "train" in train[0]["img_path"]
    assert "val" in val[0]["img_path"]
    assert "test" in test[0]["img_path"]


@pytest.mark.parametrize("bad_value", ["", "NaN", "2", "-1", "true"])
def test_rejects_missing_nan_and_non_binary_label_values(tmp_path, bad_value):
    row = _row()
    row[SOURCE_LABEL_COLUMNS[4]] = bad_value
    manifest_path, data_root = _write_fixture(tmp_path, [row])

    with pytest.raises(ValueError, match="binary"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


def test_rejects_missing_required_label_column(tmp_path):
    row = _row()
    row.pop(SOURCE_LABEL_COLUMNS[7])
    manifest_path, data_root = _write_fixture(tmp_path, [row])

    with pytest.raises(ValueError, match="missing required columns"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


def test_rejects_positive_label_count_that_disagrees_with_source_labels(tmp_path):
    row = _row(positives=(3, 13))
    row["positive_label_count"] = "1"
    manifest_path, data_root = _write_fixture(tmp_path, [row])

    with pytest.raises(ValueError, match="positive_label_count"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


def test_rejects_healthy_flag_that_disagrees_with_source_labels(tmp_path):
    row = _row(positives=(3,))
    row["is_healthy"] = "1"
    manifest_path, data_root = _write_fixture(tmp_path, [row])

    with pytest.raises(ValueError, match="is_healthy"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


def test_rejects_duplicate_split_path_records(tmp_path):
    rows = [
        _row(image_id="one", source_image_id="one"),
        _row(image_id="two", source_image_id="two"),
    ]
    manifest_path, data_root = _write_fixture(tmp_path, rows)

    with pytest.raises(ValueError, match="duplicate record"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


def test_rejects_same_source_identity_across_splits(tmp_path):
    rows = [
        _row(split="train", image_id="one"),
        _row(
            split="val",
            image_id="two",
            file_name="images/task_01/other.jpg",
        ),
    ]
    manifest_path, data_root = _write_fixture(tmp_path, rows)

    with pytest.raises(ValueError, match="source identity"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


@pytest.mark.parametrize(
    "bad_path",
    ["../outside.jpg", "images/../../outside.jpg", "/absolute.jpg", r"C:\absolute.jpg"],
)
def test_rejects_path_traversal_and_absolute_paths(tmp_path, bad_path):
    manifest_path, data_root = _write_fixture(
        tmp_path, [_row(file_name=bad_path)], create_images=False
    )

    with pytest.raises(ValueError, match="relative path"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


def test_rejects_missing_image_file_explicitly(tmp_path):
    manifest_path, data_root = _write_fixture(
        tmp_path, [_row()], create_images=False
    )

    with pytest.raises(FileNotFoundError, match="image file does not exist"):
        TobaccoManifestDataset(manifest_path, data_root, "train")


def test_reads_image_as_rgb(tmp_path):
    manifest_path, data_root = _write_fixture(tmp_path, [_row()], image_mode="L")

    image = TobaccoManifestDataset(manifest_path, data_root, "train")[0]["img"]

    assert image.mode == "RGB"


def test_dataloader_collates_images_and_float32_targets(tmp_path):
    rows = [
        _row(image_id="one", source_image_id="one", positives=(0,)),
        _row(
            file_name="images/task_02/two.jpg",
            image_id="two",
            source_image_id="two",
            original_file_name="two.jpg",
            positives=(18,),
        ),
    ]
    manifest_path, data_root = _write_fixture(tmp_path, rows)
    transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
    dataset = TobaccoManifestDataset(
        manifest_path, data_root, "train", transform=transform
    )

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert batch["img"].shape == (2, 3, 8, 8)
    assert batch["target"].shape == (2, 18)
    assert batch["target"].dtype == torch.float32


def test_shared_transform_preserves_official_order_and_parameters(monkeypatch):
    dataset_module = _load_official_dataset_module(monkeypatch)
    cfg = SimpleNamespace(img_size=32)

    training_transform = dataset_module.build_coco_transform(cfg, training=True)
    eval_transform = dataset_module.build_coco_transform(cfg, training=False)

    assert [type(t).__name__ for t in training_transform.transforms] == [
        "Resize",
        "RandomHorizontalFlip",
        "CutoutPIL",
        "RandAugment",
        "ToTensor",
        "Normalize",
    ]
    assert [type(t).__name__ for t in eval_transform.transforms] == [
        "Resize",
        "ToTensor",
        "Normalize",
    ]
    assert training_transform.transforms[0].size == (32, 32)
    assert training_transform.transforms[-1].mean == [0.485, 0.456, 0.406]
    assert training_transform.transforms[-1].std == [0.229, 0.224, 0.225]


def test_official_mldataset_interface_and_transform_remain_compatible(
    tmp_path, monkeypatch
):
    dataset_module = _load_official_dataset_module(monkeypatch)
    image_path = tmp_path / "official.jpg"
    Image.new("RGB", (12, 10), color=(12, 34, 56)).save(image_path)
    label_path = tmp_path / "label.txt"
    label_path.write_text("disease_a\ndisease_b\n", encoding="utf-8")
    data_path = tmp_path / "train.txt"
    data_path.write_text(f"{image_path}\tdisease_b\n", encoding="utf-8")
    cfg = SimpleNamespace(label_path=str(label_path), img_size=16)

    dataset = dataset_module.MLDataset(cfg, str(data_path), training=False)
    item = dataset[0]

    assert set(item) == {"img", "target", "img_path"}
    assert item["img"].shape == (3, 16, 16)
    np.testing.assert_array_equal(item["target"], np.array([0, 1], dtype=np.float32))
    assert item["img_path"] == str(image_path)
