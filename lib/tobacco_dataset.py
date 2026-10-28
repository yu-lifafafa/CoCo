import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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
EXCLUDED_SOURCE_ID = 13
MODEL_LABEL_COLUMNS = SOURCE_LABEL_COLUMNS[:13] + SOURCE_LABEL_COLUMNS[14:]
VALID_SPLITS = frozenset(("train", "val", "test"))
REQUIRED_COLUMNS = frozenset(
    ("file_name", "split", "is_healthy", "positive_label_count")
    + SOURCE_LABEL_COLUMNS
)


@dataclass(frozen=True)
class TobaccoRecord:
    image_path: Path
    file_name: str
    target: np.ndarray
    source_positive_count: int
    task_positive_count: int
    is_healthy: bool


def _relative_path_parts(file_name, line_number):
    if not file_name or file_name != file_name.strip():
        raise ValueError(
            "line {} file_name must be a non-empty relative path: {!r}".format(
                line_number, file_name
            )
        )
    normalized = file_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(file_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
        or not posix_path.parts
        or posix_path == PurePosixPath(".")
    ):
        raise ValueError(
            "line {} file_name must be a safe relative path: {!r}".format(
                line_number, file_name
            )
        )
    return posix_path.parts


def _parse_binary(row, column, line_number):
    value = row[column]
    if value not in ("0", "1"):
        raise ValueError(
            "line {} column {} must be binary 0 or 1, got {!r}".format(
                line_number, column, value
            )
        )
    return int(value)


class TobaccoManifestDataset(Dataset):
    def __init__(self, manifest_path, data_root, split, transform=None):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        if split not in VALID_SPLITS:
            raise ValueError(
                "split must be one of {}, got {!r}".format(
                    sorted(VALID_SPLITS), split
                )
            )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "manifest file does not exist: {}".format(self.manifest_path)
            )

        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
            if missing_columns:
                raise ValueError(
                    "manifest is missing required columns: {}".format(
                        ", ".join(missing_columns)
                    )
                )
            rows = list(reader)

        self.records = []
        seen_split_paths = set()
        seen_file_splits = {}
        seen_source_identities = {}
        selected_root = (self.data_root / split).resolve()

        for line_number, row in enumerate(rows, start=2):
            row_split = row["split"]
            if row_split not in VALID_SPLITS:
                raise ValueError(
                    "line {} has invalid split {!r}".format(line_number, row_split)
                )

            path_parts = _relative_path_parts(row["file_name"], line_number)
            normalized_file_name = "/".join(path_parts)
            split_path_key = (row_split, normalized_file_name)
            if split_path_key in seen_split_paths:
                raise ValueError(
                    "line {} is a duplicate record for split/file_name: {}/{}".format(
                        line_number, row_split, row["file_name"]
                    )
                )
            seen_split_paths.add(split_path_key)

            previous_split = seen_file_splits.get(normalized_file_name)
            if previous_split is not None and previous_split != row_split:
                raise ValueError(
                    "line {} repeats file_name across splits {} and {}: {}".format(
                        line_number, previous_split, row_split, row["file_name"]
                    )
                )
            seen_file_splits[normalized_file_name] = row_split

            identity_columns = ("source_file", "source_image_id", "original_file_name")
            if all(column in row and row[column] for column in identity_columns):
                source_identity = tuple(row[column] for column in identity_columns)
                previous_identity_split = seen_source_identities.get(source_identity)
                if (
                    previous_identity_split is not None
                    and previous_identity_split != row_split
                ):
                    raise ValueError(
                        "line {} repeats source identity across splits {} and {}: {}".format(
                            line_number,
                            previous_identity_split,
                            row_split,
                            source_identity,
                        )
                    )
                seen_source_identities[source_identity] = row_split

            source_values = [
                _parse_binary(row, column, line_number)
                for column in SOURCE_LABEL_COLUMNS
            ]
            is_healthy = bool(_parse_binary(row, "is_healthy", line_number))
            try:
                declared_positive_count = int(row["positive_label_count"])
            except (TypeError, ValueError):
                raise ValueError(
                    "line {} positive_label_count must be an integer, got {!r}".format(
                        line_number, row["positive_label_count"]
                    )
                )
            source_positive_count = sum(source_values)
            if declared_positive_count != source_positive_count:
                raise ValueError(
                    "line {} positive_label_count {} does not match 19 source labels {}".format(
                        line_number, declared_positive_count, source_positive_count
                    )
                )
            if is_healthy != (source_positive_count == 0):
                raise ValueError(
                    "line {} is_healthy {} disagrees with 19 source labels".format(
                        line_number, int(is_healthy)
                    )
                )

            if row_split != split:
                continue

            source_target = np.asarray(source_values, dtype=np.float32)
            target = np.delete(source_target, EXCLUDED_SOURCE_ID).astype(
                np.float32, copy=False
            )
            image_path = selected_root.joinpath(*path_parts).resolve()
            if image_path != selected_root and selected_root not in image_path.parents:
                raise ValueError(
                    "line {} file_name must remain under the split directory: {!r}".format(
                        line_number, row["file_name"]
                    )
                )
            if not image_path.is_file():
                raise FileNotFoundError(
                    "line {} image file does not exist: {}".format(
                        line_number, image_path
                    )
                )
            self.records.append(
                TobaccoRecord(
                    image_path=image_path,
                    file_name=row["file_name"],
                    target=target,
                    source_positive_count=source_positive_count,
                    task_positive_count=int(target.sum()),
                    is_healthy=is_healthy,
                )
            )

        self.samples = self.records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.image_path) as source_image:
            image = source_image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "img": image,
            "target": record.target.copy(),
            "img_path": str(record.image_path),
        }
