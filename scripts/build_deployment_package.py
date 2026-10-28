"""Build a deterministic deployment snapshot from the current working tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_DATA_MANIFEST_SHA256 = (
    "ce51205b537c3d1eee28b3d8112ce1edb6454c6f16cd3b441387bff2be2c56b5"
)
BASELINE_COMMIT = "342c4e35b19fc773ae71565dbe17be70d56bfcdc"
DEFAULT_REQUIRED_UNTRACKED = (
    "docs/superpowers/plans/2026-10-09-coco-tobacco-evaluation.md",
    "docs/superpowers/plans/2026-10-09-coco-tobacco-training-protocol.md",
    "docs/superpowers/plans/2026-10-10-coco-tobacco-gpu-smoke-deployment.md",
    "docs/superpowers/specs/2026-10-09-coco-tobacco-evaluation-design.md",
    "docs/superpowers/specs/2026-10-09-coco-tobacco-training-protocol-design.md",
    "docs/superpowers/specs/2026-10-10-coco-tobacco-gpu-smoke-deployment-design.md",
    "lib/evaluation_protocol.py",
    "lib/tobacco_dataset.py",
    "lib/training_protocol.py",
    "tests/test_coco_contrastive.py",
    "tests/test_deployment_package.py",
    "tests/test_evaluation_protocol.py",
    "tests/test_smoke_tobacco_coco.py",
    "tests/test_tobacco_dataset.py",
    "tests/test_training_protocol.py",
    "scripts/build_deployment_package.py",
    "scripts/smoke_tobacco_coco.py",
    "deploy/environment.yml",
    "deploy/server_setup.md",
    "deploy/verify_deployment.py",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "git ls-files failed: " + result.stderr.decode("utf-8", errors="replace")
        )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _excluded_from_deployment(path: str) -> bool:
    pure = PurePosixPath(path)
    lowered = tuple(part.lower() for part in pure.parts)
    name = pure.name.lower()
    suffix = pure.suffix.lower()
    if not lowered:
        return True
    if lowered[0] in {"data", "experiments", "outputs", ".git", ".superpowers"}:
        return True
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in lowered):
        return True
    if name.startswith("image_manifest") and suffix == ".csv":
        return True
    if suffix in {".pth", ".pt", ".ckpt", ".jpg", ".jpeg", ".png", ".gif", ".key", ".pem"}:
        return True
    if name in {".env", ".env.local", "deployment_manifest.json", "coco_tobacco_deploy.zip"}:
        return True
    if lowered[0] == "deploy" and suffix == ".zip":
        return True
    return False


def collect_snapshot_files(repo_root, required_untracked=DEFAULT_REQUIRED_UNTRACKED):
    repo_root = Path(repo_root).resolve()
    tracked = set(_git_tracked_files(repo_root))
    required = set(required_untracked)
    for relative in sorted(required):
        path = repo_root / Path(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise FileNotFoundError(f"required deployment file is missing: {relative}")
        if _excluded_from_deployment(relative):
            raise ValueError(f"required deployment file is excluded by policy: {relative}")
    members = sorted(
        relative
        for relative in tracked | required
        if not _excluded_from_deployment(relative)
        and (repo_root / Path(*PurePosixPath(relative).parts)).is_file()
    )
    return members


def _zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _replace_pair_with_rollback(temp_archive, archive_path, temp_manifest, manifest_path):
    archive_backup = archive_path.with_name(archive_path.name + ".backup")
    manifest_backup = manifest_path.with_name(manifest_path.name + ".backup")
    for backup in (archive_backup, manifest_backup):
        if backup.exists():
            backup.unlink()
    if archive_path.exists():
        os.replace(archive_path, archive_backup)
    if manifest_path.exists():
        os.replace(manifest_path, manifest_backup)
    try:
        os.replace(temp_archive, archive_path)
        os.replace(temp_manifest, manifest_path)
    except Exception:
        if archive_path.exists():
            archive_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
        if archive_backup.exists():
            os.replace(archive_backup, archive_path)
        if manifest_backup.exists():
            os.replace(manifest_backup, manifest_path)
        raise
    else:
        if archive_backup.exists():
            archive_backup.unlink()
        if manifest_backup.exists():
            manifest_backup.unlink()


def build_package(
    repo_root,
    archive_path,
    manifest_path,
    baseline_commit=BASELINE_COMMIT,
    required_untracked=DEFAULT_REQUIRED_UNTRACKED,
    expected_data_manifest_sha256=EXPECTED_DATA_MANIFEST_SHA256,
):
    from deploy.verify_deployment import verify_archive

    repo_root = Path(repo_root).resolve()
    archive_path = Path(archive_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    members = collect_snapshot_files(repo_root, required_untracked)
    file_entries = []
    temp_archive = archive_path.with_name(archive_path.name + f".tmp-{os.getpid()}")
    temp_manifest = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temp_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for relative in members:
                data = (repo_root / Path(*PurePosixPath(relative).parts)).read_bytes()
                archive.writestr(_zip_info(relative), data, compresslevel=9)
                file_entries.append(
                    {"path": relative, "size": len(data), "sha256": _sha256_bytes(data)}
                )
        payload = {
            "format_version": 1,
            "package_kind": "coco-tobacco-task6a-source-snapshot",
            "baseline_commit": baseline_commit,
            "expected_data_manifest_sha256": expected_data_manifest_sha256,
            "archive_sha256": _sha256_file(temp_archive),
            "file_count": len(file_entries),
            "files": file_entries,
        }
        verify_archive(temp_archive, payload)
        temp_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if json.loads(temp_manifest.read_text("utf-8")) != payload:
            raise RuntimeError("deployment manifest reload verification failed")
        _replace_pair_with_rollback(
            temp_archive, archive_path, temp_manifest, manifest_path
        )
        return payload
    finally:
        for temporary in (temp_archive, temp_manifest):
            if temporary.exists():
                temporary.unlink()


def build_parser():
    parser = argparse.ArgumentParser(description="Build the CoCo Task 6A deployment ZIP.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    parser.add_argument(
        "--expected-data-manifest-sha256", default=EXPECTED_DATA_MANIFEST_SHA256
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = build_package(
        args.repo_root,
        args.archive,
        args.manifest,
        args.baseline_commit,
        expected_data_manifest_sha256=args.expected_data_manifest_sha256,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
