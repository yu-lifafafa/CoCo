"""Verify and safely extract a CoCo Task 6A deployment package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or ".." in pure.parts
        or any(part in {"", "."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return pure.as_posix()


def load_deployment_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version",
        "baseline_commit",
        "expected_data_manifest_sha256",
        "archive_sha256",
        "file_count",
        "files",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError("deployment manifest missing keys: " + ", ".join(sorted(missing)))
    return payload


def verify_archive(archive_path, manifest) -> bool:
    archive_path = Path(archive_path)
    if _sha256_file(archive_path) != manifest["archive_sha256"]:
        raise ValueError("archive SHA-256 does not match deployment manifest")
    expected = {entry["path"]: entry for entry in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("deployment manifest contains duplicate file paths")
    with zipfile.ZipFile(archive_path) as archive:
        normalized_infos = {}
        for info in archive.infolist():
            normalized = validate_member_name(info.filename)
            if normalized in normalized_infos:
                raise ValueError(f"duplicate normalized archive member: {normalized}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                raise ValueError(f"archive member is a symbolic link: {normalized}")
            if info.is_dir():
                raise ValueError(f"archive must contain files only: {normalized}")
            normalized_infos[normalized] = info
        if set(normalized_infos) != set(expected):
            raise ValueError("archive member file set does not match deployment manifest")
        for relative, entry in expected.items():
            data = archive.read(normalized_infos[relative])
            if len(data) != entry["size"]:
                raise ValueError(f"archive member size mismatch: {relative}")
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"archive member SHA-256 mismatch: {relative}")
    return True


def verify_extracted_tree(root, manifest) -> bool:
    root = Path(root)
    expected = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != set(expected):
        raise ValueError("extracted file set does not match deployment manifest")
    for relative, entry in expected.items():
        path = root / Path(*PurePosixPath(relative).parts)
        if path.is_symlink():
            raise ValueError(f"extracted file is a symbolic link: {relative}")
        if path.stat().st_size != entry["size"] or _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"extracted file SHA-256 mismatch: {relative}")
    return True


def extract_verified_archive(archive_path, destination, manifest) -> bool:
    verify_archive(archive_path, manifest)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"extraction destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = validate_member_name(info.filename)
            target = destination / Path(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    return verify_extracted_tree(destination, manifest)


def verify_repository_baseline(repo_root, manifest) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("unable to read repository baseline: " + result.stderr)
    actual = result.stdout.strip()
    if actual != manifest["baseline_commit"]:
        raise ValueError(
            f"repository baseline commit mismatch: expected {manifest['baseline_commit']}, got {actual}"
        )
    return True


def build_parser():
    parser = argparse.ArgumentParser(description="Verify a CoCo Task 6A deployment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--archive", type=Path, required=True)
    archive_parser.add_argument("--manifest", type=Path, required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--archive", type=Path, required=True)
    extract_parser.add_argument("--manifest", type=Path, required=True)
    extract_parser.add_argument("--destination", type=Path, required=True)
    tree_parser = subparsers.add_parser("tree")
    tree_parser.add_argument("--root", type=Path, required=True)
    tree_parser.add_argument("--manifest", type=Path, required=True)
    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--repo-root", type=Path, required=True)
    baseline_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = load_deployment_manifest(args.manifest)
    if args.command == "archive":
        verify_archive(args.archive, manifest)
    elif args.command == "extract":
        extract_verified_archive(args.archive, args.destination, manifest)
    elif args.command == "tree":
        verify_extracted_tree(args.root, manifest)
    else:
        verify_repository_baseline(args.repo_root, manifest)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
