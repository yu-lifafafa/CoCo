import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml


EXPECTED_DATA_SHA = "ce51205b537c3d1eee28b3d8112ce1edb6454c6f16cd3b441387bff2be2c56b5"


def _git(repo, *args):
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Deployment Tests")
    (repo / "source.py").write_text("VALUE = 'baseline'\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / "image.jpg").write_bytes(b"excluded-image")
    _git(repo, "add", "source.py", "README.md", "data/image.jpg")
    _git(repo, "commit", "-m", "baseline")
    baseline = _git(repo, "rev-parse", "HEAD")
    (repo / "source.py").write_text("VALUE = 'working-tree'\n", encoding="utf-8")
    (repo / "extra").mkdir()
    (repo / "extra" / "required.py").write_text("REQUIRED = True\n", encoding="utf-8")
    (repo / "checkpoint.pth").write_bytes(b"excluded-checkpoint")
    (repo / "image_manifest_updated.csv").write_text("excluded\n", encoding="utf-8")
    return repo, baseline


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_collect_snapshot_uses_working_tree_bytes_and_explicit_untracked(tmp_path):
    from scripts.build_deployment_package import collect_snapshot_files

    repo, _ = _make_repo(tmp_path)
    members = collect_snapshot_files(repo, ["extra/required.py"])
    assert members == ["README.md", "extra/required.py", "source.py"]
    assert (repo / members[-1]).read_text("utf-8") == "VALUE = 'working-tree'\n"
    assert "data/image.jpg" not in members
    assert "checkpoint.pth" not in members
    assert "image_manifest_updated.csv" not in members


def test_collect_snapshot_rejects_missing_required_untracked_file(tmp_path):
    from scripts.build_deployment_package import collect_snapshot_files

    repo, _ = _make_repo(tmp_path)
    with pytest.raises(FileNotFoundError, match="required deployment file"):
        collect_snapshot_files(repo, ["extra/missing.py"])


def test_package_is_deterministic_and_records_baseline_and_external_data_sha(tmp_path):
    from scripts.build_deployment_package import build_package

    repo, baseline = _make_repo(tmp_path)
    outputs = []
    for suffix in ("one", "two"):
        archive = tmp_path / f"deploy-{suffix}.zip"
        manifest = tmp_path / f"manifest-{suffix}.json"
        payload = build_package(
            repo,
            archive,
            manifest,
            baseline,
            required_untracked=["extra/required.py"],
            expected_data_manifest_sha256=EXPECTED_DATA_SHA,
        )
        outputs.append((archive, manifest, payload))
    assert outputs[0][0].read_bytes() == outputs[1][0].read_bytes()
    assert outputs[0][2] == outputs[1][2]
    assert outputs[0][2]["archive_sha256"] == _sha(outputs[0][0])
    assert outputs[0][2]["baseline_commit"] == baseline
    assert outputs[0][2]["expected_data_manifest_sha256"] == EXPECTED_DATA_SHA
    assert [item["path"] for item in outputs[0][2]["files"]] == [
        "README.md",
        "extra/required.py",
        "source.py",
    ]
    assert json.loads(outputs[0][1].read_text("utf-8")) == outputs[0][2]


def test_verify_archive_rejects_archive_hash_mismatch(tmp_path):
    from deploy.verify_deployment import verify_archive
    from scripts.build_deployment_package import build_package

    repo, baseline = _make_repo(tmp_path)
    archive = tmp_path / "deploy.zip"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_package(
        repo,
        archive,
        manifest_path,
        baseline,
        required_untracked=["extra/required.py"],
    )
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="archive SHA-256"):
        verify_archive(archive, manifest)


def test_verify_archive_rejects_member_corruption_even_with_updated_archive_sha(tmp_path):
    from deploy.verify_deployment import verify_archive
    from scripts.build_deployment_package import build_package

    repo, baseline = _make_repo(tmp_path)
    archive = tmp_path / "deploy.zip"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_package(
        repo,
        archive,
        manifest_path,
        baseline,
        required_untracked=["extra/required.py"],
    )
    corrupted = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(corrupted, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "source.py":
                data = bytes([data[0] ^ 1]) + data[1:]
            target.writestr(info, data)
    altered_manifest = copy.deepcopy(manifest)
    altered_manifest["archive_sha256"] = _sha(corrupted)
    with pytest.raises(ValueError, match="member SHA-256"):
        verify_archive(corrupted, altered_manifest)


@pytest.mark.parametrize(
    "unsafe_name", ["../escape.py", "/absolute.py", "C:/drive.py", "safe/../../escape.py"]
)
def test_validate_member_name_rejects_unsafe_paths(unsafe_name):
    from deploy.verify_deployment import validate_member_name

    with pytest.raises(ValueError, match="unsafe archive member"):
        validate_member_name(unsafe_name)


def test_verify_archive_rejects_duplicate_normalized_members(tmp_path):
    from deploy.verify_deployment import verify_archive

    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a/b.py", b"one")
        handle.writestr("a\\b.py", b"two")
    manifest = {"archive_sha256": _sha(archive), "files": []}
    with pytest.raises(ValueError, match="duplicate normalized"):
        verify_archive(archive, manifest)


def test_verify_archive_rejects_symlink_members(tmp_path):
    from deploy.verify_deployment import verify_archive

    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link.py")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "target.py")
    manifest = {"archive_sha256": _sha(archive), "files": []}
    with pytest.raises(ValueError, match="symbolic link"):
        verify_archive(archive, manifest)


def test_extract_and_verify_tree_require_exact_file_set_and_hashes(tmp_path):
    from deploy.verify_deployment import (
        extract_verified_archive,
        verify_extracted_tree,
    )
    from scripts.build_deployment_package import build_package

    repo, baseline = _make_repo(tmp_path)
    archive = tmp_path / "deploy.zip"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_package(
        repo,
        archive,
        manifest_path,
        baseline,
        required_untracked=["extra/required.py"],
    )
    extracted = tmp_path / "extracted"
    extract_verified_archive(archive, extracted, manifest)
    assert verify_extracted_tree(extracted, manifest) is True
    (extracted / "source.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="extracted file SHA-256"):
        verify_extracted_tree(extracted, manifest)
    (extracted / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify_extracted_tree(extracted, manifest)


def test_repository_baseline_verifier_rejects_wrong_commit(tmp_path):
    from deploy.verify_deployment import verify_repository_baseline

    repo, baseline = _make_repo(tmp_path)
    assert verify_repository_baseline(repo, {"baseline_commit": baseline}) is True
    with pytest.raises(ValueError, match="baseline commit"):
        verify_repository_baseline(repo, {"baseline_commit": "0" * 40})


def test_environment_contract_pins_python_but_not_unverified_cuda_stack():
    environment_path = Path(__file__).resolve().parents[1] / "deploy" / "environment.yml"
    payload = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    dependencies = payload["dependencies"]
    flattened = []
    for dependency in dependencies:
        if isinstance(dependency, str):
            flattened.append(dependency.lower())
        else:
            flattened.extend(item.lower() for item in dependency.get("pip", []))
    assert "python=3.10" in flattened
    for required in (
        "numpy",
        "scipy",
        "pillow",
        "scikit-learn",
        "pyyaml",
        "tqdm",
        "tensorboard",
        "randaugment",
        "timm",
        "pytest",
    ):
        assert required in flattened
    forbidden_fragments = ("pytorch", "torch=", "torchvision", "cuda", "cudatoolkit")
    assert not any(
        fragment in dependency
        for dependency in flattened
        for fragment in forbidden_fragments
    )


def test_server_runbook_uses_live_cli_contract_and_bounded_commands():
    from deploy.verify_deployment import build_parser as verifier_parser
    from scripts.smoke_tobacco_coco import build_parser as smoke_parser

    runbook_path = Path(__file__).resolve().parents[1] / "deploy" / "server_setup.md"
    text = runbook_path.read_text(encoding="utf-8")
    smoke_options = {
        option
        for action in smoke_parser()._actions
        for option in action.option_strings
    }
    required_smoke_options = {
        "--manifest-path",
        "--data-root",
        "--img-size",
        "--batch-size",
        "--device",
        "--amp",
        "--seed",
        "--backbone-init",
        "--backbone-weights",
        "--output-dir",
        "--max-steps",
        "--scenarios",
        "--expected-manifest-sha256",
    }
    assert required_smoke_options <= smoke_options
    assert all(option in text for option in required_smoke_options)
    assert {"archive", "extract", "tree", "baseline"} <= set(
        verifier_parser()._subparsers._group_actions[0].choices
    )
    for variable in (
        "COCO_BASELINE_DIR",
        "COCO_WORKDIR",
        "PACKAGE_DIR",
        "MANIFEST_PATH",
        "DATA_ROOT",
        "BACKBONE_WEIGHTS",
        "BACKBONE_SHA256",
        "SMOKE_OUTPUT",
        "CUDA_DEVICE",
    ):
        assert f"{variable}=" in text
    assert "342c4e35b19fc773ae71565dbe17be70d56bfcdc" in text
    assert EXPECTED_DATA_SHA in text
    assert "git diff --quiet" in text
    assert "sha256sum" in text
    assert "verify_deployment.py archive" in text
    assert "verify_deployment.py extract" in text
    assert re.search(r'verify_deployment\.py"? tree', text)
    for batch_size in (1, 2, 4):
        assert f"--batch-size {batch_size}" in text
    assert all(f"--max-steps {steps}" in text for steps in (1, 2, 4))
    assert "python train.py" not in text
    assert "SERVER GPU TEST: NOT RUN" in text
