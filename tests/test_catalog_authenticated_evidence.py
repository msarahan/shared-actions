# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# ruff: noqa: S101, S603, S607, S108
"""Use pytest assertions and fixed repository shell scripts in isolated fixtures."""

import json
import os
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _environment(tmp_path, descriptor):
    (tmp_path / "demo.jar").write_bytes(b"artifact")
    (tmp_path / "package.json").write_text(json.dumps({"ecosystem": "maven", "name": "demo", "version": "1.0"}))
    (tmp_path / "dependency.cdx.json").write_text('{"bomFormat":"CycloneDX"}')
    (tmp_path / "bundle.jsonl").write_text("synthetic attestation fixture")
    return {
        **os.environ,
        "RELEASE_CATALOG_KEY": "maven:demo",
        "RELEASE_ARTIFACT_DIRECTORY": str(tmp_path),
        "RELEASE_SOURCE_ARTIFACT_NAME": "demo",
        "RELEASE_ENTRIES_NAME": "release-catalog-entries.json",
        "RAPIDS_SHA": "a" * 40,
        "GITHUB_REPOSITORY": "rapidsai/demo",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKFLOW_REF": "rapidsai/demo/.github/workflows/build.yaml@refs/heads/main",
        "RELEASE_ARTIFACTS": json.dumps([descriptor]),
    }


def _descriptor():
    return {
        "path": "demo.jar",
        "package_identity_file": "package.json",
        "dependency_sbom": "dependency.cdx.json",
        "attestation_bundle": "bundle.jsonl",
    }


def test_materializer_preserves_dependency_sbom_and_bundle(tmp_path):
    environment = _environment(tmp_path, _descriptor())
    subprocess.run(["bash", str(_ROOT / "release-catalog/materialize.sh")], env=environment, check=True)
    entry = json.loads((tmp_path / "release-catalog-entries.json").read_text())["entries"][0]
    assert entry["sbom_kind"] == "producer-dependency"
    assert entry["sboms"]["cyclonedx"] == "dependency.cdx.json"
    assert entry["signature"] == "bundle.jsonl"
    assert "sbom" not in entry
    assert entry["sboms"]["spdx"].startswith("release-evidence/")


@pytest.mark.parametrize("missing", ["dependency_sbom", "attestation_bundle"])
def test_config_rejects_partial_evidence_association(tmp_path, missing):
    descriptor = _descriptor()
    del descriptor[missing]
    config = {"release_catalog_key": "maven:demo", "artifacts": [descriptor]}
    environment = {
        **os.environ,
        "RELEASE_CATALOG_CONFIG": json.dumps(config),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    result = subprocess.run(
        ["bash", str(_ROOT / "release-catalog/validate-config.sh")],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be supplied together" in result.stderr


@pytest.mark.parametrize("path", ["../outside.json", "/tmp/outside.json", "missing.json"])
def test_evidence_paths_cannot_escape_or_silently_fallback(tmp_path, path):
    descriptor = {**_descriptor(), "dependency_sbom": path}
    result = subprocess.run(
        ["bash", str(_ROOT / "release-catalog/materialize.sh")],
        env=_environment(tmp_path, descriptor),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "release-catalog-entries.json").exists()


def test_evidence_symlink_cannot_escape_artifact_directory(tmp_path):
    environment = _environment(tmp_path, _descriptor())
    external = tmp_path.parent / "outside-sbom.json"
    external.write_text("outside")
    (tmp_path / "dependency.cdx.json").unlink()
    (tmp_path / "dependency.cdx.json").symlink_to(external)
    result = subprocess.run(
        ["bash", str(_ROOT / "release-catalog/materialize.sh")],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must resolve inside" in result.stderr
