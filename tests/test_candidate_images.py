# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise canonical train hashing and pre-container image admission."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "release-candidate-images" / "validate.py"
_SPEC = importlib.util.spec_from_file_location("candidate_images", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load candidate image validator")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_exact_canonical_train_hash_including_unicode_and_numbers():
    train = {"description": "é", "number": 1.0, "nested": {"z": 0, "a": True}}
    expected = hashlib.sha256(json.dumps(train, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if _MODULE.canonical_train_digest(train) != expected:
        pytest.fail("canonical digest differs from release-platform serialization")
    train["train_sha256"] = expected
    if _MODULE.canonical_train_digest(train) != expected:
        pytest.fail("canonical digest differs from release-platform serialization")
    train["train_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canonical content"):
        _MODULE.canonical_train_digest(train)


@pytest.mark.parametrize("family", ["ci-conda", "ci-wheel"])
def test_complete_matrix_requires_every_immutable_image(family):
    matrix = {"include": [{"CUDA_VER": "13.0", "LINUX_VER": "ubuntu24.04", "PY_VER": "3.13"}]}
    name = f"rapidsai/{family}:26.10-cuda13.0-ubuntu24.04-py3.13"
    train = {"candidate_container_images": {name: "registry/ci@sha256:" + "a" * 64}}
    digest = _MODULE.canonical_train_digest(train)
    if _MODULE.validate(train, digest, {}, matrix, family, "") != train["candidate_container_images"]:
        pytest.fail("valid reviewed matrix images were rejected")
    matrix["include"].append({"CUDA_VER": "13.0", "LINUX_VER": "ubuntu24.04", "PY_VER": "3.14"})
    with pytest.raises(ValueError, match=r"lacks image digests.*py3\.14"):
        _MODULE.validate(train, digest, {}, matrix, family, "")


@pytest.mark.parametrize("image", ["registry/ci:latest", "registry/ci@sha256:abc", None])
def test_reject_mutable_train_images(image):
    train = {"candidate_container_images": {"ci:tag": image}}
    with pytest.raises(ValueError, match="immutable registry"):
        _MODULE.validate(train, _MODULE.canonical_train_digest(train), {}, {}, "", "ci:tag")


def test_custom_images_cannot_be_overridden_or_omitted():
    train = {"candidate_container_images": {"ci:tag": "registry/ci@sha256:" + "a" * 64}}
    digest = _MODULE.canonical_train_digest(train)
    if _MODULE.validate(train, digest, {}, {}, "", "ci:tag") != train["candidate_container_images"]:
        pytest.fail("valid reviewed custom image was rejected")
    with pytest.raises(ValueError, match="differs from reviewed"):
        _MODULE.validate(train, digest, {"ci:tag": "other"}, {}, "", "ci:tag")
    with pytest.raises(ValueError, match="lacks image digests"):
        _MODULE.validate(train, digest, {}, {}, "", "absent")


def test_current_image_series_requires_matching_reviewed_image():
    matrix = {"include": [{"CUDA_VER": "13.0", "LINUX_VER": "ubuntu24.04", "PY_VER": "3.13"}]}
    name = "rapidsai/ci-conda:26.12-cuda13.0-ubuntu24.04-py3.13"
    train = {"candidate_container_images": {name: "registry/ci@sha256:" + "a" * 64}}
    digest = _MODULE.canonical_train_digest(train)
    if _MODULE.validate(train, digest, {}, matrix, "ci-conda", "", "26.12") != train["candidate_container_images"]:
        pytest.fail("reviewed current-series image was rejected")
    with pytest.raises(ValueError, match="lacks image digests"):
        _MODULE.validate(train, digest, {}, matrix, "ci-conda", "", "26.10")


@pytest.mark.parametrize("series", ["26.12/../../bad", "26.13", "latest", "26.1", None])
def test_reject_unreviewable_image_series(series):
    with pytest.raises(ValueError, match=r"YY\.MM"):
        _MODULE.validate({}, "0" * 64, {}, {}, "", "ci:tag", series)
