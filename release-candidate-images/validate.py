# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate train-owned executable container inputs before a container starts."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path


def canonical_train_digest(train):
    if not isinstance(train, dict):
        raise ValueError("candidate train must be a JSON object")
    payload = dict(train)
    recorded = payload.pop("train_sha256", None)
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if recorded not in {None, actual}:
        raise ValueError("candidate train digest does not match canonical content")
    return actual


def validate(train, digest, supplied, matrix, family, custom_image, image_series="26.10"):
    if not isinstance(image_series, str) or not re.fullmatch(r"[0-9]{2}\.(?:0[1-9]|1[0-2])", image_series):
        raise ValueError("candidate image series must use YY.MM format")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or canonical_train_digest(train) != digest:
        raise ValueError("candidate train digest does not match canonical content")
    images = train.get("candidate_container_images")
    if not isinstance(images, dict) or not images:
        raise ValueError("candidate train must contain a reviewed container image map")
    for name, image in images.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(image, str)
            or not re.fullmatch(r"[A-Za-z0-9./_-]+@sha256:[0-9a-f]{64}", image)
        ):
            raise ValueError("candidate container images must use immutable registry digests")
    if supplied != {} and supplied != images:
        raise ValueError("caller container image map differs from reviewed train")
    if custom_image:
        expected = {custom_image}
    else:
        if family not in {"ci-conda", "ci-wheel"} or not isinstance(matrix, dict) or not matrix.get("include"):
            raise ValueError("candidate image validation requires a complete build matrix")
        expected = {
            f"rapidsai/{family}:{image_series}-cuda{cell['CUDA_VER']}-{cell['LINUX_VER']}-py{cell['PY_VER']}"
            for cell in matrix["include"]
        }
    missing = sorted(expected - set(images))
    if missing:
        raise ValueError("candidate train lacks image digests for: " + ", ".join(missing))
    return images


def main():
    images = validate(
        json.loads(Path(os.environ["TRAIN_FILE"]).read_text()),
        os.environ["TRAIN_SHA256"],
        json.loads(os.environ["CALLER_IMAGES"]),
        json.loads(os.environ["BUILD_MATRIX"] or "{}"),
        os.environ["IMAGE_FAMILY"],
        os.environ["CUSTOM_IMAGE"],
        os.environ.get("IMAGE_SERIES", "26.10"),
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write("images=" + json.dumps(images, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    if sys.argv[1:]:
        print(canonical_train_digest(json.loads(Path(sys.argv[1]).read_text())))
    else:
        main()
