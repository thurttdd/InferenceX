"""``docker/build.sh`` run with recording ``docker`` and ``git`` stubs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.tests.bench.stubs import executable

ROOT = Path(__file__).resolve().parents[3]
DOCKER = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["STUB_DOCKER_TRACE"], "a") as trace:
    trace.write(json.dumps(sys.argv[1:]) + "\\n")
"""
GIT = "#!/bin/sh\necho 0123abcd\n"
VALID = {"IMAGE": "registry.test/bench", "PLATFORMS": "linux/arm64", "PUSH": "false"}


def build(tmp_path: Path, env: dict[str, str], *args: str) -> tuple[subprocess.CompletedProcess, list]:
    """Run the build script; return its result and every ``docker`` invocation."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    executable(bin_dir / "docker", DOCKER)
    executable(bin_dir / "git", GIT)
    trace = tmp_path / "docker.jsonl"
    result = subprocess.run(
        ["bash", str(ROOT / "docker" / "build.sh"), *args],
        env={"PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin", "STUB_DOCKER_TRACE": str(trace), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    return result, calls


@pytest.mark.parametrize(
    ("push", "platforms", "output"),
    [("false", "linux/arm64", "--load"), ("true", "linux/amd64,linux/arm64", "--push")],
)
def test_build_tags_the_version_and_records_the_revision(tmp_path, push, platforms, output):
    env = {**VALID, "IMAGE_VERSION": "1.4.0-rc.2", "PLATFORMS": platforms, "PUSH": push}

    result, calls = build(tmp_path, env, "--no-cache")

    assert result.returncode == 0, result.stderr
    assert calls == [[
        "buildx", "build",
        "--file", str(ROOT / "docker" / "Dockerfile"),
        "--platform", platforms,
        "--build-arg", "IMAGE_VERSION=1.4.0-rc.2",
        "--build-arg", "VCS_REF=0123abcd",
        "--tag", "registry.test/bench:1.4.0-rc.2",
        output,
        "--no-cache",
        str(ROOT),
    ]]  # fmt: skip


def test_missing_inputs_are_all_named_and_nothing_builds(tmp_path):
    result, calls = build(tmp_path, {"IMAGE": "registry.test/bench"})

    assert result.returncode == 1
    assert "  - IMAGE_VERSION\n  - PLATFORMS\n  - PUSH\n" in result.stdout
    assert calls == []


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"IMAGE_VERSION": "v1.2.3"}, "IMAGE_VERSION must be X.Y.Z"),
        ({"IMAGE_VERSION": "1.2"}, "IMAGE_VERSION must be X.Y.Z"),
        ({"PUSH": "yes"}, "PUSH must be true or false"),
        ({"PLATFORMS": "linux/amd64,linux/arm64"}, "lists several platforms; set PUSH=true"),
    ],
)
def test_invalid_inputs_fail_before_building(tmp_path, overrides, error):
    result, calls = build(tmp_path, {**VALID, "IMAGE_VERSION": "1.2.3", **overrides})

    assert result.returncode == 1
    assert error in result.stderr
    assert calls == []
