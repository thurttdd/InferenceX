"""The client image entrypoint: which program each command becomes, and with what."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.bench import container
from infx.tests.bench.stubs import executable

ROOT = Path(__file__).resolve().parents[3]
RECORDER = """#!/bin/sh
python_path="$PYTHONPATH" "$REAL_PYTHON" -c '
import json, os, sys
with open(os.environ["CALLS"], "a") as calls:
    calls.write(json.dumps({"argv": sys.argv[1:], "pythonpath": os.environ["python_path"]}) + "\\n")
' "$0" "$@"
exit "${STUB_STATUS:-0}"
"""


@pytest.fixture
def run(tmp_path: Path):
    """Run the entrypoint as its own process with a recording ``python3`` first on ``PATH``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "python3", RECORDER)
    calls = tmp_path / "calls.jsonl"

    def invoke(*args: str, **environ: str) -> tuple[subprocess.CompletedProcess[str], list]:
        base = {k: v for k, v in os.environ.items() if not k.startswith(("AIPERF_", "INFERENCEX_"))}
        result = subprocess.run(
            [sys.executable, "-m", "infx.bench.container", *args],
            env={
                **base,
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "PYTHONPATH": f"{ROOT}{os.pathsep}/elsewhere",
                "REAL_PYTHON": sys.executable,
                "CALLS": str(calls),
                **environ,
            },
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        lines = calls.read_text().splitlines() if calls.exists() else []
        return result, [json.loads(line) for line in lines]

    return invoke


def test_serving_becomes_benchmark_serving_with_flags_untouched(run, tmp_path):
    result, calls = run("serving", "--model", "a model", "--save-result")

    assert result.returncode == 0, result.stderr
    assert calls == [
        {
            "argv": [
                str(tmp_path / "bin" / "python3"),
                "-m", "infx.bench_serving.benchmark_serving",
                "--model", "a model", "--save-result",
            ],
            "pythonpath": f"{ROOT}{os.pathsep}/elsewhere",
        }
    ]  # fmt: skip


def test_fixed_seq_point_runs_the_client_and_returns_its_status(run, tmp_path):
    result, calls = run(
        "fixed-seq", "point", "--base-url", "http://server:8000", "--model", "m",
        "--backend", "vllm", "--tokenizer", "tok", "--isl", "8", "--osl", "4",
        "--random-range-ratio", "1", "--conc", "2", "--num-prompts", "6",
        "--result", str(tmp_path / "out" / "r.json"),
        STUB_STATUS="3",
    )  # fmt: skip

    assert result.returncode == 3
    [call] = calls
    argv = call["argv"][1:]
    assert argv[:2] == ["-m", "infx.bench_serving.benchmark_serving"]
    assert argv[argv.index("--base-url") + 1] == "http://server:8000"
    assert argv[argv.index("--num-prompts") + 1] == "6"
    assert argv[argv.index("--result-dir") + 1] == str(tmp_path / "out")
    assert argv[argv.index("--result-filename") + 1] == "r.json"


def test_agentic_runs_in_the_prebuilt_runtime_without_bootstrapping(run, tmp_path):
    venv_bin = tmp_path / "runtime" / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    executable(venv_bin / "python", RECORDER)

    result, calls = run("agentic", AIPERF_RUNTIME_DIR=str(tmp_path / "runtime"))

    assert result.returncode == 0, result.stderr
    assert calls == [
        {
            "argv": [str(venv_bin / "python"), "-m", "infx.bench", "agentic"],
            "pythonpath": f"{ROOT}{os.pathsep}/elsewhere",
        }
    ]
    assert not (tmp_path / "runtime" / "uv-cache").exists()


@pytest.mark.parametrize(
    ("runtime_dir", "error"),
    [
        (None, "  - AIPERF_RUNTIME_DIR"),
        ("empty", "no AgentX runtime interpreter at {tmp}/empty/venv/bin/python"),
    ],
)
def test_agentic_without_a_prebuilt_runtime_fails_before_running(
    run, tmp_path, runtime_dir, error
):
    environ = {}
    if runtime_dir:
        (tmp_path / runtime_dir).mkdir()
        environ["AIPERF_RUNTIME_DIR"] = str(tmp_path / runtime_dir)

    result, calls = run("agentic", **environ)

    assert result.returncode == 1
    assert result.stderr.startswith("ERROR: ")
    assert error.format(tmp=tmp_path) in result.stderr
    assert calls == []


@pytest.mark.parametrize("argv", [[], ["bogus"], ["version", "extra"]])
def test_unknown_or_malformed_commands_print_usage(run, argv):
    result, calls = run(*argv)

    assert result.returncode == 2
    assert result.stderr.startswith("usage: inferencex-bench <command>")
    assert calls == []


def test_version_reports_image_revision_and_pinned_harness(tmp_path, monkeypatch, capsys):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("agentx-harness==9.8.7\ndatasets>=4.7.0\n")
    monkeypatch.setattr(container, "REQUIREMENTS", requirements)
    monkeypatch.setenv("INFERENCEX_IMAGE_VERSION", "1.2.3")
    monkeypatch.setenv("INFERENCEX_REVISION", "abc123")

    assert container.main(["version"]) == 0

    assert capsys.readouterr().out == "image: 1.2.3\nrevision: abc123\nagentx-harness: 9.8.7\n"


def test_version_outside_an_image_names_every_missing_input(monkeypatch, capsys):
    monkeypatch.delenv("INFERENCEX_IMAGE_VERSION", raising=False)
    monkeypatch.delenv("INFERENCEX_REVISION", raising=False)

    assert container.main(["version"]) == 1

    err = capsys.readouterr().err
    assert "  - INFERENCEX_IMAGE_VERSION\n" in err
    assert "  - INFERENCEX_REVISION" in err
