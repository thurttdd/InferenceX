"""``python3 -m infx.bench.container``: the benchmark client image's entrypoint."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import NoReturn

from infx.bench import env, proc
from infx.bench.__main__ import main as bench_main
from infx.bench.agentic.venv import REQUIREMENTS, Runtime, harness_version

PYTHON = "python3"
DELEGATED = ("fixed-seq", "wait")
USAGE = """\
usage: inferencex-bench <command> [args]

commands:
  serving    run benchmark_serving with the given flags (serving --help)
  fixed-seq  run one fixed-sequence point (fixed-seq point --help)
  agentic    replay AgentX traces at one concurrency, configured by environment variables
  wait       block until a server answers (wait --help)
  version    print the image version, InferenceX revision, and agentx-harness version
"""


def _exec(argv: Sequence[str], environ: dict[str, str]) -> NoReturn:
    """Replace this process with ``argv`` so signals reach the client directly."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(argv[0], list(argv), environ)  # noqa: S606


def _child_env() -> dict[str, str]:
    """This environment with the InferenceX source tree first on ``PYTHONPATH``."""
    return {**os.environ, "PYTHONPATH": proc.pythonpath()}


def serving(argv: list[str]) -> NoReturn:
    """Run ``benchmark_serving`` with ``argv`` unchanged."""
    _exec([PYTHON, "-m", "infx.bench_serving.benchmark_serving", *argv], _child_env())


def agentic(argv: list[str]) -> NoReturn:
    """Run ``agentic`` inside the image's pre-built AgentX runtime, so nothing is installed."""
    env.require("AIPERF_RUNTIME_DIR")
    runtime = Runtime.for_job(os.environ)
    if not os.access(runtime.python, os.X_OK):
        raise env.BenchError(f"no AgentX runtime interpreter at {runtime.python}")
    runtime.exec_python(["-m", "infx.bench", "agentic", *argv], _child_env())


def version() -> int:
    """Print the image version, the InferenceX revision, and the pinned agentx-harness."""
    values = env.require("INFERENCEX_IMAGE_VERSION", "INFERENCEX_REVISION")
    harness = harness_version(REQUIREMENTS)
    if harness is None:
        raise env.BenchError(f"{REQUIREMENTS} pins no agentx-harness version")
    print(f"image: {values['INFERENCEX_IMAGE_VERSION']}")
    print(f"revision: {values['INFERENCEX_REVISION']}")
    print(f"agentx-harness: {harness}")
    return 0


def main(argv: list[str]) -> int:
    """Dispatch ``argv[0]``; a ``BenchError`` is one ``ERROR:`` line, not a traceback."""
    if argv[:1] in (["-h"], ["--help"]):
        print(USAGE, end="")
        return 0
    command = argv[0] if argv else None
    if command in DELEGATED:
        return bench_main(argv)
    try:
        if command == "serving":
            serving(argv[1:])
        elif command == "agentic":
            agentic(argv[1:])
        elif argv == ["version"]:
            return version()
    except env.BenchError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(USAGE, end="", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
