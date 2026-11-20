"""The built client image against a local fake server; needs ``INFERENCEX_IMAGE`` and Docker."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

IMAGE = os.environ.get("INFERENCEX_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="INFERENCEX_IMAGE names no built client image")
VOCAB = [f"w{i}" for i in range(256)]


class FakeServer:
    """Streams one ``x`` token per requested output token from ``/v1/completions``."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._send(200, b'{"data": [{"id": "fake"}]}', "application/json")

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                chunks = [{"choices": [{"text": "x "}]}] * body["max_tokens"]
                chunks.append({"choices": [], "usage": {"completion_tokens": body["max_tokens"]}})
                events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                self._send(200, f"{events}data: [DONE]\n\n".encode(), "text/event-stream")

            def _send(self, status: int, payload: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("0.0.0.0", 0), Handler)  # noqa: S104
        self.port = self.httpd.server_port


@pytest.fixture
def fake_server() -> Iterator[FakeServer]:
    """A running ``FakeServer`` reachable from containers on any host interface."""
    server = FakeServer()
    threading.Thread(target=server.httpd.serve_forever, args=(0.01,), daemon=True).start()
    yield server
    server.httpd.shutdown()
    server.httpd.server_close()


@pytest.fixture
def tokenizer(tmp_path: Path) -> Path:
    """A whitespace word-level tokenizer, so no run downloads one."""
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    vocab = {token: index for index, token in enumerate([*VOCAB, "[UNK]"])}
    (directory / "tokenizer.json").write_text(json.dumps({
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
        "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }))  # fmt: skip
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"})
    )
    return directory


def docker_run(*args: str, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    """``docker run --rm`` as the current user, with the host reachable as ``host.docker.internal``."""
    return subprocess.run(
        [
            "docker", "run", "--rm",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--add-host", "host.docker.internal:host-gateway",
            "--env", "HF_HUB_OFFLINE=1",
            *args,
        ],
        capture_output=True, text=True, timeout=timeout, check=False,
    )  # fmt: skip


def test_fixed_seq_point_writes_a_complete_result_owned_by_the_caller(
    tmp_path, fake_server, tokenizer
):
    results = tmp_path / "results"
    results.mkdir()

    run = docker_run(
        "--volume", f"{results}:/results",
        "--volume", f"{tokenizer}:/tokenizer:ro",
        IMAGE, "fixed-seq", "point",
        "--base-url", f"http://host.docker.internal:{fake_server.port}",
        "--model", "fake", "--backend", "vllm", "--tokenizer", "/tokenizer",
        "--isl", "32", "--osl", "8", "--random-range-ratio", "1",
        "--conc", "2", "--num-prompts", "6", "--result", "/results/point.json",
    )  # fmt: skip

    assert run.returncode == 0, run.stdout + run.stderr
    result_file = results / "point.json"
    assert result_file.stat().st_uid == os.getuid()
    result = json.loads(result_file.read_text())
    assert result["completed"] == 6
    assert result["total_input_tokens"] == 6 * 32
    assert result["total_output_tokens"] == 6 * 8
    assert result["benchmark_outcome"]["status"] == "passed"
    assert len(fake_server.requests) == 6 + 2 * 2
    assert {request["max_tokens"] for request in fake_server.requests} == {8}
    assert {request["model"] for request in fake_server.requests} == {"fake"}


def test_agentic_uses_the_baked_runtime_and_needs_no_installer(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    environ = {
        "MODEL": "fake", "MODEL_PREFIX": "fake", "FRAMEWORK": "vllm", "PRECISION": "fp8",
        "CONC": "1", "DURATION": "60", "RESULT_FILENAME": "agentic",
        "AIPERF_SERVER_URL": "http://127.0.0.1:9", "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "0",
        "AIPERF_EXPERIMENTAL_FAST": "0", "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "0",
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
    }  # fmt: skip
    env_args = [arg for name, value in environ.items() for arg in ("--env", f"{name}={value}")]

    run = docker_run("--network", "none", "--volume", f"{results}:/results", *env_args, IMAGE, "agentic")

    assert run.returncode != 0
    assert "Preparing the AgentX runtime" not in run.stdout
    assert (
        "+ /opt/agentx-runtime/venv/bin/hf download --repo-type dataset "
        "semianalysisai/cc-traces-weka-062126"
    ) in run.stderr


def test_an_arbitrary_user_can_write_under_home():
    probe = "import pathlib; p = pathlib.Path.home() / '.cache' / 'probe'; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('ok'); print(p.read_text())"

    run = docker_run("--entrypoint", "python3", IMAGE, "-c", probe)

    assert run.returncode == 0, run.stderr
    assert run.stdout == "ok\n"
