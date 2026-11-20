# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d AS uv

FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS builder
COPY --from=uv /uv /usr/local/bin/uv
# agentx-harness depends on crick, which publishes no linux/arm64 wheel and compiles from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python

WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
# The source tree is put on PYTHONPATH instead of installed: infx.bench re-executes it under the
# AgentX runtime's Python 3.11, which cannot load a 3.12 site-packages.
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --locked --no-dev --no-install-project \
        --extra bench-client --python /usr/local/bin/python3.12

COPY infx/bench/agentic/requirements.txt /src/agentx-requirements.txt
# Same commands as infx.bench.agentic.venv.bootstrap, run once at build time instead of per job.
# requirements.txt leaves transitive versions open, so the exact Python and the resolution date
# are pinned here to make rebuilds of one version install the same runtime.
RUN uv python install 3.11.18 \
    && uv venv --managed-python --python 3.11.18 /opt/agentx-runtime/venv \
    && uv pip install --python /opt/agentx-runtime/venv/bin/python --no-config \
        --exclude-newer 2026-11-20T00:00:00Z -r /src/agentx-requirements.txt \
    && test -x /opt/agentx-runtime/venv/bin/agentx \
    && test -x /opt/agentx-runtime/venv/bin/hf

FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/agentx-runtime /opt/agentx-runtime
COPY infx /opt/inferencex/infx

# World-writable so `docker run --user "$(id -u):$(id -g)"` can write results and the HF cache.
RUN useradd --uid 1000 --create-home bench \
    && mkdir -p /results /cache/huggingface \
    && chmod 1777 /results /cache/huggingface

ARG IMAGE_VERSION
ARG VCS_REF
LABEL org.opencontainers.image.title="InferenceX benchmark client" \
      org.opencontainers.image.description="benchmark_serving and AgentX clients for InferenceX" \
      org.opencontainers.image.source="https://github.com/SemiAnalysisAI/InferenceX" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

# HOME must be writable for any `--user`, which has no passwd entry and would otherwise get `/`.
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    PYTHONPATH=/opt/inferencex \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    AIPERF_RUNTIME_DIR=/opt/agentx-runtime \
    RESULT_DIR=/results \
    AGENTIC_OUTPUT_DIR=/results \
    EVAL_ONLY=false \
    IS_MULTINODE=false \
    INFERENCEX_IMAGE_VERSION=${IMAGE_VERSION} \
    INFERENCEX_REVISION=${VCS_REF}

USER bench
WORKDIR /results
VOLUME ["/results", "/cache/huggingface"]
ENTRYPOINT ["tini", "--", "python3", "-m", "infx.bench.container"]
CMD ["--help"]
