# Benchmark Client Container Image

The `inferencex-bench` image runs InferenceX's benchmark clients against an
already-running OpenAI-compatible server. It contains:

- `benchmark_serving` (`infx.bench_serving`) and its tokenizer dependencies,
  without torch or CUDA.
- The fixed-sequence driver, `python3 -m infx.bench fixed-seq point`.
- The AgentX replay command, `python3 -m infx.bench agentic`, with the pinned
  `agentx-harness` release pre-installed in a Python 3.11 runtime. Runs install nothing.

The image needs no GPU. It is published for `linux/amd64` and `linux/arm64`.

## Commands

The entrypoint is `python3 -m infx.bench.container`:

| Command | Runs |
| --- | --- |
| `serving <flags>` | `benchmark_serving` with the flags unchanged |
| `fixed-seq point <flags>` | One fixed-sequence point (`fixed-seq point --help`) |
| `agentic` | One AgentX concurrency point, configured by environment variables |
| `wait --url URL` | Blocks until the server answers |
| `version` | Image version, InferenceX revision, and `agentx-harness` version |

## Running

Results are written to `/results`. Mount a host directory there and run as your
own user, so the files belong to you:

```bash
IMAGE=ghcr.io/semianalysisai/inferencex-bench:1
mkdir -p results hf-cache
docker run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/results" -v "$PWD/hf-cache:/cache/huggingface" \
  -e HF_TOKEN \
  "$IMAGE" wait --url http://127.0.0.1:8000/v1/models
```

`/cache/huggingface` holds tokenizers and AgentX traces; mounting it avoids
downloading them again on every run. On Docker Desktop, replace `--network host`
and `127.0.0.1` with `host.docker.internal`.

### Fixed-sequence point

```bash
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/results" -v "$PWD/hf-cache:/cache/huggingface" -e HF_TOKEN \
  "$IMAGE" fixed-seq point \
    --base-url http://127.0.0.1:8000 \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --tokenizer deepseek-ai/DeepSeek-V4.1-Flash \
    --backend vllm \
    --isl 1024 --osl 1024 --random-range-ratio 0.8 \
    --conc 64 --num-prompts 640 \
    --result /results/dsv41flash_1k1k_conc64.json
```

### benchmark_serving with custom flags

```bash
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/results" -v "$PWD/hf-cache:/cache/huggingface" -e HF_TOKEN \
  "$IMAGE" serving \
    --base-url http://127.0.0.1:8000 --backend openai-chat --endpoint /v1/chat/completions \
    --model deepseek-ai/DeepSeek-V4.1-Flash --dataset-name random \
    --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 200 --max-concurrency 32 \
    --save-result --result-dir /results
```

### AgentX

`agentic` reads its configuration from the environment
([AgentX replay](agentx-standalone.md) explains the scenario). The image already
sets `RESULT_DIR=/results`, `AGENTIC_OUTPUT_DIR=/results`, `EVAL_ONLY=false`,
`IS_MULTINODE=false`, and `AIPERF_RUNTIME_DIR`. Put the rest in an env file:

```bash
cat > agentx.env <<'EOF'
AIPERF_SERVER_URL=http://127.0.0.1:8000
MODEL=deepseek-ai/DeepSeek-V4.1-Flash
MODEL_PREFIX=dsv41flash
FRAMEWORK=vllm
PRECISION=fp8
CONC=8
DURATION=3600
RESULT_FILENAME=dsv41flash_agentx_conc8
AIPERF_EXPERIMENTAL_FAST=0
AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0
AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=false
AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=0
EOF

docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/results" -v "$PWD/hf-cache:/cache/huggingface" -e HF_TOKEN \
  --env-file agentx.env "$IMAGE" agentic
```

Optional inputs such as `SERVED_MODEL_NAME`, `TOKENIZER`,
`AIPERF_SERVER_METRICS_URLS`, and `WEKA_LOADER_OVERRIDE` are passed the same way.
Restart the server for every concurrency point.

## Retrieving results

| Command | Output in `/results` |
| --- | --- |
| `fixed-seq point` | The `--result` JSON |
| `serving --save-result` | `--result-filename`, or `<backend>-<rate>qps-…-<timestamp>.json` |
| `agentic` | `<RESULT_FILENAME>.json` (aggregate), `aiperf_artifacts/`, `benchmark.log`, `benchmark_command.txt` |

Without a bind mount, keep the container and copy the directory out:

```bash
docker run --name bench-run -v /results "$IMAGE" fixed-seq point ...
docker cp bench-run:/results ./results
docker rm bench-run
```

## Versioning

Images use [Semantic Versioning](https://semver.org/). A release is the git tag
`image-vX.Y.Z`, following the `agentx-vX.Y.Z` convention of `agentx-harness`.

| Bump | When |
| --- | --- |
| MAJOR | A command, required input, environment variable, or `/results` layout changes incompatibly |
| MINOR | A new command or capability, an `agentx-harness` release, or a major dependency upgrade |
| PATCH | Base-image, security, or dependency fixes with no behavior change |

The AgentX runtime installs `infx/bench/agentic/requirements.txt` with its
dependencies resolved as of the `--exclude-newer` date in `docker/Dockerfile`, on a
pinned Python 3.11 release. Rebuilding a version installs the same runtime. Move the
date forward on purpose, together with an `agentx-harness` upgrade or a PATCH release.

| Event | Tags |
| --- | --- |
| `image-vX.Y.Z` tag | `X.Y.Z`, `X.Y`, `X`, `latest` |
| `image-vX.Y.Z-rc.N` tag | `X.Y.Z-rc.N` only |
| Push to `main` | `main`, `sha-<short>`, `0.0.0-main.<short>` |

Each image records its version and source commit in the
`org.opencontainers.image.version` and `org.opencontainers.image.revision`
labels, and `docker run IMAGE version` prints them. Pin a full version or a
digest when reproducing results; `X` and `latest` move.

## Building

`docker/build.sh` wraps `docker buildx build`; every input is required:

```bash
cd inferencex-e2e
IMAGE=inferencex-bench IMAGE_VERSION=0.0.0-dev PLATFORMS=linux/amd64 PUSH=false ./docker/build.sh
```

| Input | Meaning |
| --- | --- |
| `IMAGE` | Repository name; the tag is `IMAGE_VERSION` |
| `IMAGE_VERSION` | `X.Y.Z` or `X.Y.Z-<prerelease>` |
| `PLATFORMS` | One platform, or a comma-separated list when `PUSH=true` |
| `PUSH` | `true` pushes to the registry; `false` loads into the local image store |

Extra arguments are passed to `docker buildx build`, for example `--no-cache`.
A multi-platform local build needs a registry login and `PUSH=true`.

## Testing

Run from `inferencex-e2e/`:

```bash
# Entrypoint and build-script tests; also run by CI.
uv run --locked --all-extras --group test python -m pytest -c pyproject.toml \
  infx/tests/bench/test_container.py infx/tests/container/test_build_script.py

# Image tests: start a local fake server and run the built image against it.
INFERENCEX_IMAGE=inferencex-bench:0.0.0-dev \
  uv run --locked --all-extras --group test python -m pytest -c pyproject.toml \
  infx/tests/container/test_image.py
```

The image tests need Docker and no network or GPU. They check that a
fixed-sequence run writes a complete result owned by the caller, and that
`agentic` uses the pre-installed AgentX runtime. A full AgentX run needs a real
server and the Hugging Face trace dataset, so it is a manual check.

## Publishing

`.github/workflows/container-image.yml` runs on pull requests, on pushes to
`main`, on `image-v*` tags, and on manual dispatch:

1. `prepare` resolves the version from the tag, or `0.0.0-<ref>.<sha>`.
2. `test` builds the image natively on amd64 and arm64 runners and runs
   `test_image.py` against each.
3. `publish` (when enabled) pushes the exact image `test` ran, unchanged, by digest.
4. `merge` combines the digests into one multi-arch manifest and applies the tags.

Only pushes to `main` and `image-v*` tags publish. Pull requests and manual runs on
other branches only build and test. To release:

```bash
git tag -a image-v1.2.0 -m "inferencex-bench 1.2.0"
git push origin image-v1.2.0
```

## Infrastructure requirements

Nothing publishes until a maintainer completes these steps.

| Item | Setup |
| --- | --- |
| Publishing switch | Repository variable `IMAGE_PUBLISH_ENABLED=true` |
| GitHub environment | Create `container-image`. Optionally require reviewers and limit deployments to `main` and `image-v*` tags |
| GHCR | No account needed; the job's `GITHUB_TOKEN` pushes `ghcr.io/semianalysisai/inferencex-bench`. After the first push, set the package's visibility to public and link it to this repository |
| Docker Hub (optional) | Create the organization and repository, for example `semianalysis/inferencex-bench`, and a read/write access token. In the `container-image` environment, add secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` and variable `DOCKERHUB_IMAGE=semianalysis/inferencex-bench` |
| Runners | GitHub-hosted `ubuntu-latest` and `ubuntu-24.04-arm`; public repositories use both free. No GPU or self-hosted runner is needed |
| Tag protection | A ruleset on `image-v*` tags that allows only release maintainers to create, update, or delete them |
| Dependabot | `.github/dependabot.yml` reports security updates to the base images |
