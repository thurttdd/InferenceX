#!/usr/bin/env bash
set -eo pipefail

docker_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
context="$(dirname "$docker_dir")"
source "$context/benchmarks/check_env.sh"
check_env_vars IMAGE IMAGE_VERSION PLATFORMS PUSH

if [[ ! "$IMAGE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "Error: IMAGE_VERSION must be X.Y.Z or X.Y.Z-<prerelease>, got '$IMAGE_VERSION'" >&2
    exit 1
fi

case "$PUSH" in
    true)
        output=--push
        ;;
    false)
        # The local image store holds one platform per tag, so a multi-platform build must push.
        if [[ "$PLATFORMS" == *,* ]]; then
            echo "Error: PLATFORMS=$PLATFORMS lists several platforms; set PUSH=true" >&2
            exit 1
        fi
        output=--load
        ;;
    *)
        echo "Error: PUSH must be true or false, got '$PUSH'" >&2
        exit 1
        ;;
esac

vcs_ref="$(git -C "$context" rev-parse HEAD)"

exec docker buildx build \
    --file "$docker_dir/Dockerfile" \
    --platform "$PLATFORMS" \
    --build-arg "IMAGE_VERSION=$IMAGE_VERSION" \
    --build-arg "VCS_REF=$vcs_ref" \
    --tag "$IMAGE:$IMAGE_VERSION" \
    "$output" \
    "$@" \
    "$context"
