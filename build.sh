#!/usr/bin/env bash
# build.sh — reproducible build of the all-NVFP4 DFlash2 vLLM image.
#
# Recipe (matches the validated local build, 2026-08-21/24):
#   upstream vllm v0.27.1 source (commit 6e448d0ea)
#   + 0001-v0271-sm12x-dflash2-nvfp4.patch (51 files, Python-only overlay)
#   + official vLLM Dockerfile (CUDA 13.0.3, FlashInfer 0.6.16.post3 + PR #4346)
#
# Prereqs: docker, git, ~15 GB disk for the tree, ~1 h build time.
# Usage:   ./build.sh          # build only (SM120/x86_64)
#          SPARK=1 ./build.sh  # SM121/aarch64 (DGX Spark / GB10)
#          VISION_MROPE=1 ./build.sh  # full source rebuild with .3 patch
#          ./build.sh --push   # build + tag + push to ghcr.io (needs GHCR_PAT)
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
VLLM_VERSION=v0.27.1
VLLM_COMMIT=6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
PATCH="0001-v0271-sm12x-dflash2-nvfp4.patch"
PATCH_SHA256=248adb629444f143975b28013bf29b6c5c65e04789f68aecf5915ea290f0773e
VISION_PATCH="0002-qwen3-next-fused-mrope-vision.patch"
VISION_PATCH_SHA256=82d1a05364dce5151a02b02aa42b9893a05975e45207cdca9c4d87d92c093799
SOURCE_URL=https://github.com/quantamagic/vllm-sm12x-nvfp4-dflash2
GHCR_OWNER="${GHCR_OWNER:-seanyourhighness}"
TAG="v0271-dflash2-$(date +%Y%m%d)"
LOCAL_IMG="vllm-sm12x-nvfp4-dflash2:local-${TAG}"
PUSH_REPO="ghcr.io/${GHCR_OWNER}/vllm-sm12x-nvfp4-dflash2"
PUSH_IMG="${PUSH_REPO}:${TAG}"

# Target: consumer Blackwell (RTX 50-series, SM120, x86_64) by default.
# For an NVIDIA DGX Spark / GB10 (ARM64, SM121), run with SPARK=1.
if [[ "${SPARK:-0}" == "1" ]]; then
  BUILD_PLATFORM="linux/arm64"
  TORCH_ARCH_LIST="12.1"
else
  BUILD_PLATFORM="linux/amd64"
  TORCH_ARCH_LIST="12.0"
fi

WORK="$(mktemp -d /tmp/vllm-dflash2-build.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "==> cloning vllm ${VLLM_VERSION} (${VLLM_COMMIT})"
git clone --filter=blob:none https://github.com/vllm-project/vllm.git "$WORK/vllm"
git -C "$WORK/vllm" checkout --detach "$VLLM_COMMIT"

echo "==> applying sm12x-dflash2-nvfp4 overlay"
actual_patch_sha="$(sha256sum "$SCRIPT_DIR/$PATCH" | cut -d' ' -f1)"
[[ "$actual_patch_sha" == "$PATCH_SHA256" ]] || {
  echo "patch checksum mismatch: expected $PATCH_SHA256, got $actual_patch_sha" >&2
  exit 1
}
git -C "$WORK/vllm" config user.name  "${GIT_USER_NAME:-vllm-sm12x-dflash2-build}"
git -C "$WORK/vllm" config user.email "${GIT_USER_EMAIL:-build@local}"
git -C "$WORK/vllm" apply --index "$SCRIPT_DIR/$PATCH"

if [[ "${VISION_MROPE:-0}" == "1" ]]; then
  echo "==> applying optional fused M-RoPE vision overlay"
  actual_vision_patch_sha="$(sha256sum "$SCRIPT_DIR/$VISION_PATCH" | cut -d' ' -f1)"
  [[ "$actual_vision_patch_sha" == "$VISION_PATCH_SHA256" ]] || {
    echo "vision patch checksum mismatch: expected $VISION_PATCH_SHA256, got $actual_vision_patch_sha" >&2
    exit 1
  }
  git -C "$WORK/vllm" apply --index "$SCRIPT_DIR/$VISION_PATCH"
fi

echo "==> building image (official vLLM Dockerfile, CUDA 13.0.3, SM120/SM121)"
# FlashInfer 0.6.16.post3 + PR #4346 (SM120 NVFP4 paged-prefill) is baked into
# the base image via the official vLLM Dockerfile's flashinfer install step.
# For a fully reproducible build, pin the FlashInfer git commit:
#   --build-arg FLASHINFER_VERSION=0.6.16.post3
#   --build-arg FLASHINFER_GIT_COMMIT=9dc1b2495b40314dec8a8cde8cd7faf5c5206702
PARALLEL_JOBS="${MAX_JOBS:-3}"
NVCC_THREADS="${NVCC_THREADS:-2}"
docker build \
  --platform "$BUILD_PLATFORM" \
  -f "$WORK/vllm/docker/Dockerfile" \
  --build-arg CUDA_VERSION=13.0.3 \
  --build-arg torch_cuda_arch_list="$TORCH_ARCH_LIST" \
  --build-arg max_jobs="$((PARALLEL_JOBS * NVCC_THREADS))" \
  --build-arg nvcc_threads="$NVCC_THREADS" \
  --label "org.opencontainers.image.source=${SOURCE_URL}" \
  --label "org.opencontainers.image.revision=${VLLM_COMMIT}" \
  --label "org.opencontainers.image.version=${TAG}" \
  --label "org.opencontainers.image.licenses=Apache-2.0" \
  -t "$LOCAL_IMG" \
  "$WORK/vllm"

echo "==> smoke test (on a GPU node):"
echo "  ./start.sh"
echo "  # expect boot OK + DFlash2 K7 serving, KV pool ~325k tokens at the 8 GiB pin"

if [[ "${1:-}" == "--push" ]]; then
  : "${GHCR_PAT:?set GHCR_PAT (GitHub token with write:packages)}"
  docker tag "$LOCAL_IMG" "$PUSH_IMG"
  docker tag "$LOCAL_IMG" "${PUSH_REPO}:v0271-dflash2"
  echo "==> logging into ghcr.io"
  echo "$GHCR_PAT" | docker login ghcr.io --username "$GHCR_OWNER" --password-stdin
  echo "==> pushing ${PUSH_IMG} (~9 GB)"
  docker push "$PUSH_IMG"
  docker push "${PUSH_REPO}:v0271-dflash2"
  echo "done: ${PUSH_IMG} and ${PUSH_REPO}:v0271-dflash2"
fi
