#!/usr/bin/env bash
# start.sh — one-shot launch of the all-NVFP4 DFlash2 release.
#
#   ./start.sh              # multimodal-capable server, no CPU sidecar
#   ./start.sh --vision     # server + bounded CPU vision sidecar
#
# Checks GPU/SM, VRAM, Docker/Compose, disk, and ports; pulls the pinned
# runtime; downloads the pinned target + draft checkpoints into named
# Docker volumes; starts vLLM; waits for health; and sends a real chat
# completion.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f .env ]] || cp .env.example .env
set -a
# shellcheck disable=SC1091
source .env
set +a

# These values are interpolated into JSON / integer CLI arguments in
# compose.yaml. Reject placeholder or malformed values before pulling images or
# starting a restart loop that can only surface an opaque argparse error.
for numeric_setting in NUM_SPECULATIVE_TOKENS MAX_NUM_BATCHED_TOKENS; do
  numeric_value="${!numeric_setting:-}"
  if [[ ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $numeric_setting must be a positive integer in .env; found '$numeric_value'" >&2
    exit 2
  fi
done

for cmd in docker nvidia-smi curl python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: required command not found: $cmd" >&2; exit 1; }
done
docker compose version >/dev/null 2>&1 || { echo "error: Docker Compose plugin is required" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "error: Docker daemon is not reachable by this user" >&2; exit 1; }

# Probe a single device by index/UUID (default 0). No pipe: under
# `set -euo pipefail`, `nvidia-smi ... | head -n1` dies with SIGPIPE (141)
# on multi-GPU hosts when nvidia-smi is still writing after head exits.
gpu_line="$(nvidia-smi -i "${GPU_DEVICE:-0}" --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits)"
[[ -n "$gpu_line" ]] || {
  echo "error: nvidia-smi returned no data for GPU ${GPU_DEVICE:-0}; set GPU_DEVICE in .env to your Blackwell card's index or UUID" >&2
  exit 1
}
gpu_name="${gpu_line%%,*}"
gpu_rest="${gpu_line#*, }"
gpu_mem="${gpu_rest%%,*}"
gpu_cap="${gpu_line##*, }"
case "$gpu_cap" in
  12.0|12.1) : ;;
  *) echo "error: requires SM120 (RTX 5090) or SM121 (DGX Spark/GB10); found $gpu_name (compute $gpu_cap) on GPU ${GPU_DEVICE:-0}" >&2; exit 1 ;;
esac
(( gpu_mem >= 30000 )) || { echo "error: canonical profile requires ~32 GB VRAM; found ${gpu_mem} MiB" >&2; exit 1; }

free_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
(( free_kb >= 45 * 1024 * 1024 )) || echo "warning: less than 45 GB free; downloads may exhaust disk" >&2

files=(-f compose.yaml)
mode="server"
if [[ "${1:-}" == "--vision" ]]; then
  files+=("--profile" "vision")
  mode="server + CPU vision sidecar"
elif [[ -n "${1:-}" ]]; then
  echo "usage: ./start.sh [--vision]" >&2
  exit 2
fi

echo "GPU: $gpu_name, ${gpu_mem} MiB, compute $gpu_cap"

# Registry images (ghcr.io/...) are pulled; bare local tags are used as-is.
case "$IMAGE" in
  local:*|*:local-*)
    echo "Using local image $IMAGE (build first: ./build.sh)"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
      echo "error: local image $IMAGE not found; run ./build.sh first" >&2
      exit 1
    }
    ;;
  *)
    echo "Pulling pinned runtime $IMAGE ..."
    docker compose "${files[@]}" pull
    ;;
esac

# Docker-GPU check on the exact device the container will see.
docker run --rm \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICE:-0}" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --gpus all \
  --entrypoint nvidia-smi "$IMAGE" -L >/dev/null || {
  echo "error: Docker cannot access the GPU; configure NVIDIA Container Toolkit" >&2
  exit 1
}

docker compose "${files[@]}" up -d --remove-orphans

# Stream the GPU server's logs live so first-boot download, CUDA warmup, and
# engine progress are visible while we wait for health. The follower is killed
# on exit (success, timeout, or interrupt).
docker compose "${files[@]}" logs --no-color --no-log-prefix -f server 2>&1 &
logs_follower=$!
cleanup() {
  kill "$logs_follower" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + START_TIMEOUT_SECONDS))
echo "Waiting for vLLM (first boot downloads the pinned target checkpoint, ~20.6 GB)..."
echo "Streaming server logs below (Ctrl-C to stop waiting):"
while (( SECONDS < deadline )); do
  if curl -fsS "http://${BIND_ADDRESS}:${VLLM_PORT}/health" >/dev/null 2>&1; then
    if [[ "$mode" == *vision* ]]; then
      if curl -fsS "http://${BIND_ADDRESS}:${VISION_PORT}/health" >/dev/null 2>&1; then
        echo "Services are healthy."
        ./verify.sh --smoke
        echo "OpenAI API:  http://${BIND_ADDRESS}:${VLLM_PORT}/v1"
        echo "Vision proxy: http://${BIND_ADDRESS}:${VISION_PORT}/v1"
        exit 0
      fi
    else
      echo "Server is healthy."
      ./verify.sh --smoke
      echo "OpenAI API: http://${BIND_ADDRESS}:${VLLM_PORT}/v1"
      exit 0
    fi
  fi
  if (( SECONDS % 30 < 10 )); then
    docker compose "${files[@]}" ps --format 'table {{.Service}}\t{{.Status}}' || true
  fi
  sleep 10
done

echo "error: startup timed out after ${START_TIMEOUT_SECONDS}s" >&2
docker compose "${files[@]}" ps >&2 || true
docker compose "${files[@]}" logs --tail=120 server >&2 || true
exit 1
