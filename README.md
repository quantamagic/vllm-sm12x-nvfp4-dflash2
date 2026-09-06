# vLLM All-NVFP4 DFlash2 for Blackwell (SM120 / SM121) + Optional Vision Sidecar

Community vLLM build for a single Blackwell GPU: ModelOpt **NVFP4 target
weights**, **NVFP4 DFlash2 draft weights**, **NVFP4 KV cache**, and **DFlash2
K=7 block-diffusion speculative decoding** — 196K context, four concurrent
streams, tool calling, and an optional bounded CPU vision sidecar that turns
images into embeddings for the vLLM server.

> This is not an official vLLM or NVIDIA image. It is a pinned community
> overlay on vLLM v0.27.1, validated on an RTX 5090 (SM120, 32 GB). The SM121
> (DGX Spark / GB10) build ships under the same contract; see
> [SM121 notes](#sm121--dgx-spark--gb10).

## What makes this stack different

- **All-NVFP4, end to end.** Target weights, DFlash2 draft weights, and the KV
  cache are all NVFP4. No public 5090 recipe ships 4-of-4 (NVFP4 weights +
  NVFP4 KV + DFlash2 + concurrency); the universal pattern is NVFP4 weights +
  FP8 KV.
- **DFlash2 K7 speculative decoding.** A 5-layer block-diffusion drafter
  (1.92B) proposes 7 tokens per verification step (8 target query tokens),
  with FULL_AND_PIECEWISE CUDA graphs `[8,16,24,32]` and an eager drafter
  that avoids integrated XQA graph interference. Measured ~61% draft
  acceptance and ~2.3x aggregate throughput vs. the legacy c3 profile.
- **Capacity-first profile.** Explicit 6 GiB NVFP4 KV pin → 234,755-token
  pool at 196K max context (196,608), BF16 GDN/SSM state, prefix caching,
  priority scheduling, chunked prefill.
- **Fused multimodal decode.** The `.3` incremental overlay extends the
  fused QK-norm + RoPE + gate Triton kernel to Qwen3.5's three-axis M-RoPE,
  preserving the fast decode path while `--enable-mm-embeds` is active.
- **Optional bounded CPU vision.** `./start.sh --vision` enables the 8-CPU,
  6-GB sidecar (INT8-quantized ViT tower by default); ordinary text serving
  can use the same multimodal-capable server without starting the sidecar.

> **Vision status:** restored as an optional supported profile in
> `v0.27.1-sm12x-dflash2.3`. The SM120 fixture and CUDA kernel gate passed;
> SM121 remains unvalidated.

## Deploy in two commands

Prerequisites: Linux or WSL2, an RTX 5090 (SM120) or DGX Spark/GB10 (SM121),
a working NVIDIA driver, [Docker Engine with the Compose
plugin](https://docs.docker.com/engine/install/), and the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Allow roughly 30 GB of downloads for the ~9 GB runtime image, the 20.6 GB
target checkpoint, and the 1.3 GB draft model.

```bash
git clone https://github.com/quantamagic/vllm-sm12x-nvfp4-dflash2.git
cd vllm-sm12x-nvfp4-dflash2 && ./start.sh
```

To start the bounded CPU vision sidecar as well:

```bash
./start.sh --vision
```

`start.sh` checks the GPU/SM, VRAM, Docker/Compose, disk space, and ports;
pulls the pinned runtime; downloads the exact pinned target and draft
checkpoints into Docker named volumes; starts vLLM; waits for health; and
sends a real chat completion (deterministic canary: `19×23 → 437`).

On hosts with more than one GPU, set `GPU_DEVICE` in `.env` to the
Blackwell card's index or UUID (default `0`). `start.sh` probes only that
device and the compose stack exposes only that device to the container
(`NVIDIA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), so an older
second card cannot interfere with device selection or enumeration.

Endpoints after startup:

- OpenAI-compatible vLLM API: `http://127.0.0.1:18089/v1`
- Vision-capable proxy (with `--vision`): `http://127.0.0.1:8016/v1`
- Served model: `qwen3.8-27b-nvfp4-dflash2`

First startup is dominated by the two downloads and CUDA/FlashInfer warmup.
Subsequent starts reuse the Docker image and the named model caches.

## Exact pinned stack

| Component | Pinned artifact |
|---|---|
| Runtime | `ghcr.io/seanyourhighness/vllm-sm12x-nvfp4-dflash2` (digest in `.env.example`; validated local tag `local-v0271-dflash2-capacity-k7-20260824`, image id `sha256:06f0c21d…`) |
| Target model + revision | [`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090@69274a0`](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/tree/69274a0d8dff5dd35bcee8290612f71e03b6e981) |
| Draft model | [`YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4`](https://huggingface.co/YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4) (`model.safetensors` sha256 `db19f849…`) |
| vLLM base | [v0.27.1 commit `6e448d0ea`](https://github.com/vllm-project/vllm/commit/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac) |
| FlashInfer | 0.6.16.post3 (git `9dc1b24`), with [PR #4346](https://github.com/flashinfer-ai/flashinfer/pull/4346) SM120 NVFP4 paged-prefill backport |
| Overlay | [`0001-v0271-sm12x-dflash2-nvfp4.patch`](0001-v0271-sm12x-dflash2-nvfp4.patch) (51 files, Python-only; `sha256:248adb62…`) |
| Vision overlay | [`0002-qwen3-next-fused-mrope-vision.patch`](0002-qwen3-next-fused-mrope-vision.patch) (2 production Python files + targeted CUDA test; minimal layer recorded as 12,600 bytes) |
| Minimal candidate image | [`Dockerfile.vision-mrope`](Dockerfile.vision-mrope) over the unchanged `.2` base |
| Chat template | [`chat-template.jinja`](chat-template.jinja) (`sha256:4dd9e48a…`, `qwen3.8-froggeric-v22.4`) |
| Checksums | [`SHA256SUMS`](SHA256SUMS) — template `4dd9e48a…`, sidecar `f370f541…` |

The image and models are pinned by immutable digests/revisions, not floating
tags. Compose passes the pinned model revision to vLLM and mounts the
shipped release template with `--chat-template`; this intentionally
overrides the different template bundled with the model. Model weights are
not redistributed in the runtime image.

## Common operations

```bash
./status.sh                 # containers, health, GPU, and model-cache status
./verify.sh --smoke         # health + model routing + deterministic canary
./verify.sh --full          # + long-decode determinism, NIAH, tool calling
./verify.sh --vision       # exact two-image fixture + vision concurrency gate
./stop.sh                   # stop + remove containers (volumes kept)
./stop.sh --purge-cache     # also remove the named model/vllm/draft caches
```

## Validated runtime profile (the "everything we run" defaults)

| Knob | Value |
|---|---|
| Speculative config | `{"method":"dflash","model":"/models/draft","num_speculative_tokens":7,"kv_cache_dtype":"nvfp4"}` |
| Target KV cache | NVFP4, explicit 6 GiB pin (`6442450944` bytes) → 234,755-token pool |
| GDN/SSM state | `bfloat16` |
| Max model len | 196,608 |
| Max concurrent seqs | 4 (capacity-first; 1.19× max concurrency at full-length 196K requests) |
| Max batched tokens | 2,048 (engine caps scheduled tokens at 2,024 for K7 draft slots) |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes `[8,16,24,32]` (K7 → 8-token verifier queries) |
| Drafter | forced eager (`VLLM_DFLASH_FORCE_EAGER=1`) — avoids integrated XQA graph interference |
| Target XQA | dedicated CUDA stream (`VLLM_XQA_DEDICATED_STREAM=1`) |
| FlashInfer autotune | enabled (loaded from the persisted autotune cache at boot) |
| Scheduling | priority + prefix caching + chunked prefill, long-prefill threshold 2048 |
| Chat template | `qwen3.8-froggeric-v22.4` — thinking on by default (medium effort); supports inline `<|think_on/off|>` / `<|think_low|>/<|think_medium|>/<|think_xhigh|>` markers, `preserve_reasoning`, and `auto_disable_thinking_with_tools` |
| Tooling | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| Serving mode | `--enable-mm-embeds` + zero image/video limits; fused M-RoPE kernel |
| Triton JIT cache | `/home/vllm/.cache/vllm/triton` (persisted in the named vLLM cache volume) |

Measured on the RTX 5090 (SM120), published `v0.27.1-sm12x-dflash2.3`
image (3 warm + 5 measured, cache-busted):

| Concurrency | Narrative (tok/s) | Code (tok/s) |
|---:|---:|---:|
| c1 | ~98 | ~173 |
| c2 | ~192 (agg) | ~330 (agg) |
| c3 | ~278 (agg) | ~447 (agg) |
| c4 | ~344 (agg) | ~587 (agg) |

Per-lane decode stays flat (~90–97 narrative, ~169–189 code) across c1–c4;
aggregate scales ~3.5× from c1 → c4 with ~61% draft acceptance at K7, zero
restarts, zero OOM. These numbers were measured on the earlier 8 GiB /
262K-capacity profile; the current 6 GiB / 196K profile has not been
re-benchmarked (see the [changelog](CHANGELOG.md) 2026-09-06 entry). See
[BENCHMARKS.md](BENCHMARKS.md) for the TLDR of expected 5090
decode/prefill/vision numbers and [EVIDENCE.md](EVIDENCE.md) for the full
measurement record.

## Why the M-RoPE overlay is required

The CPU sidecar itself was idle during text requests and did not consume
meaningful CPU. The slowdown came from keeping the GPU server in multimodal /
embedding-capable mode. In the `.2` source, Qwen3.5's fused QK-norm + RoPE +
gate decoder path did not support three-axis M-RoPE, so the multimodal path
fell back to the slower eager sequence. `--limit-mm-per-prompt
'{"image":0,"video":0}'` prunes the vision tower but does not fix that
kernel-selection gap.

The `.3` patch carries the narrow release backport: T/H/W M-RoPE section
selection (contiguous and interleaved) is handled inside the existing Triton
kernel, and Qwen3Next passes the full three-axis positions. The kernel JIT
compiles on first use; no CUDA/native rebuild is required for the minimal
Dockerfile overlay.

On the same RTX 5090 release image and benchmark prompts, the old
embedding-capable path measured narrative 61.6 and code 105.7 tok/s, versus
language-only 116.2 and 202.4 tok/s. The `.3` candidate restores the
multimodal configuration while preserving the matched c1-c4 decode profile;
see [EVIDENCE.md](EVIDENCE.md) for the exact measurements.

Two upstream efforts cover the same general kernel area and are treated as
duplicate work rather than a new upstream PR: [vLLM #49744](https://github.com/vllm-project/vllm/pull/49744)
and [vLLM #43056](https://github.com/vllm-project/vllm/pull/43056). This release
uses a narrower pinned-vLLM backport for the DFlash2 image.

### Upgrade from `.2`

```bash
git pull --ff-only
docker compose down --remove-orphans
./start.sh
```

The `.2` base image and model artifacts are unchanged. Build the small
candidate layer with `Dockerfile.vision-mrope`, set `IMAGE` to that candidate,
then use `./start.sh` or `./start.sh --vision`. The first multimodal request
will warm the new Triton kernel variants.

Run `./verify.sh --vision` for the exact two-image fixture and concurrency
gate after the sidecar is healthy.

## Verify this release

Every artifact is pinned by immutable digest/revision, not floating tag, and
the `release-integrity` workflow re-checks them on every push:

```bash
sha256sum --check SHA256SUMS   # patches, Dockerfile, template, sidecar, bench gates
./verify.sh --full             # determinism, NIAH, tools
./verify.sh --vision           # exact two-image fixture + vision concurrency gate
```

Expected 5090 decode/prefill/vision numbers, with the raw evidence and a
reproduce path, are in [BENCHMARKS.md](BENCHMARKS.md); the full measurement
record is in [EVIDENCE.md](EVIDENCE.md).

## SM121 (DGX Spark / GB10)

The same source + patch contract builds for `linux/arm64` with
`torch_cuda_arch_list=12.1`:

```bash
SPARK=1 ./build.sh          # native build on the Spark (fast path)
```

The SM121 build is **not yet natively validated** (no SM121 hardware was
available at release time); the SM120 artifact and `.3` vision measurements
are the validated release evidence.
The multi-arch tag will not be promoted until the SM121 build passes the
full correctness matrix (greedy determinism, NIAH, tools, c4 soak)
natively on a GB10.

## Rollback / coexistence

This release coexists with the MTP release (`vllm-sm120-nvfp4-mtp`) as a
blue/green alternative: separate repo, image, Compose project
(`qwen38-dflash2`), container names, ports (18089/8016 vs. 18079/8006), and
model cache volumes. Only one 27B GPU server runs at a time on a single
32 GiB card. Rollback: `./stop.sh`, then start the unchanged pinned MTP
project.
