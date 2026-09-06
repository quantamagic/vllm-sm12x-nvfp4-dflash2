# Benchmarks — what to expect on a single RTX 5090 (SM120)

All numbers measured on the published `v0.27.1-sm12x-dflash2.3` image,
single RTX 5090 (32 GB, SM120), model `qwen3.8-27b-nvfp4-dflash2`.
GPU decode/prefill from `bench.sh` (3 warm + 5 measured, cache-busted).
Vision from the CPU sidecar (8-core, INT8 default-on).

> Profile note (2026-09-06): the decode/prefill tables below were measured
> on the 8 GiB KV / 262K-context capacity profile. The release has since
> retuned to a 6 GiB KV / 196K-context profile (`v0.27.1-sm12x-dflash2.4`);
> per-lane decode is not expected to change, but the GPU capacity tables
> are `.3`-profile numbers until re-benchmarked.

## TLDR — the numbers at a glance

| Metric | Value | Notes |
|---|---|---|
| **Narrative decode** | **~100 tok/s** (CV 7%) | 1000-token generation, 5 runs |
| **Code decode** | **~195 tok/s** (CV 7%) | 489–800-token generation, 5 runs |
| **Prefill 10K** | **~7,400 tok/s** (CV 2%) | 10,375 prompt tokens |
| **Prefill 90K** | **~3,100 tok/s** (CV 1%) | ~87–90K prompt tokens |
| **TTFT (narrative)** | **~140 ms** | time-to-first-token |
| **TTFT (code)** | **~120 ms** | |
| **c4 aggregate (narrative)** | **~344 tok/s** | 4 concurrent streams |
| **c4 aggregate (code)** | **~587 tok/s** | 4 concurrent streams |
| **Vision 1 MP (1280×960)** | **~7.2 s** end-to-end | 8-core INT8 sidecar |
| **Vision 1080p (1920×1080)** | **~6.8 s** end-to-end | 8-core INT8 sidecar |
| **Vision 1440p (2560×1440)** | **~6.7 s** end-to-end | 8-core INT8 sidecar |

## GPU decode (bench.sh, 3 warm + 5 measured)

| Concurrency | Narrative (tok/s) | Code (tok/s) | Notes |
|---:|---:|---:|---|
| c1 | 97.5 | 172.8 | per-lane |
| c2 | 191.6 (agg) | 329.8 (agg) | ~97/lane |
| c3 | 278.4 (agg) | 446.7 (agg) | ~95/lane |
| c4 | 343.5 (agg) | 586.6 (agg) | ~91/lane |

Per-lane decode is flat (~90–97 narrative, ~170–190 code) across c1–c4;
aggregate scales ~3.5× from c1→c4 (clean linear, no c4 cliff).

## GPU prefill (bench.sh, 3 measured, cache-busted)

| Depth | Prefill (tok/s) | TTFT | CV |
|---:|---:|---:|---:|
| 10K (10,375 tok) | 7,434 | 1,396 ms | 2.3% |
| 90K (~87–90K tok) | 3,137 | 28,344 ms | 1.4% |

> 90K prefill is thermally sensitive: measured ~4,589 tok/s in a cool morning
> run vs ~3,100 tok/s in sustained afternoon runs (GPU downclocks from
> 2,872 MHz to ~2,857 MHz at 58–60 °C idle). The afternoon number is the
> conservative expectation.

## CPU vision sidecar (8-core, INT8 default-on)

| Input | 4-core eager (baseline) | 8-core INT8 (current) | Speedup |
|---|---:|---:|---:|
| 640×480 | 2.88 s | 1.72 s | 1.67× |
| 1280×960 (1 MP) | 16.97 s | 7.18 s | 2.36× |
| 1920×1080 | 10.67 s | 6.84 s | 1.56× |
| 2560×1440 | 10.52 s | 6.68 s | 1.58× |
| Text-only (no image) | 0.37 s | 0.40 s | ~1.0× |

End-to-end 1 MP image: **~16.9 s → ~6.8 s (≈2.5×)** from the original
4-core eager baseline to the current 8-core INT8 configuration.

**Accuracy cost:** INT8 embeddings are ~0.90 cosine-similar to fp32
(0.897–0.906 across test sets). Disable with `SIDECAR_INT8=0`.

## How to reproduce

The numbers above come from the club-3090 `bench.sh` protocol (3 warm + 5
measured, cache-busted, sampler `temp 0.6 / top_p 0.95 / top_k 20 / min_p 0.0`)
and the CPU sidecar timing harness. The release repo ships its own
reproducible bench tools under `bench/`:

```bash
# Reproducible streaming client (decode + prefill, pinned workloads)
python3 bench/publication_bench.py --url http://127.0.0.1:18089

# c8 decode proof (per-stream first-to-last-token rates, >=75 tok/s gate)
BASE_URL=http://127.0.0.1:18089 python3 bench/c8_proof.py

# Full paired publication phase (serving / longbench / ifeval)
./bench/run_publication_phase.sh serving
```

Vision sidecar timing (time an image request end-to-end):

```bash
time curl -s http://127.0.0.1:8016/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-nvfp4-dflash2","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}'
```

> The canonical `bench.sh` (narrative + code + prefill, 3 warm + 5 measured)
> and the `bench-concurrency.py` c1–c4 harness live in the club-3090 repo
> (`~/tmp/club-3090-quality-20260825/scripts/`); the release repo's `bench/`
> tools are the self-contained, in-tree equivalents.

## Thermal variance note

All GPU numbers are thermally sensitive. A cool morning run measures
~15–20% higher than a sustained afternoon run (GPU downclocks from
2,872 MHz to ~2,857 MHz at 58–60 °C idle). The numbers above are from
afternoon runs; expect the higher numbers on a cold GPU.

## Provenance — how to verify these numbers

The TLDR table is a summary; the raw evidence is committed under
`bench/evidence-2026-08-26-5090-bench/`:

- `concurrency-c1-c4.json` — raw c1–c4 concurrent-decode JSON (source of the
  decode table; per-lane + aggregate + TTFT, 5 rounds × 3 warmups).
- `full-bench-summary.md` — the full `bench.sh` summary blocks (narrative,
  code, prefill-10K, prefill-90K) with means/std/CV.

To re-run and confirm:

```bash
# 1. Pull the pinned image + models, start the stack
./start.sh --vision

# 2. Run the in-tree reproducible bench tools (see "How to reproduce")
python3 bench/publication_bench.py --url http://127.0.0.1:18089

# 3. Check the release artifacts match the pinned checksums
sha256sum --check SHA256SUMS

# 4. Run the correctness gates (determinism, NIAH, tools, vision fixture)
./verify.sh --full
./verify.sh --vision
```

The `release-integrity` GitHub workflow (`.github/workflows/integrity.yml`)
runs `./scripts/check-release.sh` on every push, which re-verifies
`SHA256SUMS` and patch applicability — so a tampered or drifted artifact
fails CI.
