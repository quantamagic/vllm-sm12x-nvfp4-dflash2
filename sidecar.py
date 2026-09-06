"""CPU vision sidecar for vLLM --enable-mm-embeds.

Default proxy: :8016 -> vLLM :18089. Rewrites image_url content parts into
image_embeds parts computed by the checkpoint's vision tower on CPU.

Resilience layer (all transparent to clients, each disableable via env):
  AUTOCONTINUE   finish_reason=length (output truncation) -> resume the
                 generation from where it stopped and stitch the parts.
  LOOPBREAK      finished response is highly repetitive (model stuck in a
                 loop) -> regenerate once with an injected "change strategy"
                 instruction, returning only the clean prefix + the new try.
  AUTOCOMPACT    upstream rejects the prompt as too long (400 context
                 length) -> summarize the middle of the conversation with
                 the model itself, keep system + recent turns verbatim,
                 retry (reduces further if still too long).

Env:
  SIDECAR_MODEL           local path or Hugging Face model ID
  SIDECAR_MODEL_REVISION  immutable Hugging Face revision
  SIDECAR_UPSTREAM   vLLM base URL (default http://127.0.0.1:18089)
  SIDECAR_AUTOCONTINUE   1/0 (default 1) — resume truncated outputs
  SIDECAR_AUTOCONTINUE_MAX  max resume round-trips (default 3)
  SIDECAR_LOOPBREAK      1/0 (default 1) — break repetition loops
  SIDECAR_AUTOCOMPACT    1/0 (default 1) — summarize over-long histories
  SIDECAR_CTX_FLOOR      tokens trimmed to on compaction (default 24576)
  SIDECAR_AUTO_SUMMARIZE  1/0 (default 0) — model-summarize compacted history
                         (slower: adds a full generation pass; mechanical
                         note is used instead when 0)

Run:  python sidecar.py 8016
"""
import base64
import asyncio
import io
import json
import os
import re
import sys

import httpx
import torch
import torch.nn as nn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open
from torch.ao.quantization import quantize_dynamic
from transformers import AutoConfig, AutoProcessor

MODEL = os.environ.get(
    "SIDECAR_MODEL", "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
)
MODEL_REVISION = os.environ.get(
    "SIDECAR_MODEL_REVISION", "69274a0d8dff5dd35bcee8290612f71e03b6e981"
)
UPSTREAM = os.environ.get("SIDECAR_UPSTREAM", "http://127.0.0.1:18089")
MAX_PIXELS = 1280 * 28 * 28  # cap ~1.28 MP like the Qwen3-VL default
MAX_IMAGE_BYTES = 25 * 1024 * 1024
TORCH_NUM_THREADS = max(1, int(os.environ.get("TORCH_NUM_THREADS", "4")))
TORCH_NUM_INTEROP_THREADS = max(
    1, int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1"))
)
# Opt-in INT8 dynamic quantization of the vision tower's Linear layers
# (INT8 weights, fp32 activations). ~1.5-2x faster CPU encode; embedding
# cosine similarity vs fp32 is ~0.90. Default on (set SIDECAR_INT8=0 to
# fall back to fp32 eager).
SIDECAR_INT8 = os.environ.get("SIDECAR_INT8", "1") == "1"

# Resilience switches (client-visible behavior identical when all on).
AUTOCONTINUE = os.environ.get("SIDECAR_AUTOCONTINUE", "1") == "1"
AUTOCONTINUE_MAX = max(1, int(os.environ.get("SIDECAR_AUTOCONTINUE_MAX", "3")))
LOOPBREAK = os.environ.get("SIDECAR_LOOPBREAK", "1") == "1"
AUTOCOMPACT = os.environ.get("SIDECAR_AUTOCOMPACT", "1") == "1"
CTX_FLOOR = max(4096, int(os.environ.get("SIDECAR_CTX_FLOOR", "24576")))
AUTO_SUMMARIZE = os.environ.get("SIDECAR_AUTO_SUMMARIZE", "0") == "1"

torch.set_num_threads(TORCH_NUM_THREADS)
torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

app = FastAPI()

# Allow browser clients on any origin (localhost dev UIs, LAN tools, etc.).
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class VisionEncoder:
    def __init__(self) -> None:
        model_dir = MODEL if os.path.isdir(MODEL) else snapshot_download(
            repo_id=MODEL,
            revision=MODEL_REVISION,
            token=os.environ.get("HF_TOKEN") or None,
        )
        self.model_dir = model_dir
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        from transformers import AutoModel

        vision_cls = AutoModel._model_mapping[type(cfg.vision_config)]
        self.model = vision_cls(cfg.vision_config)
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as index_file:
            wm = json.load(index_file)["weight_map"]
        state = {}
        for shard in sorted(set(wm.values())):
            with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("model.visual."):
                        state[k[len("model.visual."):]] = f.get_tensor(k)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vision load mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
        self.model.eval()
        if SIDECAR_INT8:
            self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            print("[sidecar] vision tower INT8-quantized (dynamic, Linear layers)")
        print(f"[sidecar] vision tower loaded on CPU: {sum(v.numel() for v in self.model.parameters())} params")

    @torch.no_grad()
    def encode(self, image_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if img.width * img.height > MAX_PIXELS:
            scale = (MAX_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        inputs = self.processor(images=[img], return_tensors="pt")
        out = self.model(
            hidden_states=inputs["pixel_values"],
            grid_thw=inputs["image_grid_thw"],
        )
        embeds = out.pooler_output.float().to(torch.bfloat16).contiguous()
        grid = inputs["image_grid_thw"][0].to(torch.int64).contiguous()  # [3]
        return embeds, grid


encoder: VisionEncoder | None = None
encode_semaphore = asyncio.Semaphore(1)
client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0))


def _load_image(url: str) -> bytes:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        data = base64.b64decode(b64, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 25 MiB limit")
        return data
    if os.path.exists(url):
        if os.path.getsize(url) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds 25 MiB limit")
        with open(url, "rb") as image_file:
            return image_file.read()
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    data = response.content
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 25 MiB limit")
    return data


def _b64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode()


async def _rewrite(messages: list) -> list:
    out = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part["image_url"]
                if isinstance(url, dict):
                    url = url.get("url", "")
                data = await asyncio.to_thread(_load_image, url)
                # PyTorch CPU inference uses its own native thread pool. Keep only
                # one encode in flight so concurrent image requests cannot multiply
                # that pool and starve WSLg, Docker, or the Hermes gateway.
                async with encode_semaphore:
                    embeds, grid = await asyncio.to_thread(encoder.encode, data)
                parts.append({
                    "type": "image_embeds",
                    "image_embeds": {
                        "image_embeds": _b64(embeds),
                        "image_grid_thw": _b64(grid),
                    },
                })
            else:
                parts.append(part)
        new_msg = dict(msg)
        new_msg["content"] = parts
        out.append(new_msg)
    return out


# ---------------------------------------------------------------------------
# Resilience layer: auto-continue / loop-break / auto-compact
# ---------------------------------------------------------------------------

def _msg_text(msg: dict) -> str:
    """Best-effort plain text of a message (for estimation/summarization)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("type") in ("image_url", "image_embeds"):
                parts.append("[image]")
        return "\n".join(parts)
    return ""


def _est_tokens(messages: list) -> int:
    """Cheap token estimate (~3.5 chars/token for mixed zh/en) + per-msg overhead."""
    chars = sum(len(_msg_text(m)) for m in messages)
    return int(chars / 3.5) + 6 * len(messages)


async def _upstream_chat(body: dict) -> httpx.Response:
    req = client.build_request("POST", f"{UPSTREAM}/v1/chat/completions", json=body)
    return await client.send(req)


async def _one_completion(body: dict) -> dict:
    """Single non-stream upstream call; raises on upstream error status."""
    resp = await _upstream_chat(body)
    if resp.status_code != 200:
        raise UpstreamError(resp.status_code, resp.text)
    if not resp.content.strip():
        raise UpstreamError(502, "upstream returned an empty body")
    try:
        return json.loads(resp.content)
    except json.JSONDecodeError as e:
        raise UpstreamError(502, f"upstream returned invalid JSON: {e}") from None


class UpstreamError(Exception):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(f"upstream {status}: {text[:200]}")
        self.status = status
        self.text = text





def _flush_held(held: str, tail: str) -> str:
    """Trim repeated overlap with the streamed tail, then boundary-join."""
    if not held:
        return held
    for k in (min(120, len(tail)), 60, 30, 12, 4, 2, 1):
        if k and held.startswith(tail[-k:]):
            return _join_continuation(tail, held[k:])[len(tail):]
    return _join_continuation(tail, held)[len(tail):] if tail else held

def _join_continuation(left: str, right: str) -> str:
    """Stitch a continuation onto a truncated left part.

    If the model repeated the tail, that was already trimmed. What remains:
    when the truncation cut mid-token, the newline that would have started
    the next line is often swallowed. If the boundary does not look like a
    mid-word split (both sides alphanumeric) AND the left part's last line
    looks like a finished list/line item, rejoin with a newline.
    """
    if not left or not right:
        return left + right
    if left[-1].isspace() or right[0].isspace():
        return left + right
    last_line = left.rstrip("\n").rsplit("\n", 1)[-1]
    # Mid-word split: letters/digits both sides AND last line long or ending
    # with punctuation → it's genuinely mid-sentence; join directly.
    if last_line and last_line[-1] in ".:;，。；:" :
        return left + right
    stripped = right.lstrip()
    # Both sides alphanumeric and the right side starts a new list/line marker
    # (digit+. / - / * / #) → newline belongs between.
    import re as _re
    if _re.match(r"^(\d{1,3}[.)]\s|[-*+] |#{1,6} |\d+\s)", stripped) and (
        left[-1].isalnum()
    ):
        return left + "\n" + right
    return left + right

CONTINUE_INSTRUCTION = (
    "Your previous reply was cut off by the output length limit. "
    "Continue EXACTLY where you stopped — do not repeat any content you "
    "already produced, do not re-introduce, re-greet, or re-summarize. "
    "Resume mid-sentence if that is where you stopped."
)

LOOPBREAK_INSTRUCTION = (
    "⚠️ SYSTEM: Your previous answer fell into a repetition loop — the same "
    "content kept recurring. That attempt has been discarded. Produce your "
    "answer again, but you MUST take a fundamentally different approach: "
    "re-read the original request, question your last assumption, and do "
    "NOT restart from the same phrasing or the same plan."
)


def _content_str(msg: dict) -> str:
    c = msg.get("content")
    return c if isinstance(c, str) else _msg_text(msg)


# --- auto-continue ----------------------------------------------------------

def _is_reasoning_flushed(msg: dict) -> bool:
    return not msg.get("reasoning_content")


async def autocontinue(body: dict, first: dict) -> dict:
    """Stitch truncated completions (finish_reason=length) back together.

    The chat template always closes assistant turns (<|im_end|>), so we
    cannot ask the model to continue inside the same assistant turn.
    Instead we send the partial text back in a user instruction asking for
    the verbatim continuation, then stitch: the overlap where the model
    repeats the tail of the partial is detected and trimmed.
    """
    choice = first["choices"][0]
    full_content = choice["message"].get("content") or ""
    full_reasoning = choice["message"].get("reasoning_content") or ""
    rounds = 0
    while choice.get("finish_reason") == "length" and rounds < AUTOCONTINUE_MAX:
        rounds += 1
        tail = full_content[-400:]
        resume = dict(body)
        msgs = [dict(m) for m in body.get("messages", [])]
        msgs.append({"role": "user", "content":
            "Your reply above was cut off by the output length limit. Here is "
            "how it currently ends:\n<partial_tail>\n" + tail +
            "\n</partial_tail>\nOutput ONLY the missing continuation that "
            "follows <partial_tail> — no repetition of any earlier content, "
            "no commentary, no <partial_tail> tags. Resume mid-sentence if "
            "that is where it stopped."})
        resume["messages"] = msgs
        kwargs = resume.get("chat_template_kwargs") or {}
        resume["chat_template_kwargs"] = {**kwargs, "enable_thinking": False}
        nxt = await _one_completion(resume)
        nchoice = nxt["choices"][0]
        nmsg = nchoice["message"]
        ncontent = (nmsg.get("content") or "").strip()
        if ncontent:
            ncontent = _flush_held(ncontent.strip(), tail)
            full_content = _join_continuation(full_content, ncontent)
        if nmsg.get("reasoning_content"):
            full_reasoning += nmsg["reasoning_content"]
        choice = nchoice
        if nchoice.get("finish_reason") == "stop":
            break
    out = json.loads(json.dumps(first))  # deep copy
    out["choices"][0]["message"]["content"] = full_content
    if full_reasoning:
        out["choices"][0]["message"]["reasoning_content"] = full_reasoning
    if rounds:
        out["choices"][0]["finish_reason"] = "stop"
        out.setdefault("sidecar", {})["autocontinue_rounds"] = rounds
    return out


# --- loop-break ---------------------------------------------------------------

_REPEAT_PAT = re.compile(r"(\b.{8,60}\b)(?:[\s\S]{0,80}?\1){3,}")


def _repetition_prefix(text: str) -> int | None:
    """Return char index where runaway repetition starts, else None.

    Heuristics: (a) same non-trivial line repeated 4+ times in a row;
    (b) an 8-60 char span recurring 4+ times within a short window.
    """
    lines = text.split("\n")
    run_start = 0
    run_len = 1
    for i in range(1, len(lines)):
        prev, cur = lines[i - 1].strip(), lines[i].strip()
        if cur and prev and (cur == prev or (len(cur) > 12 and cur in prev) or (len(prev) > 12 and prev in cur)):
            if run_len == 1:
                run_start = i - 1  # loop begins at the first repeated pair's start
            run_len += 1
            if run_len >= 4:
                return len("\n".join(lines[: run_start]))
        else:
            run_len = 1
    m = _REPEAT_PAT.search(text)
    if m:
        # Only treat as a loop if the repetition occupies the tail substantially.
        if m.end() > 0.6 * len(text) and (m.end() - m.start()) > 120:
            return m.start()
    return None


async def loopbreak(body: dict, first: dict) -> dict:
    """Detect a stuck/repetitive completion and regenerate past the loop."""
    choice = first["choices"][0]
    if choice.get("finish_reason") != "stop":
        return first
    content = choice["message"].get("content") or ""
    if not content or len(content) < 200:
        return first
    cut = _repetition_prefix(content)
    if cut is None:
        return first
    clean = content[:cut].rstrip()
    retry = dict(body)
    msgs = [dict(m) for m in body.get("messages", [])]
    msgs.append({"role": "assistant", "content": clean})
    msgs.append({"role": "user", "content": LOOPBREAK_INSTRUCTION})
    retry["messages"] = msgs
    try:
        second = await _one_completion(retry)
    except UpstreamError:
        return first  # better a loop than a hard failure
    schoice = second["choices"][0]
    scontent = schoice["message"].get("content") or ""
    scut = _repetition_prefix(scontent)
    if scut is not None:
        # Second attempt also looped: return the cleanest result we have.
        scontent = scontent[:scut].rstrip()
    out = json.loads(json.dumps(first))
    out["choices"][0]["message"]["content"] = (clean + "\n\n" + scontent).strip()
    out["choices"][0]["finish_reason"] = schoice.get("finish_reason", "stop")
    out.setdefault("sidecar", {})["loopbreak"] = True
    return out


# --- auto-compact -----------------------------------------------------------

COMPACT_PROMPT = (
    "Summarize the following conversation excerpt for use as running "
    "context. Preserve: the user's goals and constraints, decisions made, "
    "tool results with concrete values (paths, ids, numbers, errors), and "
    "any unfinished threads. Be dense and factual; no preamble. "
    "Write the summary in the same language as the excerpt."
)


def _compact_split(messages: list):
    """Split into (system_head, middle, tail_keep). Tail keeps the last
    user/tool exchange chain so the model can act immediately."""
    head = []
    i = 0
    while i < len(messages) and messages[i].get("role") in ("system", "developer"):
        head.append(messages[i])
        i += 1
    rest = messages[i:]
    if len(rest) <= 4:
        return messages, None  # nothing safe to compress
    # Keep the trailing chain: last user msg and everything after it.
    tail_start = len(rest) - 1
    while tail_start > 0 and rest[tail_start - 1].get("role") == "tool":
        tail_start -= 1
    if rest[tail_start].get("role") == "user" and tail_start > 0:
        tail_start -= 1
    tail_start = max(tail_start, 2)
    middle = rest[:tail_start]
    tail = rest[tail_start:]
    if len(middle) < 2:
        return messages, None
    return head + tail, middle


async def _summarize_middle(body: dict, middle: list) -> str:
    text = "\n\n".join(
        f"[{m.get('role')}] {_content_str(m)[:4000]}" for m in middle
    )
    sb = {
        "model": body.get("model"),
        "messages": [
            {"role": "user", "content": COMPACT_PROMPT + "\n\n" + text[-96000:]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.2,
        "max_tokens": 1536,
    }
    resp = await _one_completion(sb)
    return resp["choices"][0]["message"].get("content") or ""


async def autocompact(body: dict, status: int, err_text: str) -> dict | None:
    """Retry an over-context request with a summarized middle. Returns the
    successful completion payload or None if compaction cannot help."""
    if not AUTOCOMPACT or status != 400:
        return None
    low = err_text.lower()
    if "maximum context length" not in low and "context length" not in low and "too long" not in low:
        return None
    for _ in range(2):  # at most two compaction rounds
        msgs = body.get("messages", [])
        if _est_tokens(msgs) <= CTX_FLOOR // 2:
            return None
        kept, middle = _compact_split(msgs)
        if middle is None:
            return None
        # Mechanical summary first: instant, no model round-trip, so the
        # client never times out waiting for compaction to finish.
        n_chars = sum(len(_content_str(m)) for m in middle)
        summary = (
            f"[{len(middle)} earlier messages (~{n_chars // 4} tokens) were "
            f"dropped to fit the context window. They predate the current "
            f"task; the recent messages below contain everything relevant.]"
        )
        if AUTO_SUMMARIZE:
            # Try a real model summary, but any failure (timeout, client
            # gone, empty body) falls back to the mechanical note above.
            try:
                summary = await _summarize_middle(body, middle)
            except Exception:
                pass
        compact_block = (
            "[CONVERSATION SUMMARY — earlier turns compressed to save context]\n"
            + summary
            + "\n[END SUMMARY — recent messages follow verbatim]"
        )
        body = dict(body)
        summary_msg = {"role": "user", "content": compact_block}
        body["messages"] = [kept[0], summary_msg] + kept[1:] if kept else [summary_msg]
        try:
            return await _one_completion(body)
        except UpstreamError as e:
            if e.status != 400:
                return None
            continue
    return None


async def _resilient_completion(body: dict) -> Response:
    """Non-stream path: compact → complete → continue → loop-break."""
    try:
        first = await _one_completion(body)
    except UpstreamError as e:
        compacted = await autocompact(body, e.status, e.text)
        if compacted is None:
            return Response(
                content=json.dumps({"error": {"message": e.text[:500], "type": "upstream_error",
                                             "code": f"upstream_{e.status}"}}).encode(),
                status_code=e.status,
                media_type="application/json",
            )
        first = compacted
    if AUTOCONTINUE:
        first = await autocontinue(body, first)
    if LOOPBREAK:
        first = await loopbreak(body, first)
    return Response(
        content=json.dumps(first).encode(),
        status_code=200,
        media_type="application/json",
    )


async def _resilient_stream(body: dict) -> Response:
    """Streaming path with seamless truncation-resume.

    Loop-break and auto-compact are non-stream only (they need the full
    text / a 400 status before deciding); streams pass through untouched
    apart from resume stitching.
    """
    async def gen():
        attempt_body = dict(body)
        resume_hold = None
        streamed_tail = ""
        done_sent = False
        for round_no in range(AUTOCONTINUE_MAX + 1 if AUTOCONTINUE else 1):
            req = client.build_request("POST", f"{UPSTREAM}/v1/chat/completions", json=attempt_body)
            try:
                resp = await client.send(req, stream=True)
            except httpx.HTTPError:
                yield ('data: {"error":{"message":"upstream connection failed",'
                       '"type":"server_error","code":"upstream_connect"}}\n\n')
                yield "data: [DONE]\n\n"
                return
            if resp.status_code != 200:
                err_body = (await resp.aread()).decode(errors="replace")
                await resp.aclose()
                compacted = await autocompact(attempt_body, resp.status_code, err_body)
                if compacted is not None:
                    # We already have the full result: emit it as one chunk.
                    import time as _t
                    cid = compacted.get("id", "sidecar-compact")
                    created = compacted.get("created", int(_t.time()))
                    payload = {
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": attempt_body.get("model"),
                        "choices": [{"index": 0, "delta": {
                            "content": compacted["choices"][0]["message"].get("content") or ""
                        }, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    fin = {"choices": [{"index": 0, "delta": {},
                                        "finish_reason": compacted["choices"][0].get("finish_reason", "stop")}]}
                    yield f"data: {json.dumps(fin)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                try:
                    err_obj = json.loads(err_body)
                except Exception:
                    err_obj = {"error": {"message": err_body[:300], "type": "upstream_error"}}
                yield "data: " + json.dumps(err_obj).replace("\n", " ") + "\n\n"
                if not done_sent:
                    done_sent = True
                    yield "data: [DONE]\n\n"
                return
            finish = None
            buf_content = []
            nlines = 0
            if os.environ.get("SIDECAR_DEBUG"):
                print(f"[sidecar][debug] round={round_no} status={resp.status_code}", flush=True)
            async for line in resp.aiter_lines():
                nlines += 1
                if os.environ.get("SIDECAR_DEBUG") and nlines <= 8:
                    print(f"[sidecar][debug] line{nlines}: {line[:110]!r}", flush=True)
                if not line.startswith("data: "):
                    continue  # drop blanks/comments; SSE events get own \n\n
                data = line[6:]
                data = line[6:]
                if data.strip() == "[DONE]":
                    if resume_hold is not None and resume_hold:
                        held = "".join(resume_hold)
                        resume_hold = None
                        held = _flush_held(held, streamed_tail)
                        if held:
                            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': held}, 'finish_reason': None}]})}\n\n"
                    if round_no > 0 and finish is None:
                        fin_frame = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                        yield f"data: {json.dumps(fin_frame)}\n\n"
                    if not (finish == "length" and AUTOCONTINUE and round_no < AUTOCONTINUE_MAX):
                        if not done_sent:
                            done_sent = True
                            yield "data: [DONE]\n\n"
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    yield line + "\n\n"
                    continue
                ch = chunk.get("choices", [{}])[0] if chunk.get("choices") else {}
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    if round_no > 0 and resume_hold is not None:
                        # Resume round: buffer ALL content so overlap with the
                        # previous tail can be trimmed and the boundary joined
                        # before anything reaches the client.
                        resume_hold.append(delta["content"])
                        continue
                    buf_content.append(delta["content"])
                fr = ch.get("finish_reason")
                if fr:
                    finish = fr
                    # Hold back only the finish marker when we are going to
                    # resume; the frame's own delta content (if any) was
                    # already buffered above and will be re-emitted as part
                    # of the next attempt's prefix... actually it was already
                    # streamed to the client in earlier frames only if it came
                    # in its own frame. To be safe: when holding back, re-emit
                    # any pending delta content as its own frame first.
                    if fr == "length" and AUTOCONTINUE and round_no < AUTOCONTINUE_MAX:
                        if delta.get("content"):
                            hold = {"choices": [{"index": 0, "delta": {"content": delta["content"]},
                                                 "finish_reason": None}]}
                            yield f"data: {json.dumps(hold)}\n\n"
                        continue
                    if resume_hold is not None and resume_hold:
                        held = "".join(resume_hold)
                        resume_hold = None
                        held = _flush_held(held, streamed_tail)
                        if held:
                            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': held}, 'finish_reason': None}]})}\n\n"
                    yield line + "\n\n"
                    continue
                yield line + "\n\n"
            await resp.aclose()
            if finish != "length" or not AUTOCONTINUE or round_no >= AUTOCONTINUE_MAX:
                if not done_sent:
                    done_sent = True
                    yield "data: [DONE]\n\n"
                return
            # Resume: ask for the verbatim continuation of the partial text.
            partial = "".join(buf_content)
            tail = partial[-400:]
            attempt_body = dict(attempt_body)
            msgs = [dict(m) for m in attempt_body.get("messages", [])]
            msgs.append({"role": "user", "content":
                "Your reply above was cut off by the output length limit. Here is "
                "how it currently ends:\n<partial_tail>\n" + tail +
                "\n</partial_tail>\nOutput ONLY the missing continuation that "
                "follows <partial_tail> — no repetition of any earlier content, "
                "no commentary, no <partial_tail> tags. Resume mid-sentence if "
                "that is where it stopped."})
            attempt_body["messages"] = msgs
            kwargs = attempt_body.get("chat_template_kwargs") or {}
            attempt_body["chat_template_kwargs"] = {**kwargs, "enable_thinking": False}
            resume_hold = []
            streamed_tail = partial[-400:]
            if os.environ.get("SIDECAR_DEBUG"):
                print(f"[sidecar][debug] resume partial={partial!r} tail={tail!r}", flush=True)
            # Emit a zero-content frame so clients know the assistant turn continues.
            resume_mark = {"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]}
            yield f"data: {json.dumps(resume_mark)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.on_event("startup")
def startup() -> None:
    global encoder
    encoder = VisionEncoder()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "vision": encoder is not None,
        "model": MODEL,
        "revision": MODEL_REVISION,
        "resilience": {
            "autocontinue": AUTOCONTINUE,
            "loopbreak": LOOPBREAK,
            "autocompact": AUTOCOMPACT,
        },
    }


@app.post("/v1/chat/completions")
async def chat(req: Request) -> Response:
    body = await req.json()
    body["messages"] = await _rewrite(body.get("messages", []))
    stream = body.get("stream", False)
    if stream:
        return await _resilient_stream(body)
    return await _resilient_completion(body)


@app.api_route("/{path:path}", methods=["GET", "HEAD", "OPTIONS"])
async def discovery_proxy(path: str, req: Request) -> Response:
    """Pass model discovery and other read-only probes through to vLLM."""
    upstream_resp = await client.request(
        req.method,
        f"{UPSTREAM}/{path}",
        params=req.query_params,
        headers={
            key: value
            for key, value in req.headers.items()
            if key.lower() not in {"host", "content-length"}
        },
    )
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8016
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
