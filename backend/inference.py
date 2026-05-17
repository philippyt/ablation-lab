from __future__ import annotations
import asyncio
import threading
import time
from collections import OrderedDict
from typing import AsyncIterator
import torch
import torch.nn.functional as F
from ablation.checkpoints import load_checkpoint
from ablation.config import ArchConfig, arch_config_from_dict, arch_hash
from ablation.model import TinyTransformer
from ablation.tokenizer import CharTokenizer
from backend.instrumentation import Instrumenter

_LRU_MAX = 3
_lru: "OrderedDict[str, tuple[TinyTransformer, CharTokenizer, ArchConfig, float]]" = OrderedDict()
_lock = threading.Lock()

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_for_hash(h: str) -> tuple[TinyTransformer, CharTokenizer, ArchConfig, float]:
    with _lock:
        if h in _lru:
            _lru.move_to_end(h)
            return _lru[h]
    ckpt = load_checkpoint(h, map_location="cpu")
    cfg = arch_config_from_dict(ckpt["arch_config"])
    tokenizer = CharTokenizer.from_state(ckpt["tokenizer_state"])
    model = TinyTransformer(cfg).to(_device())
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    val_loss = float(ckpt.get("final_val_loss", float("nan")))
    entry = (model, tokenizer, cfg, val_loss)
    with _lock:
        _lru[h] = entry
        _lru.move_to_end(h)
        while len(_lru) > _LRU_MAX:
            _lru.popitem(last=False)
    return entry

_EOS_MIN_GENERATED_CHARS = 30
_EOS_TERMINATORS = (".", "!", "?", '"')

def _hit_story_boundary(generated: str) -> bool:
    if len(generated) < _EOS_MIN_GENERATED_CHARS:
        return False
    tail = generated[-6:]
    if "\n\n" not in tail:
        return False
    nl = tail.find("\n")
    if nl <= 0:
        return False
    prev = tail[nl - 1]
    return prev in _EOS_TERMINATORS

def _build_runtime_overrides(requested_cfg: ArchConfig, trained_cfg: ArchConfig) -> dict:
    runtime: dict = {
        "enable_attention": requested_cfg.enable_attention,
        "enable_ffn": requested_cfg.enable_ffn,
        "enable_norms": requested_cfg.enable_norms,
        "tie_embeddings": requested_cfg.tie_embeddings,
    }
    rp = requested_cfg.pos_embed
    tp = trained_cfg.pos_embed
    if tp == "rope" and rp in ("rope", "alibi", "sinusoidal", "nope"):
        runtime["pos_embed"] = rp
    elif tp == "absolute" and rp in ("absolute", "nope"):
        runtime["pos_embed"] = rp
    else:
        runtime["pos_embed"] = tp
    return runtime

def _instrumented_generate(model, idx, max_new_tokens, temperature, top_k, runtime, instrumenter, tokenizer, stop_flag):
    model.eval()
    cfg = model.cfg
    kv_caches = [{} for _ in range(cfg.n_layers)]
    prompt = idx[:, -cfg.ctx:]
    with torch.no_grad():
        logits, _ = model.forward(prompt, kv_caches=kv_caches, runtime=runtime)
    last_logits = logits[:, -1, :]
    cur_len = prompt.size(1)
    eff_pos = (runtime or {}).get("pos_embed", cfg.pos_embed)
    absolute_pos_active = (eff_pos == "absolute") and (model.abs_pos is not None)
    for _ in range(max_new_tokens):
        if stop_flag.is_set():
            return
        if absolute_pos_active and cur_len >= cfg.ctx:
            return
        with torch.no_grad():
            if temperature > 0:
                scaled = last_logits / max(temperature, 1e-8)
                if top_k is not None:
                    v, _ = scaled.topk(min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = last_logits.argmax(dim=-1, keepdim=True)
        tok_id = int(next_tok.item())
        if instrumenter is not None:
            instrumenter.record_token(last_logits, tok_id, tokenizer.decode([tok_id]))
        yield tok_id
        cur_len += 1
        with torch.no_grad():
            logits, _ = model.forward(next_tok, kv_caches=kv_caches, runtime=runtime)
        last_logits = logits[:, -1, :]

async def stream_generate(*, requested_config: ArchConfig, resolution: dict, prompt: str, max_tokens: int, temperature: float, top_k: int | None, instrument: bool = False) -> AsyncIterator[dict]:
    h = resolution["hash"]
    model, tokenizer, cfg, val_loss = load_for_hash(h)
    device = next(model.parameters()).device
    runtime = _build_runtime_overrides(requested_config, cfg)
    prompt_ids = tokenizer.encode(prompt) or [0]
    prompt_tokens = min(len(prompt_ids), cfg.ctx)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kind_b_active = [k for k in ("enable_attention", "enable_ffn", "enable_norms") if runtime.get(k) is False]
    if runtime.get("pos_embed") and runtime["pos_embed"] != cfg.pos_embed:
        kind_b_active.append(f"pos_embed={runtime['pos_embed']}")
    yield {
        "event": "start",
        "prompt_tokens": prompt_tokens,
        "ctx": cfg.ctx,
        "hash": h,
        "state": resolution["state"],
        "differing_fields": resolution["differing_fields"],
        "distance": resolution["distance"],
        "kind_b_active": kind_b_active,
    }
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    stop_flag = threading.Event()

    instrumenter: Instrumenter | None = None
    if instrument:
        instrumenter = Instrumenter(model)
        instrumenter.attach()

    def worker():
        try:
            for tok in _instrumented_generate(
                model, idx, max_tokens, temperature, top_k, runtime, instrumenter, tokenizer, stop_flag,
            ):
                if stop_flag.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(tok), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

    t_start = time.time()
    threading.Thread(target=worker, daemon=True).start()

    generated = ""
    tokens_emitted = 0
    stop_reason: str | None = None
    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, tuple) and item[0] == "error":
            if instrumenter is not None:
                instrumenter.detach()
            yield {"event": "error", "message": item[1]}
            return
        text = tokenizer.decode([int(item)])
        generated += text
        tokens_emitted += 1
        yield {"event": "token", "text": text}
        if _hit_story_boundary(generated):
            stop_reason = "eos"
            stop_flag.set()
            break

    if stop_reason == "eos":
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is SENTINEL:
                break

    if stop_reason is None:
        if tokens_emitted >= max_tokens:
            stop_reason = "max_tokens"
        else:
            stop_reason = "ctx_limit"

    snapshot: dict | None = None
    if instrumenter is not None:
        snapshot = instrumenter.snapshot()
        instrumenter.detach()
        yield {"event": "instr", **snapshot}

    elapsed = time.time() - t_start
    yield {
        "event": "done",
        "val_loss": val_loss,
        "hash": h,
        "stop_reason": stop_reason,
        "tokens": tokens_emitted,
        "elapsed_s": round(elapsed, 4),
    }