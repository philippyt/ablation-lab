from __future__ import annotations
import asyncio
import threading
from collections import OrderedDict
from typing import AsyncIterator
import torch
from ablation.checkpoints import load_checkpoint
from ablation.config import ArchConfig, arch_config_from_dict, arch_hash
from ablation.model import TinyTransformer
from ablation.tokenizer import CharTokenizer

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

async def stream_generate(*, requested_config: ArchConfig, resolution: dict, prompt: str, max_tokens: int, temperature: float, top_k: int | None) -> AsyncIterator[dict]:
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

    def worker():
        try:
            for tok in model.generate(
                idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, runtime=runtime
            ):
                if stop_flag.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(tok), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

    threading.Thread(target=worker, daemon=True).start()

    generated = ""
    tokens_emitted = 0
    stop_reason: str | None = None
    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, tuple) and item[0] == "error":
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

    yield {"event": "done", "val_loss": val_loss, "hash": h, "stop_reason": stop_reason, "tokens": tokens_emitted}