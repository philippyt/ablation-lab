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

async def stream_generate(*, config: ArchConfig, prompt: str, max_tokens: int, temperature: float, top_k: int | None) -> AsyncIterator[dict]:
    h = arch_hash(config)
    model, tokenizer, cfg, val_loss = load_for_hash(h)
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt) or [0]
    prompt_tokens = min(len(prompt_ids), cfg.ctx)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    yield {"event": "start", "prompt_tokens": prompt_tokens, "ctx": cfg.ctx, "hash": h}
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def worker():
        try:
            for tok in model.generate(
                idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k
            ):
                asyncio.run_coroutine_threadsafe(queue.put(tok), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, tuple) and item[0] == "error":
            yield {"event": "error", "message": item[1]}
            return
        yield {"event": "token", "text": tokenizer.decode([int(item)])}

    yield {"event": "done", "val_loss": val_loss, "hash": h}