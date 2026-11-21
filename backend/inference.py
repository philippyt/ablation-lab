from __future__ import annotations
import asyncio
import math
import threading
import time
from collections import OrderedDict
from typing import AsyncIterator
import torch
import torch.nn.functional as F
from ablation.checkpoints import load_checkpoint
from ablation.config import ArchConfig, arch_config_from_dict
from ablation.model import TinyTransformer
from ablation.tokenizer import CharTokenizer
from backend.instrumentation import Instrumenter

_LRU_MAX = 6
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

def _build_runtime_overrides(
    requested_cfg: ArchConfig,
    trained_cfg: ArchConfig,
    surgery: dict | None = None,
    kv: dict | None = None,
) -> dict:
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
    if surgery:
        runtime["surgery"] = surgery
    if kv:
        runtime["kv"] = kv
    return runtime

def _sample_next(last_logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
    if temperature > 0:
        scaled = last_logits / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = scaled.topk(min(top_k, scaled.size(-1)))
            scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
    else:
        next_tok = last_logits.argmax(dim=-1, keepdim=True)
    return int(next_tok.item())

def _top_k_probs(last_logits: torch.Tensor, tokenizer: CharTokenizer, k: int = 5) -> list[dict]:
    probs = F.softmax(last_logits[0].float(), dim=-1)
    vals, idxs = probs.topk(min(k, probs.size(-1)))
    return [
        {"ch": tokenizer.decode([int(i.item())]), "p": round(float(v.item()), 4)}
        for v, i in zip(vals, idxs)
    ]

def _step_telemetry_from_metrics(step_metrics: dict, last_logits: torch.Tensor) -> tuple[float, float]:
    lmax_dict = step_metrics.get("attn_logit_max", {})
    lm = max(lmax_dict.values()) if lmax_dict else 0.0
    probs = F.softmax(last_logits[0].float(), dim=-1)
    ent = -(probs * probs.clamp_min(1e-12).log()).sum().item()
    return round(lm, 3), round(ent, 3)

def _instrumented_generate(model, idx, max_new_tokens, temperature, top_k, runtime, instrumenter, tokenizer, stop_flag):
    model.eval()
    cfg = model.cfg
    kv_caches = [{} for _ in range(cfg.n_layers)]
    prompt = idx[:, -cfg.ctx:]
    step_metrics: dict = {}
    rt = {**runtime, "step_metrics": step_metrics} if instrumenter is not None else runtime
    with torch.no_grad():
        logits, _ = model.forward(prompt, kv_caches=kv_caches, runtime=rt)
    last_logits = logits[:, -1, :]
    cur_len = prompt.size(1)
    eff_pos = rt.get("pos_embed", cfg.pos_embed)
    absolute_pos_active = (eff_pos == "absolute") and (model.abs_pos is not None)
    for _ in range(max_new_tokens):
        if stop_flag.is_set():
            return
        if absolute_pos_active and cur_len >= cfg.ctx:
            return
        with torch.no_grad():
            tok_id = _sample_next(last_logits, temperature, top_k)
        tok_str = tokenizer.decode([tok_id])
        tele = None
        if instrumenter is not None:
            tele = instrumenter.record_step(step_metrics, last_logits, tok_id, tok_str)
        yield (tok_id, tele)
        cur_len += 1
        step_metrics = {}
        rt = {**runtime, "step_metrics": step_metrics} if instrumenter is not None else runtime
        next_tensor = torch.tensor([[tok_id]], dtype=torch.long, device=idx.device)
        with torch.no_grad():
            logits, _ = model.forward(next_tensor, kv_caches=kv_caches, runtime=rt)
        last_logits = logits[:, -1, :]

async def stream_generate(
    *,
    requested_config: ArchConfig,
    resolution: dict,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    instrument: bool = False,
    surgery: dict | None = None,
    kv: dict | None = None,
) -> AsyncIterator[dict]:
    h = resolution["hash"]
    model, tokenizer, cfg, val_loss = load_for_hash(h)
    device = next(model.parameters()).device
    runtime = _build_runtime_overrides(requested_config, cfg, surgery=surgery, kv=kv)
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
        instrumenter = Instrumenter(model, runtime=runtime)
        instrumenter.attach()

    def worker():
        try:
            for tok_id, tele in _instrumented_generate(
                model, idx, max_tokens, temperature, top_k, runtime, instrumenter, tokenizer, stop_flag,
            ):
                if stop_flag.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put((tok_id, tele)), loop)
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
        tok_id, tele = item
        text = tokenizer.decode([int(tok_id)])
        generated += text
        tokens_emitted += 1
        ev: dict = {"event": "token", "text": text}
        if tele is not None:
            ev.update(tele)
        yield ev
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

async def stream_race(
    *,
    config_a: ArchConfig,
    resolution_a: dict,
    config_b: ArchConfig,
    resolution_b: dict,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    surgery_a: dict | None = None,
    surgery_b: dict | None = None,
    kv_a: dict | None = None,
    kv_b: dict | None = None,
) -> AsyncIterator[dict]:
    ha, hb = resolution_a["hash"], resolution_b["hash"]
    model_a, tok_a, cfg_a, val_a = load_for_hash(ha)
    model_b, tok_b, cfg_b, val_b = load_for_hash(hb)
    dev_a = next(model_a.parameters()).device
    dev_b = next(model_b.parameters()).device
    rt_a_base = _build_runtime_overrides(config_a, cfg_a, surgery=surgery_a, kv=kv_a)
    rt_b_base = _build_runtime_overrides(config_b, cfg_b, surgery=surgery_b, kv=kv_b)

    yield {
        "event": "race_start",
        "hash_a": ha,
        "hash_b": hb,
        "val_loss_a": val_a,
        "val_loss_b": val_b,
        "state_a": resolution_a["state"],
        "state_b": resolution_b["state"],
    }

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    stop_flag = threading.Event()

    def worker():
        try:
            ids_a = tok_a.encode(prompt) or [0]
            ids_b = tok_b.encode(prompt) or [0]
            prompt_a = torch.tensor([ids_a[-cfg_a.ctx:]], dtype=torch.long, device=dev_a)
            prompt_b = torch.tensor([ids_b[-cfg_b.ctx:]], dtype=torch.long, device=dev_b)
            kv_ca = [{} for _ in range(cfg_a.n_layers)]
            kv_cb = [{} for _ in range(cfg_b.n_layers)]
            ma: dict = {}
            mb: dict = {}
            with torch.no_grad():
                la, _ = model_a.forward(prompt_a, kv_caches=kv_ca, runtime={**rt_a_base, "step_metrics": ma})
                lb, _ = model_b.forward(prompt_b, kv_caches=kv_cb, runtime={**rt_b_base, "step_metrics": mb})
            last_a = la[:, -1, :]
            last_b = lb[:, -1, :]
            len_a = prompt_a.size(1)
            len_b = prompt_b.size(1)
            abs_a = (rt_a_base.get("pos_embed", cfg_a.pos_embed) == "absolute") and (model_a.abs_pos is not None)
            abs_b = (rt_b_base.get("pos_embed", cfg_b.pos_embed) == "absolute") and (model_b.abs_pos is not None)
            done_a = False
            done_b = False
            stop_a: str | None = None
            stop_b: str | None = None
            gen_a = ""
            gen_b = ""
            first_diverge: int | None = None
            diff_detail_sent = 0

            for step in range(max_tokens):
                if stop_flag.is_set():
                    break
                if not done_a and abs_a and len_a >= cfg_a.ctx:
                    done_a = True
                    stop_a = "ctx_limit"
                if not done_b and abs_b and len_b >= cfg_b.ctx:
                    done_b = True
                    stop_b = "ctx_limit"
                if done_a and done_b:
                    break

                ch_a = ""
                ch_b = ""
                lm_a = 0.0
                ent_a = 0.0
                lm_b = 0.0
                ent_b = 0.0
                next_a_id: int | None = None
                next_b_id: int | None = None

                with torch.no_grad():
                    if not done_a:
                        next_a_id = _sample_next(last_a, temperature, top_k)
                        ch_a = tok_a.decode([next_a_id])
                        lm_a, ent_a = _step_telemetry_from_metrics(ma, last_a)
                    if not done_b:
                        next_b_id = _sample_next(last_b, temperature, top_k)
                        ch_b = tok_b.decode([next_b_id])
                        lm_b, ent_b = _step_telemetry_from_metrics(mb, last_b)

                diverged = (ch_a != ch_b)
                if diverged and first_diverge is None:
                    first_diverge = step

                payload: dict = {
                    "step": step,
                    "a": ch_a,
                    "b": ch_b,
                    "diff": diverged,
                    "lm_a": lm_a,
                    "lm_b": lm_b,
                    "ent_a": ent_a,
                    "ent_b": ent_b,
                }
                if diverged and diff_detail_sent < 60:
                    with torch.no_grad():
                        payload["top_a"] = _top_k_probs(last_a, tok_a, 5) if not done_a else []
                        payload["top_b"] = _top_k_probs(last_b, tok_b, 5) if not done_b else []
                    diff_detail_sent += 1

                asyncio.run_coroutine_threadsafe(queue.put(payload), loop)

                if not done_a and next_a_id is not None:
                    gen_a += ch_a
                    len_a += 1
                    if _hit_story_boundary(gen_a):
                        done_a = True
                        stop_a = "eos"
                    else:
                        ma = {}
                        t_a = torch.tensor([[next_a_id]], dtype=torch.long, device=dev_a)
                        with torch.no_grad():
                            la, _ = model_a.forward(t_a, kv_caches=kv_ca, runtime={**rt_a_base, "step_metrics": ma})
                        last_a = la[:, -1, :]

                if not done_b and next_b_id is not None:
                    gen_b += ch_b
                    len_b += 1
                    if _hit_story_boundary(gen_b):
                        done_b = True
                        stop_b = "eos"
                    else:
                        mb = {}
                        t_b = torch.tensor([[next_b_id]], dtype=torch.long, device=dev_b)
                        with torch.no_grad():
                            lb, _ = model_b.forward(t_b, kv_caches=kv_cb, runtime={**rt_b_base, "step_metrics": mb})
                        last_b = lb[:, -1, :]

            asyncio.run_coroutine_threadsafe(
                queue.put((
                    "done",
                    {
                        "first_diverge": first_diverge,
                        "stop_reason_a": stop_a or "max_tokens",
                        "stop_reason_b": stop_b or "max_tokens",
                        "tokens_a": len(gen_a),
                        "tokens_b": len(gen_b),
                    },
                )),
                loop,
            )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

    t_start = time.time()
    threading.Thread(target=worker, daemon=True).start()

    done_info: dict = {}
    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        if isinstance(item, tuple) and item[0] == "error":
            yield {"event": "error", "message": item[1]}
            return
        if isinstance(item, tuple) and item[0] == "done":
            done_info = item[1]
            continue
        yield {"event": "race_step", **item}

    elapsed = time.time() - t_start
    yield {"event": "race_done", **done_info, "elapsed_s": round(elapsed, 4)}


def inspect_sequence(
    *,
    requested_config: ArchConfig,
    resolution: dict,
    text: str,
    surgery: dict | None = None,
    kv: dict | None = None,
    include_attn: bool = True,
    include_logit_lens: bool = True,
    max_attn_tokens: int = 32,
) -> dict:
    h = resolution["hash"]
    model, tokenizer, cfg, val_loss = load_for_hash(h)
    device = next(model.parameters()).device
    ids = tokenizer.encode(text) or [0]
    ids = ids[-cfg.ctx:]
    step_metrics: dict = {
        "capture_attn_weights": include_attn,
        "capture_logit_lens": include_logit_lens,
    }
    runtime = _build_runtime_overrides(requested_config, cfg, surgery=surgery, kv=kv)
    runtime["step_metrics"] = step_metrics
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model.forward(idx, runtime=runtime)

    out: dict = {
        "hash": h,
        "val_loss": val_loss,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
    }

    if include_attn:
        slice_len = min(len(ids), max_attn_tokens)
        start = len(ids) - slice_len
        out["tokens"] = [tokenizer.decode([t]) for t in ids[start:]]
        raw_attn: dict[int, torch.Tensor] = step_metrics.get("attn_weights", {})
        disabled_heads_map = (surgery or {}).get("disabled_heads", {}) if surgery else {}
        disabled_attn_layers = set((surgery or {}).get("disabled_attn_layers", [])) if surgery else set()
        attn_out: list[list[list[list[float]]]] = []
        for l_idx in range(cfg.n_layers):
            layer_off = (not requested_config.enable_attention) or (l_idx in disabled_attn_layers)
            t_w = raw_attn.get(l_idx)
            if layer_off or t_w is None:
                zeros = [[[0.0] * slice_len for _ in range(slice_len)] for _ in range(cfg.n_heads)]
                attn_out.append(zeros)
            else:
                sub = t_w[:, start:, start:]
                off_h = set(disabled_heads_map.get(l_idx, disabled_heads_map.get(str(l_idx), [])))
                layer_list: list[list[list[float]]] = []
                for h_idx in range(cfg.n_heads):
                    if h_idx in off_h:
                        layer_list.append([[0.0] * slice_len for _ in range(slice_len)])
                    else:
                        layer_list.append([[round(float(v), 3) for v in row] for row in sub[h_idx].tolist()])
                attn_out.append(layer_list)
        out["attn_weights"] = attn_out

    if include_logit_lens:
        lens_raw = step_metrics.get("logit_lens", [])
        stages = []
        for label, logits_vec in lens_raw:
            probs = F.softmax(logits_vec[0], dim=-1)
            vals, top_ids = probs.topk(min(5, probs.size(-1)))
            stages.append({
                "stage": label,
                "top": [
                    {"ch": tokenizer.decode([int(i.item())]), "p": round(float(v.item()), 4)}
                    for v, i in zip(vals, top_ids)
                ],
            })
        out["logit_lens"] = stages

    return out
