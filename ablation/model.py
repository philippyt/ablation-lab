from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ArchConfig
from .costs import swiglu_hidden, _mla_latent_dim, _mla_rope_dim

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

def make_norm(cfg: ArchConfig) -> nn.Module:
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model)
    return RMSNorm(cfg.d_model)

def _activation_fn(kind: str):
    if kind == "silu":
        return F.silu
    if kind == "gelu":
        return F.gelu
    if kind == "relu2":
        return lambda x: F.relu(x).pow(2)
    raise ValueError(f"unknown activation: {kind}")

def _build_rope_cache(head_dim: int, max_seq: int, base: float = 10000.0, device=None):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_seq, device=device).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()

def _rope_at(head_dim: int, offset: int, T: int, base: float = 10000.0, device=None, dtype=None):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = (torch.arange(T, device=device).float() + offset)
    angles = torch.outer(t, freqs)
    cos = angles.cos()
    sin = angles.sin()
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def _alibi_slopes(n_heads: int) -> torch.Tensor:
    base = 2.0 ** -(8.0 / n_heads)
    slopes = torch.tensor([base ** (i + 1) for i in range(n_heads)])
    return slopes

def _alibi_bias(slopes: torch.Tensor, q_len: int, k_len: int, q_offset: int, device, dtype) -> torch.Tensor:
    i = torch.arange(q_len, device=device).unsqueeze(1) + q_offset
    j = torch.arange(k_len, device=device).unsqueeze(0)
    dist = (j - i).clamp(max=0).to(dtype)
    return slopes.to(device=device, dtype=dtype).view(-1, 1, 1) * dist.unsqueeze(0)

def _sinusoidal_table(seq_len: int, d_model: int, offset: int, device, dtype) -> torch.Tensor:
    pos = (torch.arange(seq_len, device=device) + offset).float()
    div = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2, device=device).float() / d_model)
    angle = pos.unsqueeze(1) * div.unsqueeze(0)
    out = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    out[:, 0::2] = torch.sin(angle).to(dtype)
    out[:, 1::2] = torch.cos(angle).to(dtype)
    return out

class _PerHeadRMSNorm(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

class Attention(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.group_size = cfg.n_heads // cfg.n_kv_heads
        self.trained_pos = cfg.pos_embed
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        if cfg.pos_embed == "rope":
            cos, sin = _build_rope_cache(cfg.head_dim, cfg.ctx)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.register_buffer("alibi_slopes", _alibi_slopes(cfg.n_heads), persistent=False)
        if cfg.qk_norm:
            self.q_norm = _PerHeadRMSNorm(cfg.n_heads, cfg.head_dim)
            self.k_norm = _PerHeadRMSNorm(cfg.n_kv_heads, cfg.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None, runtime: dict | None = None) -> torch.Tensor:
        rt = runtime or {}
        eff_pos = rt.get("pos_embed", self.trained_pos)
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        pos = 0 if kv_cache is None else kv_cache.get("pos", 0)
        if self.trained_pos == "rope" and eff_pos == "rope":
            cos, sin = _rope_at(self.head_dim, pos, T, device=q.device, dtype=q.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            if k.size(2) > self.cfg.ctx:
                k = k[:, :, -self.cfg.ctx:, :].contiguous()
                v = v[:, :, -self.cfg.ctx:, :].contiguous()
            kv_cache["k"] = k
            kv_cache["v"] = v
            kv_cache["len"] = k.size(2)
            kv_cache["pos"] = pos + T
        if self.group_size != 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        Tk = k.size(2)
        cache_offset = Tk - T
        attn_mask = None
        is_causal = (kv_cache is None or cache_offset == 0)
        if kv_cache is not None and cache_offset > 0:
            attn_mask = _causal_mask(T, Tk, cache_offset, q.device, q.dtype)
            is_causal = False
        if self.trained_pos == "rope" and eff_pos == "alibi":
            alibi = _alibi_bias(self.alibi_slopes, T, Tk, cache_offset, q.device, q.dtype)
            if attn_mask is None:
                attn_mask = _causal_mask(T, Tk, cache_offset, q.device, q.dtype)
                is_causal = False
            attn_mask = attn_mask + alibi
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)

class MLAAttention(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_dim = _mla_rope_dim(cfg)
        self.nope_dim = cfg.head_dim - self.rope_dim
        self.latent = _mla_latent_dim(cfg)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.kv_down = nn.Linear(cfg.d_model, self.latent, bias=False)
        self.k_rope_proj = nn.Linear(cfg.d_model, self.rope_dim, bias=False)
        self.k_up = nn.Linear(self.latent, cfg.n_heads * self.nope_dim, bias=False)
        self.v_up = nn.Linear(self.latent, cfg.n_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        cos, sin = _build_rope_cache(self.rope_dim, cfg.ctx)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if cfg.qk_norm:
            self.q_norm = _PerHeadRMSNorm(cfg.n_heads, cfg.head_dim)
            self.k_norm = _PerHeadRMSNorm(cfg.n_heads, cfg.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None, runtime: dict | None = None) -> torch.Tensor:
        rt = runtime or {}
        eff_pos = rt.get("pos_embed", "rope")
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        latent = self.kv_down(x)
        k_rope = self.k_rope_proj(x).unsqueeze(2)
        pos = 0 if kv_cache is None else kv_cache.get("pos", 0)
        if eff_pos == "rope":
            cos, sin = _rope_at(self.rope_dim, pos, T, device=q.device, dtype=q.dtype)
            q_rope_rot = _apply_rope(q_rope, cos, sin)
            k_rope_for_rope = k_rope.transpose(1, 2)
            k_rope_rot = _apply_rope(k_rope_for_rope, cos, sin).transpose(1, 2)
        else:
            q_rope_rot = q_rope
            k_rope_rot = k_rope
        if kv_cache is not None:
            if "latent" in kv_cache:
                latent = torch.cat([kv_cache["latent"], latent], dim=1)
                k_rope_rot = torch.cat([kv_cache["k_rope"], k_rope_rot], dim=1)
            if latent.size(1) > self.cfg.ctx:
                latent = latent[:, -self.cfg.ctx:, :].contiguous()
                k_rope_rot = k_rope_rot[:, -self.cfg.ctx:, :, :].contiguous()
            kv_cache["latent"] = latent
            kv_cache["k_rope"] = k_rope_rot
            kv_cache["len"] = latent.size(1)
            kv_cache["pos"] = pos + T
        Tk = latent.size(1)
        cache_offset = Tk - T
        k_nope = self.k_up(latent).view(B, Tk, self.n_heads, self.nope_dim).transpose(1, 2)
        v = self.v_up(latent).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        k_rope_full = k_rope_rot.expand(B, Tk, self.n_heads, self.rope_dim).transpose(1, 2)
        k = torch.cat([k_nope, k_rope_full], dim=-1)
        q_full = torch.cat([q_nope, q_rope_rot], dim=-1)
        if self.q_norm is not None:
            q_full = self.q_norm(q_full.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q_full, k, v,
            attn_mask=_causal_mask(T, Tk, cache_offset, q_full.device, q_full.dtype) if kv_cache is not None and cache_offset > 0 else None,
            is_causal=(kv_cache is None or cache_offset == 0),
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)

def _causal_mask(q_len: int, k_len: int, offset: int, device, dtype) -> torch.Tensor:
    i = torch.arange(q_len, device=device).unsqueeze(1) + offset
    j = torch.arange(k_len, device=device).unsqueeze(0)
    allowed = j <= i
    mask = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask

def make_attention(cfg: ArchConfig) -> nn.Module:
    if cfg.attention_type == "mla":
        return MLAAttention(cfg)
    return Attention(cfg)

class GatedFFN(nn.Module):
    def __init__(self, d: int, hidden: int, activation: str):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.act_fn = _activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.act_fn(self.w_gate(x)) * self.w_up(x))

class MoE(nn.Module):
    def __init__(self, cfg: ArchConfig, hidden: int):
        super().__init__()
        self.cfg = cfg
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_experts
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([GatedFFN(cfg.d_model, hidden, cfg.activation) for _ in range(cfg.n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = (top_idx == e)
            if not mask.any():
                continue
            tok = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            w = (mask[tok].to(top_w.dtype) * top_w[tok]).sum(dim=-1, keepdim=True)
            out[tok] = out[tok] + w * self.experts[e](x_flat[tok])
        return out.view(B, T, D)

def make_ffn(cfg: ArchConfig) -> nn.Module:
    h = swiglu_hidden(cfg)
    if cfg.ffn == "dense":
        return GatedFFN(cfg.d_model, h, cfg.activation)
    return MoE(cfg, h)

class Block(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = make_attention(cfg)
        self.norm2 = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None, runtime: dict | None = None) -> torch.Tensor:
        rt = runtime or {}
        enable_norms = rt.get("enable_norms", True)
        enable_attention = rt.get("enable_attention", True)
        enable_ffn = rt.get("enable_ffn", True)
        if enable_attention:
            x_in = self.norm1(x) if enable_norms else x
            x = x + self.attn(x_in, kv_cache=kv_cache, runtime=rt)
        if enable_ffn:
            x_in = self.norm2(x) if enable_norms else x
            x = x + self.ffn(x_in)
        return x

class TinyTransformer(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        if cfg.pos_embed not in ("rope", "absolute"):
            raise NotImplementedError(f"pos_embed={cfg.pos_embed!r} not a Kind-A training value; only 'rope' or 'absolute'.")
        if cfg.activation not in ("silu", "gelu", "relu2"):
            raise NotImplementedError(f"activation={cfg.activation!r} not implemented")
        if cfg.attention_type not in ("standard", "mla"):
            raise NotImplementedError(f"attention_type={cfg.attention_type!r} not implemented")
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_embed == "absolute":
            self.abs_pos = nn.Parameter(torch.zeros(cfg.ctx, cfg.d_model))
        else:
            self.abs_pos = None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_caches: list[dict] | None = None, runtime: dict | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        rt = runtime or {}
        enable_norms = rt.get("enable_norms", True)
        eff_pos = rt.get("pos_embed", self.cfg.pos_embed)
        x = self.tok_embed(idx)
        abs_offset = 0
        T = idx.size(1)
        if kv_caches is not None and kv_caches[0] is not None:
            abs_offset = kv_caches[0].get("pos", 0)
        if self.abs_pos is not None and eff_pos == "absolute":
            end = abs_offset + T
            if end > self.cfg.ctx:
                end = self.cfg.ctx
                start = max(0, end - T)
                pad = T - (end - start)
                if pad > 0:
                    last = self.abs_pos[-1:].unsqueeze(0).expand(1, pad, self.cfg.d_model)
                    x = x + torch.cat([self.abs_pos[start:end].unsqueeze(0), last], dim=1)
                else:
                    x = x + self.abs_pos[start:end].unsqueeze(0)
            else:
                x = x + self.abs_pos[abs_offset:abs_offset + T].unsqueeze(0)
        elif self.cfg.pos_embed == "rope" and eff_pos == "sinusoidal":
            x = x + _sinusoidal_table(T, self.cfg.d_model, abs_offset, x.device, x.dtype).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache, runtime=rt)
        x = self.final_norm(x) if enable_norms else x
        tied = rt.get("tie_embeddings", self.cfg.tie_embeddings)
        if tied or self.lm_head is None:
            logits = x @ self.tok_embed.weight.t()
        else:
            logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, runtime: dict | None = None):
        self.eval()
        kv_caches: list[dict] = [{} for _ in range(self.cfg.n_layers)]
        prompt = idx[:, -self.cfg.ctx:]
        logits, _ = self.forward(prompt, kv_caches=kv_caches, runtime=runtime)
        last_logits = logits[:, -1, :]
        cur_len = prompt.size(1)
        eff_pos = (runtime or {}).get("pos_embed", self.cfg.pos_embed)
        absolute_pos_active = (eff_pos == "absolute") and (self.abs_pos is not None)
        for _ in range(max_new_tokens):
            if absolute_pos_active and cur_len >= self.cfg.ctx:
                return
            if temperature > 0:
                scaled = last_logits / max(temperature, 1e-8)
                if top_k is not None:
                    v, _ = scaled.topk(min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = last_logits.argmax(dim=-1, keepdim=True)
            yield int(next_tok.item())
            cur_len += 1
            logits, _ = self.forward(next_tok, kv_caches=kv_caches, runtime=runtime)
            last_logits = logits[:, -1, :]
