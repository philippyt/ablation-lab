from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ArchConfig
from .costs import swiglu_hidden

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

def _build_rope_cache(head_dim: int, max_seq: int, base: float = 10000.0, device=None):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_seq, device=device).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.group_size = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        cos, sin = _build_rope_cache(cfg.head_dim, cfg.ctx)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        offset = 0 if kv_cache is None else kv_cache.get("len", 0)
        cos = self.rope_cos[offset:offset + T]
        sin = self.rope_sin[offset:offset + T]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v
            kv_cache["len"] = k.size(2)
        if self.group_size != 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=_causal_mask(T, k.size(2), offset, q.device, q.dtype) if kv_cache is not None and offset > 0 else None,
            is_causal=(kv_cache is None or offset == 0),
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

class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class MoE(nn.Module):
    def __init__(self, cfg: ArchConfig, hidden: int):
        super().__init__()
        self.cfg = cfg
        self.top_k = cfg.top_k
        self.n_experts = cfg.n_experts
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, hidden) for _ in range(cfg.n_experts)])

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
        return SwiGLU(cfg.d_model, h)
    return MoE(cfg, h)

class Block(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        if cfg.pos_embed != "rope":
            raise NotImplementedError(f"pos_embed={cfg.pos_embed!r} not implemented yet; only 'rope' is supported")
        if cfg.activation != "silu":
            raise NotImplementedError(f"activation={cfg.activation!r} not implemented yet; only 'silu' is supported")
        if cfg.qk_norm:
            raise NotImplementedError("qk_norm=True not implemented yet")
        if cfg.attention_type != "standard":
            raise NotImplementedError(f"attention_type={cfg.attention_type!r} not implemented yet; only 'standard' is supported")
        if not cfg.enable_attention:
            raise NotImplementedError("enable_attention=False not implemented yet (no attention sublayer)")
        if not cfg.enable_ffn:
            raise NotImplementedError("enable_ffn=False not implemented yet (no FFN sublayer)")
        if not cfg.enable_norms:
            raise NotImplementedError("enable_norms=False not implemented yet (no norm layers)")
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_caches: list[dict] | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_embed(idx)
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache)
        x = self.final_norm(x)
        if self.lm_head is None:
            logits = x @ self.tok_embed.weight.t()
        else:
            logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        self.eval()
        kv_caches: list[dict] = [{} for _ in range(self.cfg.n_layers)]
        prompt = idx[:, -self.cfg.ctx:]
        logits, _ = self.forward(prompt, kv_caches=kv_caches)
        last_logits = logits[:, -1, :]
        cur_len = prompt.size(1)
        for _ in range(max_new_tokens):
            if cur_len >= self.cfg.ctx:
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
            logits, _ = self.forward(next_tok, kv_caches=kv_caches)
            last_logits = logits[:, -1, :]