import type { ArchConfig } from "../types";

function pyBool(b: boolean): string {
  return b ? "True" : "False";
}

function pyStr(s: string): string {
  return `"${s}"`;
}

function configBlock(cfg: ArchConfig): string {
  return `@dataclass
class Config:
    vocab_size: int = ${cfg.vocab_size}
    d_model: int = ${cfg.d_model}
    n_layers: int = ${cfg.n_layers}
    n_heads: int = ${cfg.n_heads}
    n_kv_heads: int = ${cfg.n_kv_heads}
    head_dim: int = ${cfg.head_dim}
    ctx: int = ${cfg.ctx}
    ffn_mult: int = ${cfg.ffn_mult}
    n_experts: int = ${cfg.n_experts}
    top_k: int = ${cfg.top_k}
    tie_embeddings: bool = ${pyBool(cfg.tie_embeddings)}
    pos_embed: str = ${pyStr(cfg.pos_embed)}
    activation: str = ${pyStr(cfg.activation)}
    norm: str = ${pyStr(cfg.norm)}
    ffn: str = ${pyStr(cfg.ffn)}
    qk_norm: bool = ${pyBool(cfg.qk_norm)}
    attention_type: str = ${pyStr(cfg.attention_type)}
    enable_attention: bool = ${pyBool(cfg.enable_attention)}
    enable_ffn: bool = ${pyBool(cfg.enable_ffn)}
    enable_norms: bool = ${pyBool(cfg.enable_norms)}

    def __post_init__(self):
        if self.head_dim * self.n_heads != self.d_model:
            assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
            self.head_dim = self.d_model // self.n_heads
        assert self.head_dim * self.n_heads == self.d_model, "head_dim * n_heads must equal d_model"
        assert self.head_dim % 2 == 0, "head_dim must be even (RoPE and MLA split require it)"
        assert self.n_kv_heads >= 1 and self.n_heads % self.n_kv_heads == 0, "n_heads must be a positive multiple of n_kv_heads"
        if self.ffn == "moe":
            assert self.top_k >= 1 and self.top_k <= self.n_experts, "top_k must be in [1, n_experts]"`;
}

const MODULES = `class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class PerHeadRMSNorm(nn.Module):
    def __init__(self, n_heads, head_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_heads, head_dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def make_norm(cfg):
    return nn.LayerNorm(cfg.d_model) if cfg.norm == "layernorm" else RMSNorm(cfg.d_model)


def activation_fn(kind):
    if kind == "silu":
        return F.silu
    if kind == "gelu":
        return F.gelu
    return lambda z: F.relu(z).pow(2)


def ffn_hidden(cfg):
    h = int((2 / 3) * cfg.ffn_mult * cfg.d_model)
    return ((h + 63) // 64) * 64


def build_rope_cache(head_dim, max_seq, base=10000.0):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
    t = torch.arange(max_seq).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def sinusoidal_table(ctx, d_model):
    pos = torch.arange(ctx).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(ctx, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def alibi_slopes(n_heads):
    return torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)])


def alibi_bias_matrix(seq_len, slopes, device, dtype):
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    distance = (j - i).clamp(max=0).to(dtype)
    return slopes.to(device=device, dtype=dtype).view(-1, 1, 1) * distance


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, H, KV, dh = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.group_size = H // KV
        self.q_proj = nn.Linear(d, H * dh, bias=False)
        self.k_proj = nn.Linear(d, KV * dh, bias=False)
        self.v_proj = nn.Linear(d, KV * dh, bias=False)
        self.o_proj = nn.Linear(H * dh, d, bias=False)
        if cfg.pos_embed == "rope":
            cos, sin = build_rope_cache(dh, cfg.ctx)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        if cfg.pos_embed == "alibi":
            self.register_buffer("alibi_slopes", alibi_slopes(H), persistent=False)
        if cfg.qk_norm:
            self.q_norm = PerHeadRMSNorm(H, dh)
            self.k_norm = PerHeadRMSNorm(KV, dh)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x):
        cfg = self.cfg
        B, T, _ = x.shape
        H, KV, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KV, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KV, dh).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        if cfg.pos_embed == "rope":
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        if self.group_size != 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        if cfg.pos_embed == "alibi":
            alibi = alibi_bias_matrix(T, self.alibi_slopes, x.device, x.dtype)
            causal = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype).triu(1)
            mask = causal.unsqueeze(0) + alibi
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.o_proj(out)


class MLAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.rope_dim = dh // 2
        self.nope_dim = dh - self.rope_dim
        self.latent = max(8, d // 8)
        self.q_proj = nn.Linear(d, H * dh, bias=False)
        self.kv_down = nn.Linear(d, self.latent, bias=False)
        self.k_rope_proj = nn.Linear(d, self.rope_dim, bias=False)
        self.k_up = nn.Linear(self.latent, H * self.nope_dim, bias=False)
        self.v_up = nn.Linear(self.latent, H * dh, bias=False)
        self.o_proj = nn.Linear(H * dh, d, bias=False)
        cos, sin = build_rope_cache(self.rope_dim, cfg.ctx)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if cfg.qk_norm:
            self.q_norm = PerHeadRMSNorm(H, dh)
            self.k_norm = PerHeadRMSNorm(H, dh)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x):
        cfg = self.cfg
        B, T, _ = x.shape
        H, dh = cfg.n_heads, cfg.head_dim
        q = self.q_proj(x).view(B, T, H, dh).transpose(1, 2)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        latent = self.kv_down(x)
        k_rope = self.k_rope_proj(x).unsqueeze(2)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        q_rope_rot = apply_rope(q_rope, cos, sin)
        k_rope_rot = apply_rope(k_rope.transpose(1, 2), cos, sin).transpose(1, 2)
        k_nope = self.k_up(latent).view(B, T, H, self.nope_dim).transpose(1, 2)
        v = self.v_up(latent).view(B, T, H, dh).transpose(1, 2)
        k_rope_full = k_rope_rot.repeat(1, 1, H, 1).transpose(1, 2)
        k = torch.cat([k_nope, k_rope_full], dim=-1).contiguous()
        q_full = torch.cat([q_nope, q_rope_rot], dim=-1).contiguous()
        if self.q_norm is not None:
            q_full = self.q_norm(q_full.transpose(1, 2)).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q_full, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * dh)
        return self.o_proj(out)


def make_attention(cfg):
    return MLAAttention(cfg) if cfg.attention_type == "mla" else Attention(cfg)


class GatedFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        h = ffn_hidden(cfg)
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)
        self.act = activation_fn(cfg.activation)

    def forward(self, x):
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([GatedFFN(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x):
        cfg = self.cfg
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        probs = F.softmax(self.gate(x_flat), dim=-1)
        top_w, top_idx = probs.topk(cfg.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for e in range(cfg.n_experts):
            mask = (top_idx == e)
            if not mask.any():
                continue
            tok = mask.any(-1).nonzero(as_tuple=True)[0]
            w = (mask[tok].to(top_w.dtype) * top_w[tok]).sum(-1, keepdim=True)
            out[tok] = out[tok] + w * self.experts[e](x_flat[tok])
        return out.view(B, T, D)


def make_ffn(cfg):
    return MoE(cfg) if cfg.ffn == "moe" else GatedFFN(cfg)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm1 = make_norm(cfg)
        self.attn = make_attention(cfg)
        self.norm2 = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    def forward(self, x):
        cfg = self.cfg
        if cfg.enable_attention:
            x_in = self.norm1(x) if cfg.enable_norms else x
            x = x + self.attn(x_in)
        if cfg.enable_ffn:
            x_in = self.norm2(x) if cfg.enable_norms else x
            x = x + self.ffn(x_in)
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_embed == "absolute":
            self.abs_pos = nn.Parameter(torch.zeros(cfg.ctx, cfg.d_model))
        else:
            self.abs_pos = None
        if cfg.pos_embed == "sinusoidal":
            self.register_buffer("sin_pe", sinusoidal_table(cfg.ctx, cfg.d_model), persistent=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx):
        cfg = self.cfg
        T = idx.size(1)
        x = self.tok_embed(idx)
        if self.abs_pos is not None and cfg.pos_embed == "absolute":
            x = x + self.abs_pos[:T].unsqueeze(0)
        elif cfg.pos_embed == "sinusoidal":
            x = x + self.sin_pe[:T].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        if cfg.enable_norms:
            x = self.final_norm(x)
        if self.lm_head is None:
            logits = x @ self.tok_embed.weight.t()
        else:
            logits = self.lm_head(x)
        return logits`;

const STUB = `cfg = Config()
model = TinyTransformer(cfg).eval()
T = min(16, cfg.ctx)
idx = torch.randint(0, cfg.vocab_size, (2, T))
with torch.no_grad():
    out = model(idx)
print("output shape:", tuple(out.shape))`;

const IMPORTS = `import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass`;

export function composeCode(cfg: ArchConfig): string {
  return [
    IMPORTS,
    configBlock(cfg),
    MODULES,
    STUB,
  ].join("\n\n").replace(/\n\n\n+/g, "\n\n");
}

const PY_KW = ["def","class","return","if","elif","else","import","from","as","for","in","while","not","and","or","is","with","lambda","None","True","False","self","super","torch","nn","F","math","int","str","bool","assert"];

export function highlightPython(code: string): string {
  const escaped = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const kwAlt = PY_KW.join("|");
  const tokenRe = new RegExp(
    `(#[^\\n]*)` +
    `|("[^"\\n]*")` +
    `|(\\b(?:${kwAlt})\\b)` +
    `|(\\b\\d+\\.?\\d*\\b)`,
    "g"
  );
  return escaped.replace(tokenRe, (m, c, s, k, n) => {
    if (c) return `<span class="py-comment">${c}</span>`;
    if (s) return `<span class="py-str">${s}</span>`;
    if (k) return `<span class="py-kw">${k}</span>`;
    if (n) return `<span class="py-num">${n}</span>`;
    return m;
  });
}
