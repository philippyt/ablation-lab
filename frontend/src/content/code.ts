import type { ArchConfig } from "../types";

const IMPORTS = `import torch
import torch.nn as nn
import torch.nn.functional as F
`;

const RMSNORM = `class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight`;

const LAYERNORM_NOTE = `# Norm: torch.nn.LayerNorm(d_model). No custom class needed.`;

const ROPE = `def build_rope_cache(head_dim, max_seq, base=10000.0):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() / half))
    t = torch.arange(max_seq).float()
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)`;

const ALIBI = `def alibi_slopes(n_heads):
    # Geometric series 1/2, 1/4, ..., 1/2^n_heads
    return torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)])


def alibi_bias(q_len, k_len, slopes, device):
    i = torch.arange(q_len, device=device).unsqueeze(1)
    j = torch.arange(k_len, device=device).unsqueeze(0)
    distance = (j - i).abs().float()
    # Shape: (n_heads, q_len, k_len)
    return -slopes.view(-1, 1, 1).to(device) * distance`;

const ABSOLUTE_POS = `# Absolute learned position embeddings:
# self.pos_embed = nn.Embedding(ctx, d_model)
# x = tok_embed(idx) + self.pos_embed(torch.arange(T, device=idx.device))`;

const SINUSOIDAL_POS = `def sinusoidal_pos(ctx, d_model):
    pos = torch.arange(ctx).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(ctx, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # add to token embedding`;

const NOPE = `# NoPE: no positional information added.
# The causal mask alone carries enough position signal for decoder-only LMs.`;

function attentionCode(cfg: ArchConfig): string {
  const usesRope = cfg.pos_embed === "rope";
  const ropeApplyQ = usesRope ? `        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])\n        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])\n` : "";
  const ropeBuffers = usesRope ? `        cos, sin = build_rope_cache(head_dim, ctx)\n        self.register_buffer("rope_cos", cos)\n        self.register_buffer("rope_sin", sin)\n` : "";
  const qkNorm = cfg.qk_norm ? `        self.q_norm = RMSNorm(head_dim)\n        self.k_norm = RMSNorm(head_dim)\n` : "";
  const qkNormApply = cfg.qk_norm ? `        q = self.q_norm(q)\n        k = self.k_norm(k)\n` : "";
  const gqaRepeat = `        if self.group_size != 1:\n            k = k.repeat_interleave(self.group_size, dim=1)\n            v = v.repeat_interleave(self.group_size, dim=1)\n`;
  return `class Attention(nn.Module):
    def __init__(self, d_model=${cfg.d_model}, n_heads=${cfg.n_heads}, n_kv_heads=${cfg.n_kv_heads}, head_dim=${cfg.head_dim}, ctx=${cfg.ctx}):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.group_size = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
${ropeBuffers}${qkNorm}
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
${ropeApplyQ}${qkNormApply}${gqaRepeat}        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)`;
}

function ffnCode(cfg: ArchConfig): string {
  const actFunc = cfg.activation === "silu" ? "F.silu"
    : cfg.activation === "gelu" ? "F.gelu"
    : "lambda z: F.relu(z) ** 2";
  const gateName = cfg.activation === "silu" ? "SwiGLU" : cfg.activation === "gelu" ? "GeGLU" : "ReGLU2";
  const denseClass = `class ${gateName}(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        gate = ${actFunc}(self.w_gate(x))
        return self.w_down(gate * self.w_up(x))`;
  if (cfg.ffn === "dense") return denseClass;
  return `${denseClass}


class MoE(nn.Module):
    def __init__(self, d_model=${cfg.d_model}, hidden=${cfg.ffn_mult * cfg.d_model}, n_experts=${cfg.n_experts}, top_k=${cfg.top_k}):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([${gateName}(d_model, hidden) for _ in range(n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        probs = F.softmax(self.gate(x_flat), dim=-1)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = (top_idx == e)
            if not mask.any():
                continue
            tok = mask.any(-1).nonzero(as_tuple=True)[0]
            w = (mask[tok].to(top_w.dtype) * top_w[tok]).sum(-1, keepdim=True)
            out[tok] += w * self.experts[e](x_flat[tok])
        return out.view(B, T, D)`;
}

function blockCode(cfg: ArchConfig): string {
  const normClass = cfg.norm === "layernorm" ? "nn.LayerNorm" : "RMSNorm";
  const ffnClass = cfg.ffn === "dense"
    ? (cfg.activation === "silu" ? "SwiGLU" : cfg.activation === "gelu" ? "GeGLU" : "ReGLU2")
    : "MoE";
  const ffnArgs = cfg.ffn === "dense" ? `${cfg.d_model}, ${cfg.ffn_mult * cfg.d_model}` : "";
  return `class Block(nn.Module):
    def __init__(self, d_model=${cfg.d_model}):
        super().__init__()
        self.norm1 = ${normClass}(d_model)
        self.attn = Attention()
        self.norm2 = ${normClass}(d_model)
        self.ffn = ${ffnClass}(${ffnArgs})

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x`;
}

export function composeCode(cfg: ArchConfig): string {
  const parts: string[] = [];
  parts.push(IMPORTS);

  if (cfg.norm === "rmsnorm" || cfg.qk_norm) parts.push(RMSNORM);
  else parts.push(LAYERNORM_NOTE);

  if (cfg.pos_embed === "rope") parts.push(ROPE);
  else if (cfg.pos_embed === "alibi") parts.push(ALIBI);
  else if (cfg.pos_embed === "absolute") parts.push(ABSOLUTE_POS);
  else if (cfg.pos_embed === "sinusoidal") parts.push(SINUSOIDAL_POS);
  else parts.push(NOPE);

  parts.push(attentionCode(cfg));
  parts.push(ffnCode(cfg));
  parts.push(blockCode(cfg));

  return parts.join("\n\n\n");
}

const PY_KW = ["def","class","return","if","elif","else","import","from","as","for","in","while","not","and","or","is","with","lambda","None","True","False","self","super","torch","nn","F","math"];

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
