from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from ablation.model import MoE, _rope_at, _apply_rope

class Instrumenter:
    def __init__(self, model):
        self.model = model
        self.cfg = model.cfg
        self.handles = []
        self.routing_counts = [[0] * self.cfg.n_experts for _ in range(self.cfg.n_layers)] if self.cfg.ffn == "moe" else []
        self.routing_totals = [0] * self.cfg.n_layers if self.cfg.ffn == "moe" else []
        self.attn_entropy_sum = [0.0] * self.cfg.n_layers
        self.attn_entropy_count = [0] * self.cfg.n_layers
        self.attn_max_entropy = [0.0] * self.cfg.n_layers
        self.residual_norm_sum = [0.0] * self.cfg.n_layers
        self.residual_norm_count = [0] * self.cfg.n_layers
        self.token_surprises: list[float] = []
        self.token_strings: list[str] = []
        self._entropy_step = 0
        self._norm_step = 0
        self.entropy_sample_every = 8
        self.norm_sample_every = 4
        self.instrument_attention = self.cfg.enable_attention and self.cfg.attention_type != "mla"

    def attach(self):
        for i, block in enumerate(self.model.blocks):
            self.handles.append(block.register_forward_hook(self._block_hook(i)))
            if self.cfg.ffn == "moe" and isinstance(block.ffn, MoE):
                self.handles.append(block.ffn.gate.register_forward_hook(self._gate_hook(i)))
            if self.instrument_attention:
                self.handles.append(block.attn.register_forward_hook(self._attn_hook(i), with_kwargs=True))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _block_hook(self, layer_idx):
        def hook(module, args, output):
            if layer_idx == 0:
                self._norm_step += 1
            if self._norm_step % self.norm_sample_every != 0:
                return
            with torch.no_grad():
                n = output.float().norm(dim=-1).mean().item()
                self.residual_norm_sum[layer_idx] += n
                self.residual_norm_count[layer_idx] += 1
        return hook

    def _gate_hook(self, layer_idx):
        def hook(module, args, output):
            with torch.no_grad():
                probs = F.softmax(output.float(), dim=-1)
                _, top_idx = probs.topk(self.cfg.top_k, dim=-1)
                flat = top_idx.reshape(-1).tolist()
                bucket = self.routing_counts[layer_idx]
                for e in flat:
                    bucket[e] += 1
                self.routing_totals[layer_idx] += len(flat)
        return hook

    def _attn_hook(self, layer_idx):
        def hook(module, args, kwargs, output):
            if layer_idx == 0:
                self._entropy_step += 1
            if self._entropy_step % self.entropy_sample_every != 0:
                return
            with torch.no_grad():
                x = args[0]
                kv_cache = kwargs.get("kv_cache")
                if kv_cache is None or "k" not in kv_cache:
                    return
                B, T, _ = x.shape
                H = module.cfg.n_heads
                KV = module.cfg.n_kv_heads
                dh = module.cfg.head_dim
                q = module.q_proj(x).view(B, T, H, dh).transpose(1, 2)
                if module.q_norm is not None:
                    q = module.q_norm(q.transpose(1, 2)).transpose(1, 2)
                pos_end = kv_cache.get("pos", T)
                pos_start = pos_end - T
                if module.trained_pos == "rope":
                    cos, sin = _rope_at(dh, pos_start, T, device=q.device, dtype=q.dtype)
                    q = _apply_rope(q, cos, sin)
                k = kv_cache["k"]
                if module.group_size != 1:
                    k = k.repeat_interleave(module.group_size, dim=1)
                Tk = k.size(2)
                scores = (q @ k.transpose(-2, -1)).float() / math.sqrt(dh)
                if T == 1:
                    valid_scores = scores
                else:
                    mask = torch.full((T, Tk), float("-inf"), device=q.device).triu(Tk - T + 1)
                    valid_scores = scores + mask
                probs = F.softmax(valid_scores, dim=-1)
                logp = probs.clamp_min(1e-12).log()
                ent = -(probs * logp).sum(dim=-1)
                self.attn_entropy_sum[layer_idx] += ent.mean().item()
                self.attn_entropy_count[layer_idx] += 1
                self.attn_max_entropy[layer_idx] = max(self.attn_max_entropy[layer_idx], math.log(Tk))
        return hook

    def record_token(self, last_logits: torch.Tensor, token_id: int, token_str: str):
        with torch.no_grad():
            probs = F.softmax(last_logits.float(), dim=-1)
            p = probs[0, token_id].item()
            self.token_surprises.append(-math.log(max(p, 1e-12)))
            self.token_strings.append(token_str)

    def snapshot(self) -> dict:
        def avg(s, c):
            return [s[i] / c[i] if c[i] else None for i in range(len(s))]
        return {
            "moe_active": self.cfg.ffn == "moe",
            "attention_active": self.cfg.enable_attention,
            "attention_instrumented": self.instrument_attention,
            "norms_active": self.cfg.enable_norms,
            "n_layers": self.cfg.n_layers,
            "n_experts": self.cfg.n_experts if self.cfg.ffn == "moe" else 0,
            "top_k": self.cfg.top_k if self.cfg.ffn == "moe" else 0,
            "routing_counts": self.routing_counts,
            "routing_totals": self.routing_totals,
            "attn_entropy_mean": avg(self.attn_entropy_sum, self.attn_entropy_count),
            "attn_max_entropy": self.attn_max_entropy,
            "residual_norm_mean": avg(self.residual_norm_sum, self.residual_norm_count),
            "token_surprises": self.token_surprises,
            "token_strings": self.token_strings,
        }
