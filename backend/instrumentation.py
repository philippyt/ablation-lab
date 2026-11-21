from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from ablation.model import MoE

class Instrumenter:
    def __init__(self, model, runtime: dict | None = None):
        self.model = model
        self.cfg = model.cfg
        self.runtime = runtime or {}
        self.handles = []
        self.routing_counts = [[0] * self.cfg.n_experts for _ in range(self.cfg.n_layers)] if self.cfg.ffn == "moe" else []
        self.routing_totals = [0] * self.cfg.n_layers if self.cfg.ffn == "moe" else []
        self.attn_entropy_sum = [0.0] * self.cfg.n_layers
        self.attn_entropy_count = [0] * self.cfg.n_layers
        self.attn_logit_max_layer = [0.0] * self.cfg.n_layers
        self.attn_logit_seen = [False] * self.cfg.n_layers
        self.residual_norm_sum = [0.0] * self.cfg.n_layers
        self.residual_norm_count = [0] * self.cfg.n_layers
        self.head_norm_sum = [[0.0] * self.cfg.n_heads for _ in range(self.cfg.n_layers)]
        self.head_norm_count = [0] * self.cfg.n_layers
        self.token_surprises: list[float] = []
        self.token_strings: list[str] = []
        self.instrument_attention = self.runtime.get("enable_attention", self.cfg.enable_attention)

    def attach(self):
        for i, block in enumerate(self.model.blocks):
            if self.cfg.ffn == "moe" and isinstance(block.ffn, MoE):
                self.handles.append(block.ffn.gate.register_forward_hook(self._gate_hook(i)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

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

    def record_step(self, step_metrics: dict, last_logits: torch.Tensor, token_id: int, token_str: str) -> dict:
        with torch.no_grad():
            probs = F.softmax(last_logits.float(), dim=-1)
            p = probs[0, token_id].item()
            surprise = -math.log(max(p, 1e-12))
            next_ent = -(probs[0] * probs[0].clamp_min(1e-12).log()).sum().item()
            self.token_surprises.append(surprise)
            self.token_strings.append(token_str)

        lmax_dict: dict[int, float] = step_metrics.get("attn_logit_max", {})
        ent_dict: dict[int, float] = step_metrics.get("attn_entropy", {})
        rn_dict: dict[int, float] = step_metrics.get("resid_norm", {})
        hn_dict: dict[int, list[float]] = step_metrics.get("head_norms", {})

        step_lmax = 0.0
        for layer_idx, v in lmax_dict.items():
            if 0 <= layer_idx < self.cfg.n_layers:
                if not self.attn_logit_seen[layer_idx] or v > self.attn_logit_max_layer[layer_idx]:
                    self.attn_logit_max_layer[layer_idx] = v
                    self.attn_logit_seen[layer_idx] = True
                if v > step_lmax:
                    step_lmax = v

        for layer_idx, v in ent_dict.items():
            if 0 <= layer_idx < self.cfg.n_layers:
                self.attn_entropy_sum[layer_idx] += v
                self.attn_entropy_count[layer_idx] += 1

        final_rn = 0.0
        for layer_idx, v in rn_dict.items():
            if 0 <= layer_idx < self.cfg.n_layers:
                self.residual_norm_sum[layer_idx] += v
                self.residual_norm_count[layer_idx] += 1
                if layer_idx == self.cfg.n_layers - 1:
                    final_rn = v

        for layer_idx, norms in hn_dict.items():
            if 0 <= layer_idx < self.cfg.n_layers:
                row = self.head_norm_sum[layer_idx]
                for h_idx, nv in enumerate(norms):
                    if h_idx < len(row):
                        row[h_idx] += nv
                self.head_norm_count[layer_idx] += 1

        return {
            "lm": round(step_lmax, 3),
            "ent": round(next_ent, 3),
            "rn": round(final_rn, 3),
            "s": round(surprise, 3),
        }

    def snapshot(self) -> dict:
        def avg(s, c):
            return [round(s[i] / c[i], 4) if c[i] else None for i in range(len(s))]
        head_activity = [
            [round(self.head_norm_sum[l][h] / self.head_norm_count[l], 4) if self.head_norm_count[l] else 0.0 for h in range(self.cfg.n_heads)]
            for l in range(self.cfg.n_layers)
        ]
        return {
            "moe_active": self.cfg.ffn == "moe",
            "attention_active": bool(self.runtime.get("enable_attention", self.cfg.enable_attention)),
            "attention_instrumented": bool(self.instrument_attention),
            "norms_active": bool(self.runtime.get("enable_norms", self.cfg.enable_norms)),
            "n_layers": self.cfg.n_layers,
            "n_heads": self.cfg.n_heads,
            "n_experts": self.cfg.n_experts if self.cfg.ffn == "moe" else 0,
            "top_k": self.cfg.top_k if self.cfg.ffn == "moe" else 0,
            "routing_counts": self.routing_counts,
            "routing_totals": self.routing_totals,
            "attn_entropy_mean": avg(self.attn_entropy_sum, self.attn_entropy_count),
            "attn_logit_max_by_layer": [round(self.attn_logit_max_layer[i], 3) if self.attn_logit_seen[i] else None for i in range(self.cfg.n_layers)],
            "attn_max_entropy": [math.log(self.cfg.ctx)] * self.cfg.n_layers,
            "residual_norm_mean": avg(self.residual_norm_sum, self.residual_norm_count),
            "head_activity": head_activity,
            "token_surprises": [round(x, 3) for x in self.token_surprises],
            "token_strings": self.token_strings,
        }
