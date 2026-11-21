import { useState } from "preact/hooks";
import {
  config,
  cacheState,
  kvDegradation,
  updateKV,
  resetKV,
  kvDegradationActive,
  effectiveKvPerToken,
  effectiveKvFullCtx,
} from "../state";
import { kvCachePerToken, attentionKind, mlaLatentDim, mlaRopeDim } from "../lib/costs";
import { fmtBytes } from "../lib/format";
import { attnModalOpen } from "./AttentionGrid";
import type { KVQuantKind, KVSinkMode } from "../types";

const QUANT_OPTS: Array<{ value: KVQuantKind; label: string }> = [
  { value: "none", label: "BF16" },
  { value: "int8", label: "INT8" },
  { value: "int4", label: "INT4" },
  { value: "int2", label: "INT2" },
];

const WINDOW_OPTS: Array<{ value: number | null; label: string }> = [
  { value: null, label: "Full" },
  { value: 128, label: "128" },
  { value: 64, label: "64" },
  { value: 32, label: "32" },
  { value: 16, label: "16" },
];

const SINK_OPTS: Array<{ value: KVSinkMode; label: string }> = [
  { value: "normal", label: "Normal" },
  { value: "keep_sinks", label: "Keep 4 sinks" },
  { value: "drop_sinks", label: "Drop 4 sinks" },
];

const DROP_OPTS: Array<{ value: number; label: string }> = [
  { value: 0, label: "0%" },
  { value: 0.15, label: "15%" },
  { value: 0.3, label: "30%" },
  { value: 0.5, label: "50%" },
];

export function KVCacheState() {
  const cfg = config.value;
  const s = cacheState.value;
  const kv = kvDegradation.value;
  const basePerToken = kvCachePerToken(cfg);
  const effPerToken = effectiveKvPerToken.value;
  const effFull = effectiveKvFullCtx.value;
  const cap = kv.window ? Math.min(cfg.ctx, kv.window) : cfg.ctx;
  const pos = Math.min(s.positions, cap);
  const evicted = Math.max(0, s.positions - cap);
  const used = pos * effPerToken;
  const pct = cap === 0 ? 0 : pos / cap;
  const promptPct = cap === 0 ? 0 : Math.min(s.prompt_tokens, cap) / cap;
  const [showFormula, setShowFormula] = useState(false);
  const kind = attentionKind(cfg);
  const isActive = kvDegradationActive.value;

  return (
    <div class="kv-lab">
      <div class="sub-head">
        <div class="sub-title">
          <strong>KV cache and degradation</strong>
          <span class="pill-count">{kind.toUpperCase()}</span>
        </div>
        <div class="quick-chips">
          <button class="chip-btn" onClick={() => (attnModalOpen.value = true)}>
            Attention grid
          </button>
          {isActive && (
            <button class="chip-btn chip-reset" onClick={resetKV}>Reset</button>
          )}
        </div>
      </div>

      <div class="kv-bar">
        {s.status !== "idle" && (
          <div class="kv-bar-prompt" style={{ width: `${promptPct * 100}%` }} />
        )}
        <div class={`kv-bar-fill ${evicted > 0 ? "sliding" : ""}`} style={{ width: `${pct * 100}%` }} />
        <div class="kv-bar-pct">
          {pos} / {cap} slots {evicted > 0 ? `(${evicted} evicted)` : ""}
        </div>
      </div>

      <div class="kv-metrics-row">
        <div class="kv-metric">
          <span class="kv-m-lbl">Per token</span>
          <strong class="kv-m-val">{fmtBytes(effPerToken)}</strong>
          {effPerToken !== basePerToken && (
            <span class="kv-m-sub">from {fmtBytes(basePerToken)}</span>
          )}
        </div>
        <div class="kv-metric">
          <span class="kv-m-lbl">Used now</span>
          <strong class="kv-m-val">{fmtBytes(used)}</strong>
        </div>
        <div class="kv-metric">
          <span class="kv-m-lbl">Window cap</span>
          <strong class="kv-m-val">{fmtBytes(effFull)}</strong>
        </div>
      </div>

      <div class="kv-controls">
        <div class="kv-ctrl-group">
          <span class="kv-ctrl-lbl">Precision</span>
          <div class="seg-row">
            {QUANT_OPTS.map((o) => (
              <button
                key={o.value}
                class={`seg-btn ${kv.quant === o.value ? "is-on" : ""}`}
                onClick={() => updateKV({ quant: o.value })}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>

        <div class="kv-ctrl-group">
          <span class="kv-ctrl-lbl">Sliding window</span>
          <div class="seg-row">
            {WINDOW_OPTS.map((o) => (
              <button
                key={String(o.value)}
                class={`seg-btn ${kv.window === o.value ? "is-on" : ""}`}
                onClick={() => updateKV({ window: o.value })}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>

        <div class="kv-ctrl-group">
          <span class="kv-ctrl-lbl">Attention sinks</span>
          <div class="seg-row">
            {SINK_OPTS.map((o) => (
              <button
                key={o.value}
                class={`seg-btn ${kv.sink_mode === o.value ? "is-on" : ""}`}
                onClick={() => updateKV({ sink_mode: o.value })}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>

        <div class="kv-ctrl-group">
          <span class="kv-ctrl-lbl">Random KV drop</span>
          <div class="seg-row">
            {DROP_OPTS.map((o) => (
              <button
                key={o.value}
                class={`seg-btn ${kv.drop_prob === o.value ? "is-on" : ""}`}
                onClick={() => updateKV({ drop_prob: o.value })}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div class="kv-formula-toggle">
        <button class="link-btn" onClick={() => setShowFormula((v) => !v)}>
          {showFormula ? "Hide formula" : "Show bytes per token formula"}
        </button>
      </div>

      {showFormula && (
        <div class="kv-formula-box">
          {cfg.attention_type === "mla" ? (
            <span>
              {cfg.n_layers} layers × ({mlaLatentDim(cfg)} latent + {mlaRopeDim(cfg)} rope) × {kv.quant === "none" ? "2 B" : kv.quant.toUpperCase()} = <strong>{fmtBytes(effPerToken)}</strong>
            </span>
          ) : (
            <span>
              2 (K,V) × {cfg.n_layers} layers × {cfg.n_kv_heads} kv heads × {cfg.head_dim} head dim × {kv.quant === "none" ? "2 B" : kv.quant.toUpperCase()} = <strong>{fmtBytes(effPerToken)}</strong>
            </span>
          )}
        </div>
      )}
    </div>
  );
}
