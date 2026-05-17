import { useState } from "preact/hooks";
import { config, cacheState } from "../state";
import { kvCachePerToken, kvCacheFullCtx, attentionKind, mlaLatentDim, mlaRopeDim } from "../lib/costs";
import { fmtBytes } from "../lib/format";
import { attnModalOpen } from "./AttentionGrid";
import type { ArchConfig } from "../types";

function kvCounterfactuals(cfg: ArchConfig): { label: string; bytes: number; ratio: number }[] {
  const baseBytes = kvCachePerToken(cfg);
  const out: { label: string; bytes: number; ratio: number }[] = [];
  if (cfg.attention_type !== "mla") {
    if (cfg.n_kv_heads !== cfg.n_heads) {
      out.push({ label: `if MHA (n_kv=${cfg.n_heads})`, bytes: kvCachePerToken({ ...cfg, n_kv_heads: cfg.n_heads }), ratio: 0 });
    }
    if (cfg.n_heads >= 4 && cfg.n_kv_heads > 2) {
      out.push({ label: "if GQA (n_kv=2)", bytes: kvCachePerToken({ ...cfg, n_kv_heads: 2 }), ratio: 0 });
    }
    if (cfg.n_kv_heads !== 1) {
      out.push({ label: "if MQA (n_kv=1)", bytes: kvCachePerToken({ ...cfg, n_kv_heads: 1 }), ratio: 0 });
    }
    out.push({ label: "if MLA", bytes: kvCachePerToken({ ...cfg, attention_type: "mla" }), ratio: 0 });
  } else {
    out.push({ label: `if MHA (n_kv=${cfg.n_heads})`, bytes: kvCachePerToken({ ...cfg, attention_type: "standard", n_kv_heads: cfg.n_heads }), ratio: 0 });
    out.push({ label: "if MQA (n_kv=1)", bytes: kvCachePerToken({ ...cfg, attention_type: "standard", n_kv_heads: 1 }), ratio: 0 });
  }
  for (const r of out) r.ratio = baseBytes === 0 ? 0 : r.bytes / baseBytes;
  return out.filter(r => r.bytes !== baseBytes);
}

export function KVCacheState() {
  const cfg = config.value;
  const s = cacheState.value;
  const perToken = kvCachePerToken(cfg);
  const full = kvCacheFullCtx(cfg);
  const ctx = cfg.ctx;
  const pos = Math.min(s.positions, ctx);
  const evicted = Math.max(0, s.positions - ctx);
  const used = pos * perToken;
  const pct = ctx === 0 ? 0 : pos / ctx;
  const promptPct = ctx === 0 ? 0 : Math.min(s.prompt_tokens, ctx) / ctx;
  const [showBreakdown, setShowBreakdown] = useState(false);
  const counterfactuals = kvCounterfactuals(cfg);
  const kind = attentionKind(cfg);
  const sliding = evicted > 0;

  return (
    <div class="panel kv-state">
      <h2>KV cache state</h2>

      <div class="kv-bar" title="The light band shows how much of the cache the prompt fills. The full band shows the prompt plus every token generated so far. Once the cache fills, oldest entries are evicted to make room for new ones (sliding window).">
        {s.status !== "idle" && (
          <div class="kv-bar-prompt" style={{ width: `${promptPct * 100}%` }} />
        )}
        <div class={`kv-bar-fill ${sliding ? "sliding" : ""}`} style={{ width: `${pct * 100}%` }} />
        <div class="kv-bar-pct">{(pct * 100).toFixed(1)}%</div>
      </div>

      <table class="kv-state-table">
        <tbody>
          <tr>
            <td>positions</td>
            <td>
              {pos} / {ctx}
              {sliding && <span class="kv-evicted"> · +{evicted} evicted</span>}
            </td>
          </tr>
          <tr>
            <td>bytes used</td>
            <td>{fmtBytes(used)} / {fmtBytes(full)}</td>
          </tr>
          <tr>
            <td>per token</td>
            <td>{fmtBytes(perToken)}</td>
          </tr>
        </tbody>
      </table>

      <p class="kv-state-hint">
        <span class="link" onClick={() => setShowBreakdown(v => !v)}>
          {showBreakdown ? "hide" : "what makes it"} {fmtBytes(perToken)}?
        </span>
        <button class="ghost ghost-sm" onClick={() => (attnModalOpen.value = true)}>
          show attention pattern
        </button>
      </p>

      {showBreakdown && (
        <div class="kv-breakdown">
          <div class="kv-eq-result">per token: <strong>{fmtBytes(perToken)}</strong></div>
          {cfg.attention_type === "mla" ? (
            <table class="kv-eq">
              <tbody>
                <tr class="kv-eq-nums">
                  <td>{cfg.n_layers}</td><td class="dot">·</td>
                  <td>({mlaLatentDim(cfg)}</td><td class="dot">+</td>
                  <td>{mlaRopeDim(cfg)})</td><td class="dot">·</td>
                  <td>2 B</td>
                </tr>
                <tr class="kv-eq-labels">
                  <td>layers</td><td></td>
                  <td>latent</td><td></td>
                  <td>rope</td><td></td>
                  <td>bf16</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <table class="kv-eq">
              <tbody>
                <tr class="kv-eq-nums">
                  <td>2</td><td class="dot">·</td>
                  <td>{cfg.n_layers}</td><td class="dot">·</td>
                  <td>{cfg.n_kv_heads}</td><td class="dot">·</td>
                  <td>{cfg.head_dim}</td><td class="dot">·</td>
                  <td>2 B</td>
                </tr>
                <tr class="kv-eq-labels">
                  <td>K,V</td><td></td>
                  <td>layers</td><td></td>
                  <td>kv-heads</td><td></td>
                  <td>d_h</td><td></td>
                  <td>bf16</td>
                </tr>
              </tbody>
            </table>
          )}
          <div class="kv-breakdown-current">
            current: <strong>{kind.toUpperCase()}</strong>
            {cfg.attention_type === "mla"
              ? <> caching a {mlaLatentDim(cfg)}-dim latent plus {mlaRopeDim(cfg)}-dim rope vector per layer per token</>
              : <> with <strong>{cfg.n_kv_heads}</strong> kv-head{cfg.n_kv_heads === 1 ? "" : "s"}</>}
          </div>
          {counterfactuals.length > 0 && (
            <table class="kv-counterfactual">
              <tbody>
                {counterfactuals.map(c => (
                  <tr>
                    <td>{c.label}</td>
                    <td>{fmtBytes(c.bytes)}</td>
                    <td>{c.ratio < 1 ? `${(1 / c.ratio).toFixed(1)} times smaller` : `${c.ratio.toFixed(1)} times larger`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}