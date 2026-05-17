import { config, cacheState, openExplainerId } from "../state";
import { kvCachePerToken, kvCacheFullCtx } from "../lib/costs";
import { fmtBytes } from "../lib/format";
import { attnModalOpen } from "./AttentionGrid";

export function KVCacheState() {
  const cfg = config.value;
  const s = cacheState.value;
  const perToken = kvCachePerToken(cfg);
  const full = kvCacheFullCtx(cfg);
  const ctx = cfg.ctx;
  const pos = Math.min(s.positions, ctx);
  const used = pos * perToken;
  const pct = ctx === 0 ? 0 : pos / ctx;
  const promptPct = ctx === 0 ? 0 : s.prompt_tokens / ctx;

  return (
    <div class="panel kv-state">
      <h2>KV cache state</h2>

      <div class="kv-bar" title="The light band shows how much of the cache the prompt fills. The full band shows the prompt plus every token generated so far.">
        {s.status !== "idle" && (
          <div class="kv-bar-prompt" style={{ width: `${promptPct * 100}%` }} />
        )}
        <div class="kv-bar-fill" style={{ width: `${pct * 100}%` }} />
        <div class="kv-bar-pct">{(pct * 100).toFixed(1)}%</div>
      </div>

      <table class="kv-state-table">
        <tbody>
          <tr>
            <td>positions</td>
            <td>{pos} / {ctx}</td>
          </tr>
          <tr>
            <td>bytes used</td>
            <td>{fmtBytes(used)} / {fmtBytes(full)}</td>
          </tr>
          <tr>
            <td>per token</td>
            <td>
              {fmtBytes(perToken)}
              <span class="kv"> = 2 · {cfg.n_layers} L · {cfg.n_kv_heads} kv-heads · {cfg.head_dim} d_h · 2 B</span>
            </td>
          </tr>
        </tbody>
      </table>

      <p class="kv-state-hint">
        <span
          class="link"
          onClick={() => (openExplainerId.value = "attention." + (cfg.n_kv_heads === cfg.n_heads ? "mha" : cfg.n_kv_heads === 1 ? "mqa" : "gqa"))}
        >
          why is the per-token cost {fmtBytes(perToken)}?
        </span>
        <button class="ghost ghost-sm" onClick={() => (attnModalOpen.value = true)}>
          show attention pattern
        </button>
      </p>
    </div>
  );
}
