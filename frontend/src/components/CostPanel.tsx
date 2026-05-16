import { currentCost, previousCost, config } from "../state";
import { fmtBytes, fmtParams, fmtPercent, relativeChange } from "../lib/format";
import {
  attentionParams,
  feedForwardParams,
  embeddingParams,
  normParamsTotal,
  flopsPerToken,
  totalParams,
} from "../lib/costs";
import { HelpHint } from "./HelpHint";
import { Consequences } from "./Consequences";

const TOTAL_PARAMS_HINT = "Every learned weight in the current architecture at the lab's tiny scale (d_model=128, n_layers=4). Presets flip architectural switches but keep this scale fixed, so the number reflects the shape of each architecture, not the real model's true size (e.g. DeepSeek V3 ships at ~671B total).";

function Delta({ prev, cur, lowerIsBetter }: { prev: number | null; cur: number; lowerIsBetter: boolean }) {
  if (prev === null || prev === cur) return null;
  const rel = relativeChange(prev, cur);
  const up = rel > 0;
  const cls = (up && lowerIsBetter) || (!up && !lowerIsBetter) ? "delta up" : "delta down";
  return <td class={cls}>{fmtPercent(rel)}</td>;
}

function fmtFlops(n: number): string {
  if (n < 1e6) return `${(n / 1e3).toFixed(1)}K`;
  if (n < 1e9) return `${(n / 1e6).toFixed(2)}M`;
  if (n < 1e12) return `${(n / 1e9).toFixed(2)}G`;
  return `${(n / 1e12).toFixed(2)}T`;
}

export function CostPanel() {
  const cur = currentCost.value;
  const prev = previousCost.value;
  const cfg = config.value;
  const total = totalParams(cfg);
  const attn = attentionParams(cfg);
  const ffn = feedForwardParams(cfg);
  const embed = embeddingParams(cfg);
  const norms = normParamsTotal(cfg);
  const flops = flopsPerToken(cfg);
  const pct = (x: number) => total === 0 ? "0%" : `${((x / total) * 100).toFixed(1)}%`;

  return (
    <>
      <table class="cost-table">
        <tbody>
          <tr>
            <td>total params <HelpHint text={TOTAL_PARAMS_HINT} /></td>
            <td>{fmtParams(cur.total_params)}</td>
            <Delta prev={prev?.total_params ?? null} cur={cur.total_params} lowerIsBetter={true} />
          </tr>
          <tr>
            <td>active / token</td>
            <td>{fmtParams(cur.active_params_per_token)}</td>
            <Delta prev={prev?.active_params_per_token ?? null} cur={cur.active_params_per_token} lowerIsBetter={false} />
          </tr>
          <tr><td colSpan={3} class="sep">params breakdown</td></tr>
          <tr><td>attention</td><td>{fmtParams(attn)} <span class="muted">({pct(attn)})</span></td></tr>
          <tr><td>ffn</td><td>{fmtParams(ffn)} <span class="muted">({pct(ffn)})</span></td></tr>
          <tr><td>norms</td><td>{fmtParams(norms)} <span class="muted">({pct(norms)})</span></td></tr>
          <tr><td>embedding</td><td>{fmtParams(embed)} <span class="muted">({pct(embed)})</span></td></tr>
          <tr><td colSpan={3} class="sep">compute & memory</td></tr>
          <tr><td>FLOPs / token</td><td>{fmtFlops(flops)}</td></tr>
          <tr>
            <td>KV / token</td>
            <td>{fmtBytes(cur.kv_cache_per_token)}</td>
            <Delta prev={prev?.kv_cache_per_token ?? null} cur={cur.kv_cache_per_token} lowerIsBetter={true} />
          </tr>
          <tr>
            <td>KV @ full ctx</td>
            <td>{fmtBytes(cur.kv_cache_full_ctx)}</td>
            <Delta prev={prev?.kv_cache_full_ctx ?? null} cur={cur.kv_cache_full_ctx} lowerIsBetter={true} />
          </tr>
          <tr>
            <td>approx memory</td>
            <td>{fmtBytes(cur.approx_memory_bytes)}</td>
            <Delta prev={prev?.approx_memory_bytes ?? null} cur={cur.approx_memory_bytes} lowerIsBetter={true} />
          </tr>
          <tr><td>SwiGLU hidden</td><td>{cur.swiglu_hidden}</td></tr>
        </tbody>
      </table>
      <Consequences />
    </>
  );
}
