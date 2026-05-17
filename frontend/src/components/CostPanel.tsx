import { currentCost, previousCost, config, selectedGpu, setSelectedGpu } from "../state";
import { fmtBytes, fmtParams, fmtPercent, relativeChange } from "../lib/format";
import {
  attentionParams,
  feedForwardParams,
  embeddingParams,
  normParamsTotal,
  flopsPerToken,
  totalParams,
  GPUS,
  decodeMsPerToken,
  decodeTokPerSec,
  arithmeticIntensityDecode,
  ridgePointFlopsPerByte,
  minGpuThatFits,
} from "../lib/costs";
import { HelpHint } from "./HelpHint";
import { Consequences } from "./Consequences";

const TOTAL_PARAMS_HINT = "Total learned weights at the lab's tiny scale (d_model=128, n_layers=4). Presets change the architecture but keep this scale fixed, so the number shows the shape of each architecture, not the real model's true size. DeepSeek V3 ships at around 671B total.";
export const BASIS_HINT = "Every number on this panel is for the model actually trained and run in this lab at toy dimensions (d_model, n_layers, ctx, vocab as shown in the diagram). No projected real-scale numbers anywhere.";
const ROOFLINE_HINT = "Arithmetic intensity is FLOPs done per byte read from memory for one decode step. The ridge point is the hardware's peak compute divided by its peak memory bandwidth. Below the ridge point the chip sits idle waiting for memory and the workload is memory-bound. Above the ridge point it is compute-bound. Decode at small scale is almost always memory-bound, which is why bandwidth matters more than peak TFLOPs. The dropdown selects a VRAM tier with the bandwidth and TFLOPs typical of accelerators at that size.";

export function CostBasisHint() {
  const cfg = config.value;
  return <HelpHint text={`${BASIS_HINT} Current toy dimensions: d_model=${cfg.d_model}, n_layers=${cfg.n_layers}, ctx=${cfg.ctx}, vocab=${cfg.vocab_size}.`} />;
}

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
          <tr><td>FFN hidden</td><td>{cur.swiglu_hidden}</td></tr>
        </tbody>
      </table>
      {(() => {
        const gpu = GPUS.find(g => g.id === selectedGpu.value) ?? GPUS[0];
        const tps = decodeTokPerSec(cfg, gpu);
        const intensity = arithmeticIntensityDecode(cfg);
        const ridge = ridgePointFlopsPerByte(gpu);
        const memBound = intensity < ridge;
        const fmtTps = (x: number) => x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1000 ? `${(x / 1000).toFixed(1)}K` : x.toFixed(0);
        return (
          <div class="roofline">
            <div class="roofline-head">
              <span>decode</span>
              <HelpHint text={ROOFLINE_HINT} />
              <select class="roofline-gpu" value={selectedGpu.value} onChange={(e) => setSelectedGpu((e.target as HTMLSelectElement).value)} title="VRAM tier. Each tier uses bandwidth and FP16 TFLOPs typical of accelerators at that memory size. The selector sets the ridge point, not a capacity check.">
                {GPUS.map(g => <option value={g.id}>{g.name}</option>)}
              </select>
            </div>
            <p class="roofline-prose">
              <strong>{memBound ? "memory-bound" : "compute-bound"}</strong> on a {gpu.name} tier. Intensity <strong>{intensity.toFixed(2)}</strong> {memBound ? "<" : ">"} ridge <strong>{ridge.toFixed(0)}</strong> FLOPs/byte. ~<strong>{fmtTps(tps)} tok/s</strong>, capped by {memBound ? "bandwidth" : "compute"}.
            </p>
          </div>
        );
      })()}
      <Consequences />
    </>
  );
}
