import { currentCost, previousCost, config, selectedGpu, setSelectedGpu } from "../state";
import { fmtBytes, fmtParams, fmtPercent, relativeChange } from "../lib/format";
import {
  attentionParams,
  feedForwardParams,
  embeddingParams,
  normParamsTotal,
  flopsPerToken,
  totalParams,
  activeParamsPerToken,
  GPUS,
  decodeMsPerToken,
  decodeTokPerSec,
  arithmeticIntensityDecode,
  ridgePointFlopsPerByte,
  minRidgePoint,
  activationBytes,
  mlaLatentDim,
  mlaRopeDim,
} from "../lib/costs";
import { HelpHint } from "./HelpHint";
import { Consequences } from "./Consequences";

const TOTAL_PARAMS_HINT = "Total learned weights at the lab's tiny scale (d_model=128, n_layers=4). Presets change the architecture but keep this scale fixed, so the number shows the shape of each architecture, not the real model's true size. DeepSeek V3 ships at around 671B total.";
export const BASIS_HINT = "Every number on this panel is for the model actually trained and run in this lab at toy dimensions (d_model, n_layers, ctx, vocab as shown in the diagram). No projected real-scale numbers anywhere.";
const ROOFLINE_HINT = "Arithmetic intensity is FLOPs done per byte read from memory for one decode step. The ridge point is the hardware's peak compute divided by its peak memory bandwidth. Below the ridge point the chip sits idle waiting for memory and the workload is memory-bound. Above the ridge point it is compute-bound. Decode at small scale is almost always memory-bound, which is why bandwidth matters more than peak TFLOPs. The dropdown selects a VRAM tier with the bandwidth and TFLOPs typical of accelerators at that size. Ridge values are representative for the tier, not card-exact constants.";
const FLOPS_HINT = "Approximation: FLOPs/token ≈ 2 * active params. Every parameter does roughly one multiply and one add per token. Holds for both attention and FFN in this model. Reconciles to the displayed number exactly because that is how it is computed.";
const PREFILL_HINT = "Prefill (processing the prompt) is compute-bound. Each prompt token has to attend to all earlier prompt tokens, so total work scales roughly as O(seq²) with prompt length. Decode (one token at a time) is what the line above describes: memory-bound, sets steady-state tokens-per-second.";

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
  const active = activeParamsPerToken(cfg);
  const pct = (x: number) => total === 0 ? "0%" : `${((x / total) * 100).toFixed(1)}%`;
  const weightsBytes = total * cur.dtype_bytes;
  const kvFullBytes = cur.kv_cache_full_ctx;
  const actBytes = activationBytes(cfg, cur.dtype_bytes, 1);
  const memTotal = cur.approx_memory_bytes;
  const memPct = (x: number) => memTotal === 0 ? "0%" : `${((x / memTotal) * 100).toFixed(0)}%`;
  const kvFormula = cfg.attention_type === "mla"
    ? `${cfg.n_layers} L · (${mlaLatentDim(cfg)} latent + ${mlaRopeDim(cfg)} rope) · ${cur.dtype_bytes} B`
    : `2 · ${cfg.n_layers} L · ${cfg.n_kv_heads} kv-heads · ${cfg.head_dim} d_h · ${cur.dtype_bytes} B`;
  const kvFormulaPrefix = cfg.attention_type === "mla"
    ? "MLA caches one latent (plus the small rope vector) per token per layer, instead of full K and V per head."
    : "Standard attention caches K and V per kv-head per layer per token.";

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
          <tr>
            <td>FLOPs / token <HelpHint text={`${FLOPS_HINT} Here: 2 * ${active.toLocaleString()} active params = ${(2 * active).toLocaleString()}.`} /></td>
            <td>{fmtFlops(flops)} <span class="muted">≈ 2 * active</span></td>
          </tr>
          <tr>
            <td>KV / token <HelpHint text={`${kvFormulaPrefix} Here: ${fmtBytes(cur.kv_cache_per_token)} = ${kvFormula}.`} /></td>
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
          <tr><td colSpan={3} class="sep">memory breakdown</td></tr>
          <tr><td>weights</td><td>{fmtBytes(weightsBytes)} <span class="muted">({memPct(weightsBytes)})</span></td></tr>
          <tr><td>KV cache</td><td>{fmtBytes(kvFullBytes)} <span class="muted">({memPct(kvFullBytes)})</span></td></tr>
          <tr><td>activations</td><td>{fmtBytes(actBytes)} <span class="muted">({memPct(actBytes)})</span></td></tr>
        </tbody>
      </table>
      {(() => {
        const gpu = GPUS.find(g => g.id === selectedGpu.value) ?? GPUS[0];
        const tps = decodeTokPerSec(cfg, gpu);
        const intensity = arithmeticIntensityDecode(cfg);
        const ridge = ridgePointFlopsPerByte(gpu);
        const minRidge = minRidgePoint();
        const memBoundEverywhere = intensity < minRidge;
        const memBoundHere = intensity < ridge;
        const fmtTps = (x: number) => x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1000 ? `${(x / 1000).toFixed(1)}K` : x.toFixed(0);
        return (
          <div class="roofline">
            <div class="roofline-head">
              <span>decode</span>
              <HelpHint text={ROOFLINE_HINT} />
              <select class="roofline-gpu" value={selectedGpu.value} onChange={(e) => setSelectedGpu((e.target as HTMLSelectElement).value)} title="VRAM tier. Each tier uses bandwidth and FP16 TFLOPs typical (not card-exact) of accelerators at that memory size. The selector sets the ridge point, not a capacity check.">
                {GPUS.map(g => <option value={g.id}>{g.name}</option>)}
              </select>
            </div>
            <p class="roofline-prose">
              {memBoundEverywhere
                ? <><strong>memory-bound across all tiers</strong>. Intensity <strong>{intensity.toFixed(2)}</strong> ≪ smallest ridge <strong>{minRidge.toFixed(0)}</strong> FLOPs/byte.</>
                : <><strong>{memBoundHere ? "memory-bound" : "compute-bound"}</strong> on a {gpu.name} tier. Intensity <strong>{intensity.toFixed(2)}</strong> {memBoundHere ? "<" : ">"} this tier's representative ridge <strong>{ridge.toFixed(0)}</strong>.</>}
              {" "}On <strong>{gpu.name}</strong>: ~<strong>{fmtTps(tps)} tok/s</strong>, capped by {memBoundHere ? "bandwidth" : "compute"}.
            </p>
            <p class="roofline-prefill">
              prefill <HelpHint text={PREFILL_HINT} /> is the other regime: processing the prompt is compute-bound and scales roughly as O(seq²) in prompt length. The line above describes decode only.
            </p>
          </div>
        );
      })()}
      <Consequences />
    </>
  );
}
