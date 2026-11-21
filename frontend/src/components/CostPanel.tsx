import {
  currentCost,
  previousCost,
  config,
  selectedGpu,
  setSelectedGpu,
  effectiveKvPerToken,
  effectiveKvFullCtx,
  kvDegradationActive,
} from "../state";
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
  decodeTokPerSec,
  arithmeticIntensityDecode,
  ridgePointFlopsPerByte,
  minRidgePoint,
  activationBytes,
} from "../lib/costs";
import { HelpHint } from "./HelpHint";
import { Consequences } from "./Consequences";

const TOTAL_PARAMS_HINT = "Total learned weights at the lab toy scale. Presets apply real architecture choices while keeping scale tiny so inference runs live.";
const ROOFLINE_HINT = "Arithmetic intensity is FLOPs done per byte read from memory for one decode step. Below the hardware ridge point decode is memory bound and capped by bandwidth.";
const FLOPS_HINT = "Approximation of 2 times active parameters per token. Every active parameter does roughly one multiply and one add per token.";

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
  const pct = (x: number) => (total === 0 ? "0%" : `${((x / total) * 100).toFixed(1)}%`);
  const weightsBytes = total * cur.dtype_bytes;
  const kvFullBytes = kvDegradationActive.value ? effectiveKvFullCtx.value : cur.kv_cache_full_ctx;
  const kvPerTok = kvDegradationActive.value ? effectiveKvPerToken.value : cur.kv_cache_per_token;
  const actBytes = activationBytes(cfg, cur.dtype_bytes, 1);
  const memTotal = weightsBytes + kvFullBytes + actBytes;
  const memPct = (x: number) => (memTotal === 0 ? "0%" : `${((x / memTotal) * 100).toFixed(0)}%`);

  const gpu = GPUS.find((g) => g.id === selectedGpu.value) ?? GPUS[0];
  const tps = decodeTokPerSec(cfg, gpu);
  const intensity = arithmeticIntensityDecode(cfg);
  const ridge = ridgePointFlopsPerByte(gpu);
  const minRidge = minRidgePoint();
  const memBoundEverywhere = intensity < minRidge;
  const memBoundHere = intensity < ridge;
  const fmtTps = (x: number) => (x >= 1e6 ? `${(x / 1e6).toFixed(2)}M` : x >= 1000 ? `${(x / 1000).toFixed(1)}K` : x.toFixed(0));

  return (
    <div class="cost-panel">
      <table class="cost-table">
        <tbody>
          <tr>
            <td>Total params <HelpHint text={TOTAL_PARAMS_HINT} /></td>
            <td>{fmtParams(cur.total_params)}</td>
            <Delta prev={prev?.total_params ?? null} cur={cur.total_params} lowerIsBetter={true} />
          </tr>
          <tr>
            <td>Active per token</td>
            <td>{fmtParams(cur.active_params_per_token)}</td>
            <Delta prev={prev?.active_params_per_token ?? null} cur={cur.active_params_per_token} lowerIsBetter={false} />
          </tr>
          <tr><td colSpan={3} class="sep">Parameter breakdown</td></tr>
          <tr><td>Attention</td><td>{fmtParams(attn)} ({pct(attn)})</td></tr>
          <tr><td>FFN</td><td>{fmtParams(ffn)} ({pct(ffn)})</td></tr>
          <tr><td>Norms</td><td>{fmtParams(norms)} ({pct(norms)})</td></tr>
          <tr><td>Embedding</td><td>{fmtParams(embed)} ({pct(embed)})</td></tr>
          <tr><td colSpan={3} class="sep">Compute and memory</td></tr>
          <tr>
            <td>FLOPs per token <HelpHint text={`${FLOPS_HINT} Active count is ${active.toLocaleString()}.`} /></td>
            <td>{fmtFlops(flops)}</td>
          </tr>
          <tr>
            <td>KV per token</td>
            <td>{fmtBytes(kvPerTok)}</td>
            <Delta prev={prev?.kv_cache_per_token ?? null} cur={cur.kv_cache_per_token} lowerIsBetter={true} />
          </tr>
          <tr>
            <td>KV at full context</td>
            <td>{fmtBytes(kvFullBytes)}</td>
            <Delta prev={prev?.kv_cache_full_ctx ?? null} cur={cur.kv_cache_full_ctx} lowerIsBetter={true} />
          </tr>
          <tr>
            <td>Approx memory</td>
            <td>{fmtBytes(memTotal)}</td>
            <Delta prev={prev?.approx_memory_bytes ?? null} cur={memTotal} lowerIsBetter={true} />
          </tr>
          <tr><td>FFN hidden</td><td>{cur.swiglu_hidden}</td></tr>
          <tr><td colSpan={3} class="sep">Memory share</td></tr>
          <tr><td>Weights</td><td>{fmtBytes(weightsBytes)} ({memPct(weightsBytes)})</td></tr>
          <tr><td>KV cache</td><td>{fmtBytes(kvFullBytes)} ({memPct(kvFullBytes)})</td></tr>
          <tr><td>Activations</td><td>{fmtBytes(actBytes)} ({memPct(actBytes)})</td></tr>
        </tbody>
      </table>

      <div class="roofline">
        <div class="roofline-head">
          <span>Decode roofline <HelpHint text={ROOFLINE_HINT} /></span>
          <select
            class="roofline-gpu"
            value={selectedGpu.value}
            onChange={(e) => setSelectedGpu((e.target as HTMLSelectElement).value)}
          >
            {GPUS.map((g) => (
              <option value={g.id}>{g.name}</option>
            ))}
          </select>
        </div>
        <p class="roofline-prose">
          {memBoundEverywhere ? (
            <><strong>Memory bound across all tiers.</strong> Intensity <strong>{intensity.toFixed(2)}</strong> is below the smallest ridge <strong>{minRidge.toFixed(0)}</strong> FLOPs/byte.</>
          ) : (
            <><strong>{memBoundHere ? "Memory bound" : "Compute bound"}</strong> on {gpu.name}. Intensity <strong>{intensity.toFixed(2)}</strong> vs ridge <strong>{ridge.toFixed(0)}</strong>.</>
          )}{" "}
          Estimated throughput on <strong>{gpu.name}</strong> is roughly <strong>{fmtTps(tps)} tok/s</strong>.
        </p>
      </div>

      <Consequences />
    </div>
  );
}
