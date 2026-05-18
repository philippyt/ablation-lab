import { instrumentation, config } from "../state";
import { CollapsiblePanel } from "./CollapsiblePanel";
import { HelpHint } from "./HelpHint";

const ROUTING_HINT = "When the FFN is MoE, each token picks a top-k subset of experts to run through. This panel counts how often each expert was picked, per layer. A healthy MoE spreads tokens across experts. If one expert handles most tokens, the router has collapsed and most of the experts are wasted capacity. Only meaningful when FFN is set to MoE.";
const ENTROPY_HINT = "How spread out attention is, per layer. Entropy is computed from the softmax over key positions for the current query. High entropy means attention is diffuse (looking at many positions roughly equally). Low entropy means attention is peaked (focused on a few positions). The max possible at this context length is shown for reference.";
const RESIDUAL_HINT = "L2 norm of the residual stream at each layer's output, averaged over generated tokens. The residual stream is the running vector that flows through the model and gets added to by each block. Norms typically grow gently with depth. A sudden spike usually means something blew up. Turning off normalization makes the values inflate sharply.";
const SURPRISE_HINT = "Per generated token, the value -log p(token) under the model's own distribution at that step. Low surprise (green-ish) means the model was confident in its choice. High surprise (red-ish) means it picked something it found unlikely. Useful for spotting where the model hesitated.";

function RoutingPanel() {
  const cfg = config.value;
  const data = instrumentation.value;
  if (cfg.ffn !== "moe") {
    return (
      <p class="instr-empty">No router. Dense FFN runs every weight for every token. Switch FFN to MoE to see expert routing.</p>
    );
  }
  if (!data || !data.moe_active) {
    return <p class="instr-empty">Run inference to populate routing stats.</p>;
  }
  const rows = [];
  for (let layer = 0; layer < data.n_layers; layer++) {
    const counts = data.routing_counts[layer] || [];
    const total = data.routing_totals[layer] || 0;
    if (total === 0) continue;
    const shares = counts.map((c) => c / total);
    const maxShare = Math.max(...shares);
    const collapsed = maxShare > 0.5;
    rows.push(
      <div class="instr-routing-layer">
        <div class="instr-routing-head">
          <span class="instr-layer-tag">L{layer}</span>
          {collapsed && <span class="instr-warn">router collapse: top expert {(maxShare * 100).toFixed(0)}%</span>}
        </div>
        <div class="instr-routing-bars">
          {shares.map((s, i) => (
            <div class="instr-routing-bar" title={`expert ${i}: ${(s * 100).toFixed(1)}%`}>
              <div class="instr-routing-bar-fill" style={{ width: `${s * 100}%` }} />
              <span class="instr-routing-bar-label">e{i}</span>
              <span class="instr-routing-bar-pct">{(s * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>,
    );
  }
  return (
    <div class="instr-routing">
      <p class="instr-meta">top-{data.top_k} of {data.n_experts} experts per token</p>
      {rows}
    </div>
  );
}

function niceTicks(min: number, max: number, target = 4): { ticks: number[]; step: number } {
  const range = max - min;
  if (range <= 0) return { ticks: [min], step: 1 };
  const rough = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 1e-9; v += step) {
    ticks.push(Number(v.toPrecision(10)));
  }
  return { ticks, step };
}

function decimalsFor(step: number): number {
  if (step >= 1) return 0;
  if (step >= 0.1) return 1;
  if (step >= 0.01) return 2;
  if (step >= 0.001) return 3;
  return 4;
}

function LayerPlot({ values, refLine, refLabel }: { values: (number | null)[]; refLine?: number; refLabel?: string }) {
  const n = values.length;
  if (n === 0) return null;
  const valid = values.filter((v): v is number => v !== null);
  if (valid.length === 0) return null;

  const dataMin = Math.min(...valid);
  const dataMax = Math.max(...valid);
  const spread = Math.max(dataMax - dataMin, dataMax * 0.05, 1e-6);

  const includeRefInPlot = refLine !== undefined && dataMax >= 0.25 * refLine;

  let yMin = Math.max(0, dataMin - spread * 0.15);
  let yMax = dataMax + spread * 0.25;
  if (includeRefInPlot && refLine !== undefined) {
    yMax = Math.max(yMax, refLine * 1.06);
  }
  if (yMax <= yMin) yMax = yMin + 1;

  const { ticks, step } = niceTicks(yMin, yMax, 4);
  if (ticks.length > 0) {
    yMin = Math.min(yMin, ticks[0]);
    yMax = Math.max(yMax, ticks[ticks.length - 1]);
  }
  const dec = decimalsFor(step);
  const fmtT = (v: number) => v.toFixed(dec);

  const W = 320;
  const H = 140;
  const ML = 30;
  const MR = 12;
  const MT = 10;
  const MB = 22;
  const PW = W - ML - MR;
  const PH = H - MT - MB;
  const xFor = (i: number) => ML + (n === 1 ? PW / 2 : (i / (n - 1)) * PW);
  const yFor = (v: number) => MT + PH - ((v - yMin) / (yMax - yMin)) * PH;
  const segPts = values.map((v, i) => v === null ? null : `${xFor(i)},${yFor(v)}`).filter(Boolean).join(" ");

  const refInPlot = refLine !== undefined && refLine >= yMin && refLine <= yMax;
  const refAbove = refLine !== undefined && refLine > yMax;

  return (
    <div class="instr-plot-wrap">
      <svg class="instr-plot" viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {ticks.map((t) => (
          <g>
            <line class="instr-plot-tick-mark" x1={ML - 4} x2={ML} y1={yFor(t)} y2={yFor(t)} />
            <text class="instr-plot-tick-label" x={ML - 6} y={yFor(t) + 3} text-anchor="end">{fmtT(t)}</text>
          </g>
        ))}
        {refInPlot && refLine !== undefined && (
          <g>
            <line class="instr-plot-ref" x1={ML} x2={W - MR} y1={yFor(refLine)} y2={yFor(refLine)} stroke-dasharray="4 3" />
            <text class="instr-plot-ref-text" x={W - MR - 3} y={yFor(refLine) - 4} text-anchor="end">{refLabel ?? `ceiling ${refLine.toFixed(2)}`}</text>
          </g>
        )}
        <polyline class="instr-plot-line" points={segPts} fill="none" />
        {values.map((v, i) => v === null ? null : (
          <circle class="instr-plot-marker" cx={xFor(i)} cy={yFor(v)} r="3.2" />
        ))}
        {values.map((_, i) => (
          <g>
            <line class="instr-plot-tick-mark" x1={xFor(i)} x2={xFor(i)} y1={H - MB} y2={H - MB + 4} />
            <text class="instr-plot-xlabel" x={xFor(i)} y={H - MB + 14} text-anchor="middle">L{i}</text>
          </g>
        ))}
        <line class="instr-plot-axis" x1={ML} x2={ML} y1={MT} y2={H - MB} />
        <line class="instr-plot-axis" x1={ML} x2={W - MR} y1={H - MB} y2={H - MB} />
      </svg>
      {refLine !== undefined && refAbove && (
        <div class="instr-plot-ref-off">{refLabel ?? `ceiling ${refLine.toFixed(2)}`} ↑ (above plot)</div>
      )}
    </div>
  );
}

function LayerValues({ values, fmt }: { values: (number | null)[]; fmt: (v: number) => string }) {
  return (
    <div class="instr-spark-values">
      {values.map((v, i) => (
        <span class="instr-spark-cell">
          <span class="instr-layer-tag">L{i}</span>
          <span class="instr-num-inline">{v === null ? "—" : fmt(v)}</span>
        </span>
      ))}
    </div>
  );
}

function AttnEntropyPanel() {
  const cfg = config.value;
  const data = instrumentation.value;
  if (!cfg.enable_attention) {
    return <p class="instr-empty">Attention sublayer is disabled. No attention distribution exists to measure.</p>;
  }
  if (cfg.attention_type === "mla") {
    return <p class="instr-empty">MLA attention. Entropy capture not implemented for the latent-decompressed path.</p>;
  }
  if (!data || !data.attention_instrumented) {
    return <p class="instr-empty">Run inference to populate attention entropy stats.</p>;
  }
  const ceiling = Math.log(cfg.ctx);
  return (
    <div class="instr-entropy">
      <p class="instr-meta">mean entropy per layer (nats). dashed line = ln(ctx) ceiling.</p>
      <LayerPlot values={data.attn_entropy_mean} refLine={ceiling} refLabel={`ceiling ${ceiling.toFixed(2)}`} />
      <LayerValues values={data.attn_entropy_mean} fmt={(v) => v.toFixed(2)} />
    </div>
  );
}

function ResidualNormPanel() {
  const cfg = config.value;
  const data = instrumentation.value;
  if (!data) {
    return <p class="instr-empty">Run inference to populate residual norm stats.</p>;
  }
  const values = data.residual_norm_mean;
  const maxV = Math.max(...values.map(v => v ?? 0), 1e-6);
  return (
    <div class="instr-residual">
      <p class="instr-meta">
        L2 norm of the residual stream at each layer's output, averaged across generated tokens.
        {!cfg.enable_norms && <span class="instr-warn"> norms ablated. expect inflated values.</span>}
      </p>
      <LayerPlot values={values} />
      <LayerValues values={values} fmt={(v) => v.toFixed(2)} />
    </div>
  );
}

function escapeChar(c: string): string {
  if (c === " ") return "·";
  if (c === "\n") return "↵";
  if (c === "\t") return "→";
  if (c.charCodeAt(0) < 32) return "?";
  return c;
}

function surpriseColor(s: number, maxS: number): string {
  if (maxS <= 0) return "transparent";
  const t = Math.min(1, s / maxS);
  const r = Math.round(255 * t);
  const g = Math.round(220 * (1 - t));
  return `rgba(${r}, ${g}, 80, ${0.15 + 0.55 * t})`;
}

function SurprisePanel() {
  const data = instrumentation.value;
  if (!data) {
    return <p class="instr-empty">Run inference to populate per-token surprise.</p>;
  }
  const ss = data.token_surprises;
  const strs = data.token_strings;
  if (!ss || ss.length === 0) {
    return <p class="instr-empty">No tokens generated this run.</p>;
  }
  const maxS = Math.max(...ss);
  const mean = ss.reduce((a, b) => a + b, 0) / ss.length;
  return (
    <div class="instr-surprise">
      <p class="instr-meta">−log p(token) under the model. high = unexpected. mean: {mean.toFixed(2)} nats. max: {maxS.toFixed(2)}</p>
      <div class="instr-surprise-stream">
        {strs.map((c, i) => (
          <span class="instr-surprise-char" style={{ background: surpriseColor(ss[i], maxS) }} title={`-log p = ${ss[i].toFixed(2)}`}>
            {escapeChar(c)}
          </span>
        ))}
      </div>
    </div>
  );
}

export function InstrumentationPanels() {
  return (
    <div class="diagnostics-col">
      <CollapsiblePanel title="Expert routing" defaultOpen={false} action={<HelpHint text={ROUTING_HINT} />}>
        <RoutingPanel />
      </CollapsiblePanel>
      <CollapsiblePanel title="Attention entropy" defaultOpen={false} action={<HelpHint text={ENTROPY_HINT} />}>
        <AttnEntropyPanel />
      </CollapsiblePanel>
      <CollapsiblePanel title="Residual norm growth" defaultOpen={false} action={<HelpHint text={RESIDUAL_HINT} />}>
        <ResidualNormPanel />
      </CollapsiblePanel>
      <CollapsiblePanel title="Token surprise" defaultOpen={false} action={<HelpHint text={SURPRISE_HINT} />}>
        <SurprisePanel />
      </CollapsiblePanel>
    </div>
  );
}
