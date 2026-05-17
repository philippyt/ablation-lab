import { instrumentation, config } from "../state";
import { CollapsiblePanel } from "./CollapsiblePanel";

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
  const maxEnt = Math.max(...data.attn_max_entropy);
  return (
    <div class="instr-entropy">
      <p class="instr-meta">mean entropy per layer (nats). max possible at this context ≈ {maxEnt.toFixed(2)}</p>
      <table class="instr-table">
        <tbody>
          {data.attn_entropy_mean.map((e, layer) => {
            const v = e ?? 0;
            const frac = maxEnt > 0 ? v / maxEnt : 0;
            return (
              <tr>
                <td><span class="instr-layer-tag">L{layer}</span></td>
                <td>
                  <div class="instr-meter">
                    <div class="instr-meter-fill" style={{ width: `${Math.min(frac, 1) * 100}%` }} />
                  </div>
                </td>
                <td class="instr-num">{e === null ? "—" : v.toFixed(2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ResidualNormPanel() {
  const cfg = config.value;
  const data = instrumentation.value;
  if (!data) {
    return <p class="instr-empty">Run inference to populate residual norm stats.</p>;
  }
  const values = data.residual_norm_mean.map((v) => v ?? 0);
  const maxV = Math.max(...values, 1e-6);
  return (
    <div class="instr-residual">
      <p class="instr-meta">
        L2 norm of residual stream at each layer's output, averaged across generated tokens.
        {!cfg.enable_norms && <span class="instr-warn"> norms ablated. expect inflated values.</span>}
      </p>
      <table class="instr-table">
        <tbody>
          {values.map((v, layer) => (
            <tr>
              <td><span class="instr-layer-tag">L{layer}</span></td>
              <td>
                <div class="instr-meter">
                  <div class="instr-meter-fill" style={{ width: `${(v / maxV) * 100}%` }} />
                </div>
              </td>
              <td class="instr-num">{v.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
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
      <p class="instr-meta">−log p(token) under the model. high = unexpected. mean: {mean.toFixed(2)} nats · max: {maxS.toFixed(2)}</p>
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
      <CollapsiblePanel title="Expert routing" defaultOpen={false}>
        <RoutingPanel />
      </CollapsiblePanel>
      <CollapsiblePanel title="Attention entropy" defaultOpen={false}>
        <AttnEntropyPanel />
      </CollapsiblePanel>
      <CollapsiblePanel title="Residual norm growth" defaultOpen={false}>
        <ResidualNormPanel />
      </CollapsiblePanel>
      <CollapsiblePanel title="Token surprise" defaultOpen={false}>
        <SurprisePanel />
      </CollapsiblePanel>
    </div>
  );
}
