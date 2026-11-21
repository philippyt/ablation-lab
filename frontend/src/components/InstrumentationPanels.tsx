import { instrumentation, config, logitLens } from "../state";

function escapeVisible(ch: string): string {
  if (ch === " ") return "space";
  if (ch === "\n") return "newline";
  if (ch === "\t") return "tab";
  return ch;
}

function LogitLensCard() {
  const lens = logitLens.value;
  return (
    <div class="internal-card lens-card">
      <div class="internal-head">
        <strong>Logit lens</strong>
        <span class="internal-ref">
          {lens
            ? `Step ${lens.stepIndex} chose ${escapeVisible(lens.chosenChar)}`
            : "Click any generated char"}
        </span>
      </div>
      {!lens ? (
        <p class="panel-note" style="margin-bottom:0;">
          Click any generated character in the output box to project each layer residual stream directly into next token probabilities.
        </p>
      ) : lens.loading ? (
        <p class="panel-note" style="margin-bottom:0;">Computing layer projections...</p>
      ) : lens.error ? (
        <p class="panel-note is-err" style="margin-bottom:0;">{lens.error}</p>
      ) : (
        <div class="lens-rows">
          {lens.stages.map((st) => (
            <div class="lens-row" key={st.stage}>
              <span class="lb-tag">{st.stage}</span>
              <div class="lens-pills">
                {st.top.map((item, i) => {
                  const hit = item.ch === lens.chosenChar;
                  return (
                    <span class={`lens-pill ${hit ? "is-hit" : ""}`} key={i}>
                      <span class="lens-ch">{escapeVisible(item.ch)}</span>
                      <span class="lens-p">{(item.p * 100).toFixed(0)}%</span>
                    </span>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function LayerBarChart({
  title,
  values,
  unit,
  color,
  refValue,
  refText,
}: {
  title: string;
  values: (number | null)[];
  unit: string;
  color: string;
  refValue?: number;
  refText?: string;
}) {
  const valid = values.filter((v): v is number => v !== null);
  const maxV = Math.max(...valid, refValue ?? 0, 0.01);

  return (
    <div class="internal-card">
      <div class="internal-head">
        <strong>{title}</strong>
        {refText && <span class="internal-ref">{refText}</span>}
      </div>
      <div class="layer-bars">
        {values.map((v, i) => {
          const pct = v === null ? 0 : Math.min(100, (v / maxV) * 100);
          return (
            <div class="layer-bar-row" key={i}>
              <span class="lb-tag">L{i}</span>
              <div class="lb-track">
                <div class="lb-fill" style={{ width: `${pct.toFixed(1)}%`, background: color }} />
              </div>
              <span class="lb-val">
                {v === null ? "off" : `${v.toFixed(2)}${unit}`}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function InstrumentationPanels() {
  const cfg = config.value;
  const data = instrumentation.value;

  if (!data) {
    return (
      <div class="internals-panel">
        <LogitLensCard />
        <div class="internal-card">
          <strong>Layer diagnostics</strong>
          <p class="panel-note" style="margin-top:0.35rem;margin-bottom:0;">
            Run inference to populate per layer attention entropy, peak attention logits, residual norms, and MoE expert routing.
          </p>
        </div>
      </div>
    );
  }

  const ceiling = Math.log(cfg.ctx);

  return (
    <div class="internals-panel">
      <LogitLensCard />
      <LayerBarChart
        title="Peak attention logit"
        values={data.attn_logit_max_by_layer ?? []}
        unit=""
        color="#c2410c"
      />
      <LayerBarChart
        title="Mean attention entropy"
        values={data.attn_entropy_mean ?? []}
        unit=" nats"
        color="#1d4ed8"
        refValue={ceiling}
        refText={`Ceiling ${ceiling.toFixed(2)} nats`}
      />
      <LayerBarChart
        title="Residual stream norm"
        values={data.residual_norm_mean ?? []}
        unit=""
        color="#0f766e"
      />
      {data.moe_active && (
        <div class="internal-card">
          <div class="internal-head">
            <strong>MoE expert routing</strong>
            <span class="internal-ref">Top {data.top_k} of {data.n_experts}</span>
          </div>
          {data.routing_counts.map((counts, layer) => {
            const total = data.routing_totals[layer] || 0;
            if (total === 0) return null;
            const shares = counts.map((c) => c / total);
            const maxShare = Math.max(...shares);
            return (
              <div class="moe-row" key={layer}>
                <span class="lb-tag">L{layer}</span>
                <div class="moe-bars">
                  {shares.map((s, e) => (
                    <span
                      key={e}
                      class="moe-cell"
                      style={{ opacity: (0.25 + s * 0.75).toFixed(2) }}
                      title={`Expert ${e} share ${(s * 100).toFixed(1)}%`}
                    >
                      e{e} {(s * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
                {maxShare > 0.5 && <span class="pill pill-warn">Collapsed</span>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
