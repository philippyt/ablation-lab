import {
  config,
  surgery,
  instrumentation,
  toggleHead,
  toggleLayerPart,
  resetSurgery,
  surgicalCount,
} from "../state";

export function SurgeryPanel() {
  const cfg = config.value;
  const s = surgery.value;
  const instr = instrumentation.value;
  const count = surgicalCount.value;
  const nLayers = cfg.n_layers;
  const nHeads = cfg.n_heads;

  const activity = instr?.head_activity ?? [];
  let maxAct = 0;
  for (const row of activity) {
    for (const v of row) {
      if (v > maxAct) maxAct = v;
    }
  }

  function ablateHalfHeads() {
    const next: Record<number, number[]> = {};
    for (let l = 0; l < nLayers; l++) {
      const hs: number[] = [];
      for (let h = 0; h < nHeads; h += 2) hs.push(h);
      next[l] = hs;
    }
    surgery.value = { ...s, disabled_heads: next };
  }

  function ablateLateAttn() {
    surgery.value = {
      ...s,
      disabled_attn_layers: nLayers >= 2 ? [nLayers - 1] : [],
    };
  }

  function ablateLateFfn() {
    surgery.value = {
      ...s,
      disabled_ffn_layers: nLayers >= 2 ? [nLayers - 1] : [],
    };
  }

  return (
    <div class="surgery-panel">
      <div class="sub-head">
        <div class="sub-title">
          <strong>Layer and head surgery</strong>
          <span class="pill-count">{count === 0 ? "All active" : `${count} ablated`}</span>
        </div>
        <div class="quick-chips">
          <button class="chip-btn" onClick={ablateHalfHeads}>Half heads</button>
          <button class="chip-btn" onClick={ablateLateAttn}>Last attn off</button>
          <button class="chip-btn" onClick={ablateLateFfn}>Last FFN off</button>
          {count > 0 && (
            <button class="chip-btn chip-reset" onClick={resetSurgery}>Reset</button>
          )}
        </div>
      </div>

      <p class="panel-note">
        Click any head or sublayer pill to zero its contribution live during inference. Darker blue cells carried higher output norm in the last run.
      </p>

      <div class="surgery-grid">
        <div class="surgery-col-headers">
          <span class="sg-layer-lbl">Layer</span>
          <span class="sg-parts-lbl">Sublayers</span>
          <div class="sg-heads-lbl">
            {Array.from({ length: nHeads }, (_, h) => (
              <span>H{h}</span>
            ))}
          </div>
        </div>

        {Array.from({ length: nLayers }, (_, layer) => {
          const attnOff = !cfg.enable_attention || s.disabled_attn_layers.includes(layer);
          const ffnOff = !cfg.enable_ffn || s.disabled_ffn_layers.includes(layer);
          const normOff = !cfg.enable_norms || s.disabled_norm_layers.includes(layer);
          const offHeads = s.disabled_heads[layer] || [];

          return (
            <div class="surgery-row" key={layer}>
              <span class="sg-layer-tag">L{layer}</span>

              <div class="sg-parts">
                <button
                  class={`sg-pill ${attnOff ? "is-off" : ""}`}
                  onClick={() => toggleLayerPart("attn", layer)}
                  disabled={!cfg.enable_attention}
                  title="Toggle attention sublayer for this layer"
                >
                  Attn
                </button>
                <button
                  class={`sg-pill ${ffnOff ? "is-off" : ""}`}
                  onClick={() => toggleLayerPart("ffn", layer)}
                  disabled={!cfg.enable_ffn}
                  title="Toggle FFN sublayer for this layer"
                >
                  FFN
                </button>
                <button
                  class={`sg-pill ${normOff ? "is-off" : ""}`}
                  onClick={() => toggleLayerPart("norm", layer)}
                  disabled={!cfg.enable_norms}
                  title="Toggle norms for this layer"
                >
                  Norm
                </button>
              </div>

              <div class="sg-heads">
                {Array.from({ length: nHeads }, (_, head) => {
                  const headOff = attnOff || offHeads.includes(head);
                  const actVal = activity[layer]?.[head] ?? 0;
                  const intensity = maxAct > 0 ? Math.min(1, actVal / maxAct) : 0;
                  const bg = headOff
                    ? undefined
                    : intensity > 0
                      ? `rgba(29, 78, 216, ${(0.08 + intensity * 0.34).toFixed(2)})`
                      : undefined;

                  return (
                    <button
                      key={head}
                      class={`sg-head ${headOff ? "is-off" : ""}`}
                      style={bg ? { background: bg } : undefined}
                      onClick={() => toggleHead(layer, head)}
                      disabled={attnOff}
                      title={
                        headOff
                          ? `Layer ${layer} head ${head} ablated`
                          : `Layer ${layer} head ${head}${actVal > 0 ? ` norm ${actVal.toFixed(2)}` : ""}`
                      }
                    >
                      {headOff ? "×" : head}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
