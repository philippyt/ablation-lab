import { signal } from "@preact/signals";
import { useEffect, useState } from "preact/hooks";
import { config, surgery, kvDegradation, lastRunText } from "../state";
import { attentionKind } from "../lib/costs";
import { fetchInspect } from "../api";

export const attnModalOpen = signal<boolean>(false);
export const attnLabels = signal<string[]>([]);

const MAX_VIEW = 28;
const CELL = 22;
const LABEL_W = 104;
const HEADER_H = 34;

function escapeChar(c: string): string {
  if (c === " ") return "␣";
  if (c === "\n") return "↵";
  if (c === "\t") return "→";
  if (c.charCodeAt(0) < 32) return "?";
  return c;
}

function heatFill(w: number): string {
  const t = Math.max(0, Math.min(1, Math.sqrt(w)));
  const r = Math.round(244 - t * (244 - 29));
  const g = Math.round(247 - t * (247 - 78));
  const b = Math.round(252 - t * (252 - 216));
  return `rgb(${r}, ${g}, ${b})`;
}

export function AttentionGridModal() {
  if (!attnModalOpen.value) return null;
  const cfg = config.value;
  const att = attentionKind(cfg).toUpperCase();
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState<number | "mean">("mean");
  const [viewMode, setViewMode] = useState<"weights" | "mask">("weights");
  const [hovered, setHovered] = useState<number | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [tokens, setTokens] = useState<string[]>([]);
  const [weights, setWeights] = useState<number[][][][]>([]);

  const textToInspect = lastRunText.value || "Once upon a time";

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setErr(null);
    fetchInspect({
      config: cfg,
      text: textToInspect.slice(0, MAX_VIEW),
      surgery: surgery.value,
      kv: kvDegradation.value,
      include_attn: true,
      include_logit_lens: false,
      max_attn_tokens: MAX_VIEW,
    })
      .then((res) => {
        if (!alive) return;
        setTokens(res.tokens ?? []);
        setWeights(res.attn_weights ?? []);
        if (layer >= (res.n_layers || 1)) setLayer(0);
      })
      .catch((e) => {
        if (!alive) return;
        setErr((e as Error).message);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [textToInspect, cfg, surgery.value, kvDegradation.value]);

  const N = tokens.length;
  const focus = hovered !== null ? hovered : sel;

  function getWeight(i: number, j: number): number {
    if (j > i) return 0;
    const layerW = weights[layer];
    if (!layerW || layerW.length === 0) return 0;
    if (head === "mean") {
      let sum = 0;
      for (let h = 0; h < layerW.length; h++) sum += layerW[h]?.[i]?.[j] ?? 0;
      return sum / layerW.length;
    }
    return layerW[head]?.[i]?.[j] ?? 0;
  }

  const gridW = N * CELL;
  const totalW = LABEL_W + gridW + 24;
  const totalH = HEADER_H + N * CELL + 24;

  const topKeysForFocus = (() => {
    if (focus === null || N === 0) return [];
    const pairs: Array<{ j: number; ch: string; w: number }> = [];
    for (let j = 0; j <= focus; j++) {
      pairs.push({ j, ch: tokens[j], w: getWeight(focus, j) });
    }
    pairs.sort((a, b) => b.w - a.w);
    return pairs.slice(0, 4);
  })();

  return (
    <div class="modal-backdrop" onClick={() => (attnModalOpen.value = false)}>
      <div class="modal attn-modal" onClick={(e) => e.stopPropagation()}>
        <div class="modal-head">
          <div>
            <h3>Attention heatmap ({att})</h3>
            <p class="panel-note">
              Real post softmax attention weights for the current weights, surgery matrix, and KV settings. Bright vertical bars on column 0 are attention sinks.
            </p>
          </div>
          <button class="close" onClick={() => (attnModalOpen.value = false)}>×</button>
        </div>

        <div class="attn-toolbar">
          <div class="attn-picker-group">
            <span class="kv-ctrl-lbl">Layer</span>
            <div class="seg-row">
              {Array.from({ length: cfg.n_layers }, (_, l) => (
                <button
                  key={l}
                  class={`seg-btn ${layer === l ? "is-on" : ""}`}
                  onClick={() => setLayer(l)}
                >
                  L{l}
                </button>
              ))}
            </div>
          </div>

          <div class="attn-picker-group">
            <span class="kv-ctrl-lbl">Head</span>
            <div class="seg-row">
              <button
                class={`seg-btn ${head === "mean" ? "is-on" : ""}`}
                onClick={() => setHead("mean")}
              >
                Mean
              </button>
              {Array.from({ length: cfg.n_heads }, (_, h) => (
                <button
                  key={h}
                  class={`seg-btn ${head === h ? "is-on" : ""}`}
                  onClick={() => setHead(h)}
                >
                  H{h}
                </button>
              ))}
            </div>
          </div>

          <div class="attn-picker-group">
            <span class="kv-ctrl-lbl">Display</span>
            <div class="seg-row">
              <button
                class={`seg-btn ${viewMode === "weights" ? "is-on" : ""}`}
                onClick={() => setViewMode("weights")}
              >
                Real weights
              </button>
              <button
                class={`seg-btn ${viewMode === "mask" ? "is-on" : ""}`}
                onClick={() => setViewMode("mask")}
              >
                Causal mask
              </button>
            </div>
          </div>
        </div>

        {loading && <p class="panel-note">Computing attention weights...</p>}
        {err && <p class="panel-note is-err">{err}</p>}

        {!loading && !err && N > 0 && (
          <>
            <div class="attn-grid-wrap">
              <svg width={totalW} height={totalH} viewBox={`0 0 ${totalW} ${totalH}`}>
                <text x={LABEL_W - 8} y={18} text-anchor="end" class="attn-axis-l">queries ↓</text>
                <text x={LABEL_W + gridW / 2} y={18} text-anchor="middle" class="attn-axis-l">keys →</text>

                {tokens.map((ch, j) => (
                  <text
                    key={j}
                    x={LABEL_W + j * CELL + CELL / 2}
                    y={HEADER_H - 4}
                    text-anchor="middle"
                    class="attn-tick"
                  >
                    {escapeChar(ch)}
                  </text>
                ))}

                {tokens.map((char, i) => {
                  const isFocus = focus === i;
                  const isDimmed = focus !== null && focus !== i;
                  return (
                    <g
                      key={i}
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                      onClick={() => setSel(i)}
                      style="cursor:pointer;"
                    >
                      <text
                        x={LABEL_W - 8}
                        y={HEADER_H + i * CELL + CELL / 2 + 4}
                        text-anchor="end"
                        class="attn-rowlabel"
                      >
                        <tspan>{i}</tspan>
                        <tspan dx="6">{escapeChar(char)}</tspan>
                      </text>
                      {tokens.map((_, j) => {
                        const allowed = j <= i;
                        const w = getWeight(i, j);
                        const fill = !allowed
                          ? "#eef0f4"
                          : viewMode === "mask"
                            ? i === j
                              ? "#1d4ed8"
                              : "#93c5fd"
                            : heatFill(w);
                        return (
                          <rect
                            key={j}
                            x={LABEL_W + j * CELL}
                            y={HEADER_H + i * CELL}
                            width={CELL - 1}
                            height={CELL - 1}
                            rx="2"
                            fill={fill}
                            opacity={isDimmed ? 0.4 : 1}
                            stroke={isFocus && allowed ? "#111318" : "none"}
                            stroke-width={isFocus && allowed ? 0.6 : 0}
                          >
                            <title>
                              {allowed
                                ? `q ${i} (${escapeChar(char)}) to k ${j} (${escapeChar(tokens[j])}) weight ${(w * 100).toFixed(1)}%`
                                : "Masked future position"}
                            </title>
                          </rect>
                        );
                      })}
                    </g>
                  );
                })}
              </svg>
            </div>

            <div class="attn-readout">
              {focus === null ? (
                <span>Hover any row to see which keys that position attends to most strongly.</span>
              ) : (
                <div class="attn-top-keys">
                  <strong>Query {focus} ({escapeChar(tokens[focus])}) top keys</strong>
                  {topKeysForFocus.map((k) => (
                    <span class="pill-count" key={k.j}>
                      pos {k.j} ({escapeChar(k.ch)}) {(k.w * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
