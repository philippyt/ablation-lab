import { signal } from "@preact/signals";
import { useState } from "preact/hooks";
import { config, cacheState } from "../state";
import { attentionKind } from "../lib/costs";

export const attnModalOpen = signal<boolean>(false);
export const attnLabels = signal<string[]>([]);

const MAX_VIEW = 28;
const CELL = 22;
const LABEL_W = 110;
const HEADER_H = 36;

function escapeChar(c: string): string {
  if (c === " ") return "␣";
  if (c === "\n") return "↵";
  if (c === "\t") return "→";
  if (c.charCodeAt(0) < 32) return "?";
  return c;
}

export function AttentionGridModal() {
  if (!attnModalOpen.value) return null;
  const cfg = config.value;
  const allLabels = attnLabels.value.length > 0 ? attnLabels.value : Array.from(cacheState.value.prompt_tokens > 0 ? "?".repeat(cacheState.value.prompt_tokens) : "");
  const N = Math.min(allLabels.length, MAX_VIEW);
  const labels = allLabels.slice(0, N);
  const att = attentionKind(cfg).toUpperCase();
  const [sel, setSel] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const cacheS = cacheState.value;
  const streaming = cacheS.status === "streaming";
  const activeRow = streaming && allLabels.length > 0 ? Math.min(allLabels.length - 1, MAX_VIEW - 1) : -1;
  const focus = hovered !== null ? hovered : sel;

  const gridW = N * CELL;
  const totalW = LABEL_W + gridW + 30;
  const totalH = HEADER_H + N * CELL + 30;

  return (
    <div class="modal-backdrop" onClick={() => (attnModalOpen.value = false)}>
      <div class="modal attn-modal" onClick={(e) => e.stopPropagation()}>
        <div class="modal-head">
          <div>
            <h3>Causal self attention ({att})</h3>
            <p class="panel-note">
              Each row is a query position and lit cells are the keys it can read. Position i only sees positions 0 through i and never the future.
            </p>
          </div>
          <button class="close" onClick={() => (attnModalOpen.value = false)}>×</button>
        </div>

        <div class="attn-formula">
          <code>Attention(Q, K, V) = softmax(Q Kᵀ / √d_h + M) V</code>
          <span>{cfg.n_heads} query heads share {cfg.n_kv_heads} K and V projection{cfg.n_kv_heads === 1 ? "" : "s"} per layer (head dim {cfg.head_dim}).</span>
        </div>

        {N === 0 ? (
          <p class="panel-note">Run a prompt first so the grid can render real generated positions.</p>
        ) : (
          <>
            <div class="attn-grid-wrap">
              <svg width={totalW} height={totalH} viewBox={`0 0 ${totalW} ${totalH}`}>
                <text x={LABEL_W - 8} y={20} text-anchor="end" class="attn-axis-l">queries ↓</text>
                <text x={LABEL_W + gridW / 2} y={20} text-anchor="middle" class="attn-axis-l">keys →</text>

                {labels.map((_, j) => {
                  const isFocusKey = focus !== null && j <= focus;
                  return (
                    <text x={LABEL_W + j * CELL + CELL / 2} y={HEADER_H - 4}
                          text-anchor="middle"
                          class={`attn-tick ${isFocusKey ? "attn-tick-focus" : ""}`}>{j}</text>
                  );
                })}

                {labels.map((char, i) => {
                  const isFocus = focus === i;
                  const isActive = activeRow === i;
                  const isDimmed = focus !== null && focus !== i;
                  return (
                    <g
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                      onClick={() => setSel(i)}
                      style="cursor:pointer;"
                    >
                      <rect
                        x={4}
                        y={HEADER_H + i * CELL}
                        width={LABEL_W - 8 + gridW}
                        height={CELL - 1}
                        class={`attn-rowband ${isFocus ? "attn-rowband-focus" : ""} ${isActive ? "attn-rowband-active" : ""} ${isDimmed ? "attn-rowband-dim" : ""}`}
                      />
                      <text x={LABEL_W - 8} y={HEADER_H + i * CELL + CELL / 2 + 4}
                            text-anchor="end" class={`attn-rowlabel ${isFocus ? "sel" : ""} ${isActive ? "live" : ""}`}>
                        <tspan class="attn-rowidx">{i}</tspan>
                        <tspan dx="6">{escapeChar(char)}</tspan>
                      </text>
                      {labels.map((_, j) => {
                        const allowed = j <= i;
                        const isDiag = i === j;
                        const cls = [
                          "attn-cell",
                          allowed ? "allowed" : "masked",
                          isDiag ? "diag" : "",
                          isFocus && allowed ? "in-focus" : "",
                          isActive && isDiag ? "live-diag" : "",
                          isDimmed ? "dim" : "",
                        ].filter(Boolean).join(" ");
                        return (
                          <rect
                            x={LABEL_W + j * CELL}
                            y={HEADER_H + i * CELL}
                            width={CELL - 1}
                            height={CELL - 1}
                            class={cls}
                          />
                        );
                      })}
                    </g>
                  );
                })}
              </svg>
            </div>

            <div class="attn-legend">
              <span><span class="sw sw-self" /> Diagonal (self)</span>
              <span><span class="sw sw-allowed" /> Attended past</span>
              <span><span class="sw sw-masked" /> Masked future</span>
            </div>

            <div class="attn-readout">
              {focus === null ? (
                <span>Hover any row to inspect which past keys that position reads.</span>
              ) : (
                <span>
                  Position <code>{focus}</code> ({escapeChar(labels[focus])}) reads {focus + 1} visible keys and masks {Math.max(0, N - focus - 1)} future keys.
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
