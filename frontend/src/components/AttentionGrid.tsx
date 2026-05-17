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
  if (c === " ") return "·";
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
            <div class="meta">attention pattern{streaming && <span class="attn-live-tag">live</span>}</div>
            <h3>Causal self-attention · {att}</h3>
            <p class="short">
              Each row is a query position. Cells lit in that row are the key positions the query can read.
              A decoder-only LM uses a causal mask: position <code>i</code> sees positions <code>0..i</code>, never the future.
            </p>
          </div>
          <button class="close" onClick={() => (attnModalOpen.value = false)}>×</button>
        </div>

        <div class="attn-formula">
          <code>Attention(Q, K, V) = softmax(Q·K<sup>ᵀ</sup> / √d<sub>h</sub> + M) · V</code>
          <span>where M[i,j] = 0 if j ≤ i, −∞ otherwise. With {att}, the {cfg.n_heads} query heads share {cfg.n_kv_heads} K/V projection{cfg.n_kv_heads === 1 ? "" : "s"} per layer (head_dim = {cfg.head_dim}).</span>
        </div>

        {N === 0 ? (
          <p class="empty">Run a prompt first. The grid renders the actual tokens once the stream starts.</p>
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
                        x={LABEL_W - LABEL_W + 4}
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
              <span><span class="sw sw-self" /> Q = K (diagonal, self-attention)</span>
              <span><span class="sw sw-allowed" /> attended (past or self)</span>
              <span><span class="sw sw-masked" /> masked (future, set to −∞)</span>
              {streaming && <span><span class="sw sw-live" /> currently generating</span>}
            </div>

            <div class="attn-readout">
              {focus === null ? (
                <span class="attn-hint">
                  {streaming
                    ? `Streaming. Row ${activeRow} is the position the model is producing right now. Hover any row to see what that position attends to.`
                    : "Hover any row to see what that position attends to."}
                </span>
              ) : (
                <span>
                  Position <code>{focus}</code> (token "<code>{escapeChar(labels[focus])}</code>") attends to{" "}
                  <code>K[0..{focus}]</code> = "<code>{labels.slice(0, focus + 1).map(escapeChar).join("")}</code>"
                  <span class="muted"> · {focus + 1} key{focus === 0 ? "" : "s"} visible, {Math.max(0, N - focus - 1)} masked</span>
                </span>
              )}
            </div>

            {allLabels.length > MAX_VIEW && (
              <p class="attn-truncate">Showing first {MAX_VIEW} of {allLabels.length} positions.</p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
