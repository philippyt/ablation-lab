import { config, openExplainerId, hoverInfo, surgicalCount, kvDegradationActive } from "../state";
import { attentionKind } from "../lib/costs";
import { diagramFfnLabel } from "../lib/labels";

const FIXED = "dia-fixed";
const SWAP = "dia-swap";
const SWAP_DIM = "dia-swap-dim";

const BOX_W = 84;
const BOX_W_SHORT = 52;
const BOX_H = 50;
const PLUS_R = 12;
const GAP = 26;
const FRAME_TOP = 12;
const RESIDUAL_Y = 32;
const TOP_PAD = 68;
const FRAME_BOTTOM_PAD = 12;

interface Item {
  type: "box" | "plus";
  x: number;
  w: number;
  label?: string;
  sub?: string;
  fill?: string;
  explainerId?: string;
}

function attentionExplainerId(cfg: { n_heads: number; n_kv_heads: number; attention_type: string }): string {
  if (cfg.attention_type === "mla") return "attention.mla";
  if (cfg.n_kv_heads === cfg.n_heads) return "attention.mha";
  if (cfg.n_kv_heads === 1) return "attention.mqa";
  return "attention.gqa";
}

export function Diagram() {
  const cfg = config.value;
  const att = attentionKind(cfg).toUpperCase();
  const normShort = cfg.norm === "layernorm" ? "LN" : "RMS";
  const ffn = diagramFfnLabel(cfg);
  const posShort =
    cfg.pos_embed === "rope"
      ? "RoPE"
      : cfg.pos_embed === "alibi"
        ? "ALiBi"
        : cfg.pos_embed === "absolute"
          ? "Absolute"
          : cfg.pos_embed === "sinusoidal"
            ? "Sinusoidal"
            : "NoPE";

  let x = 12;
  const items: Item[] = [];

  const pushBox = (label: string, sub: string | undefined, fill: string, explainerId: string | undefined, w = BOX_W) => {
    items.push({ type: "box", x, w, label, sub, fill, explainerId });
    x += w + GAP;
  };
  const pushPlus = () => {
    const cx = x + PLUS_R;
    items.push({ type: "plus", x: cx, w: 2 * PLUS_R });
    x += 2 * PLUS_R + GAP;
    return cx;
  };

  pushBox("Token", `vocab ${cfg.vocab_size}`, FIXED, "misc.token_embedding");
  pushBox(posShort, cfg.pos_embed === "absolute" ? `${cfg.ctx} × ${cfg.d_model}` : "position", SWAP, `pos.${cfg.pos_embed}`);

  const blockStart = x;
  const preNorm1X = x;
  if (cfg.enable_norms) pushBox(normShort, undefined, SWAP_DIM, `norm.${cfg.norm}`, BOX_W_SHORT);
  const attSub =
    cfg.attention_type === "mla"
      ? `latent ${Math.max(8, Math.floor(cfg.d_model / 8))}`
      : `${cfg.n_heads}q ${cfg.n_kv_heads}kv${cfg.qk_norm ? " QKn" : ""}`;
  if (cfg.enable_attention) pushBox(`Attn ${att}`, attSub, SWAP, attentionExplainerId(cfg), 96);
  else pushBox("Attn", "ablated", "dia-disabled", undefined, 96);
  const plus1X = pushPlus();
  const postPlus1X = x;
  if (cfg.enable_norms) pushBox(normShort, undefined, SWAP_DIM, `norm.${cfg.norm}`, BOX_W_SHORT);
  if (cfg.enable_ffn) pushBox(ffn.label, ffn.sub, SWAP, `ffn.${cfg.ffn}`);
  else pushBox("FFN", "ablated", "dia-disabled", undefined);
  const plus2X = pushPlus();
  const blockEnd = x;

  if (cfg.enable_norms) pushBox(normShort, undefined, FIXED, `norm.${cfg.norm}`, BOX_W_SHORT);
  pushBox("LM Head", cfg.tie_embeddings ? "tied" : "untied", FIXED, "misc.tied_embeddings");

  const totalW = x + 4;
  const cy = TOP_PAD + BOX_H / 2;
  const frameTopY = FRAME_TOP;
  const frameBottomY = TOP_PAD + BOX_H + FRAME_BOTTOM_PAD;
  const totalH = frameBottomY + 6;

  const frameX = blockStart - GAP / 2;
  const frameW = blockEnd - blockStart - GAP / 2 + 6;

  const posItem = items[1];
  const tap1X = (posItem.x + posItem.w + preNorm1X) / 2;
  const tap2X = (plus1X + PLUS_R + postPlus1X) / 2;

  function onEnter(id: string, e: any) {
    const r = (e.target as SVGElement).getBoundingClientRect();
    hoverInfo.value = { id, x: r.left + r.width / 2, y: r.bottom + 8, placement: "below" };
  }
  function onLeave() {
    hoverInfo.value = null;
  }

  function residualPath(tapX: number, endX: number) {
    return `M ${tapX} ${cy} L ${tapX} ${RESIDUAL_Y} L ${endX} ${RESIDUAL_Y} L ${endX} ${cy - PLUS_R - 1}`;
  }

  const hasSurgery = surgicalCount.value > 0;
  const hasKV = kvDegradationActive.value;

  return (
    <div class="diagram-outer">
      <div class="diagram-wrap">
        <svg class="diagram" width="100%" height="100%" viewBox={`0 0 ${totalW} ${totalH}`} preserveAspectRatio="xMidYMid meet">
          <defs>
            <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
              <path class="dia-arrow" d="M0,0 L10,5 L0,10 z" />
            </marker>
            <marker id="res-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
              <path class="dia-residual-arrow" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          <rect
            class="dia-block-frame"
            x={frameX}
            y={frameTopY}
            width={frameW}
            height={frameBottomY - frameTopY}
            stroke-dasharray="4 3"
            stroke-width="1.2"
            rx="8"
          />

          <g class="dia-block-tag">
            <rect x={frameX + frameW - 76} y={frameTopY - 9} width="68" height="18" rx="4" />
            <text x={frameX + frameW - 42} y={frameTopY + 3.5} text-anchor="middle" font-size="10.5" font-weight="600">
              ×{cfg.n_layers} layers
            </text>
          </g>

          {items.slice(0, -1).map((it, i) => {
            const next = items[i + 1];
            const x1 = it.type === "box" ? it.x + it.w : it.x + PLUS_R;
            const x2 = next.type === "box" ? next.x : next.x - PLUS_R;
            return (
              <line key={i} class="dia-line" x1={x1 + 2} y1={cy} x2={x2 - 1} y2={cy} stroke-width="1.5" marker-end="url(#arr)" />
            );
          })}

          <path class="dia-residual" d={residualPath(tap1X, plus1X)} stroke-width="1.5" marker-end="url(#res-arr)" fill="none" />
          <path class="dia-residual" d={residualPath(tap2X, plus2X)} stroke-width="1.5" marker-end="url(#res-arr)" fill="none" />
          <circle class="dia-residual-tap" cx={tap1X} cy={cy} r="2.8" />
          <circle class="dia-residual-tap" cx={tap2X} cy={cy} r="2.8" />
          <text class="dia-residual-label" x={(tap1X + plus1X) / 2} y={RESIDUAL_Y - 5} text-anchor="middle" font-size="9.5">
            residual
          </text>
          <text class="dia-residual-label" x={(tap2X + plus2X) / 2} y={RESIDUAL_Y - 5} text-anchor="middle" font-size="9.5">
            residual
          </text>

          {items.map((it, i) => {
            if (it.type === "plus") {
              return (
                <g key={i}>
                  <circle class="dia-plus" cx={it.x} cy={cy} r={PLUS_R} stroke-width="1.5" />
                  <text class="dia-plus-text" x={it.x} y={cy + 4.5} text-anchor="middle" font-size="15" font-weight="700">
                    +
                  </text>
                </g>
              );
            }
            const interactive = !!it.explainerId;
            const hasSub = !!it.sub;
            return (
              <g key={i}>
                <rect
                  class={`dia-box ${it.fill}`}
                  x={it.x}
                  y={TOP_PAD}
                  width={it.w}
                  height={BOX_H}
                  stroke-width="1.3"
                  rx="6"
                  style={{ cursor: interactive ? "pointer" : "default" }}
                  onClick={interactive ? () => (openExplainerId.value = it.explainerId!) : undefined}
                  onMouseEnter={interactive ? (e) => onEnter(it.explainerId!, e) : undefined}
                  onMouseLeave={interactive ? onLeave : undefined}
                />
                <text
                  class="dia-text"
                  x={it.x + it.w / 2}
                  y={hasSub ? cy - 4 : cy + 4}
                  text-anchor="middle"
                  font-size="11.5"
                  font-weight="600"
                  style="pointer-events:none;"
                >
                  {it.label}
                </text>
                {hasSub && (
                  <text
                    class="dia-text-sub"
                    x={it.x + it.w / 2}
                    y={cy + 12}
                    text-anchor="middle"
                    font-size="9.5"
                    font-weight="500"
                    style="pointer-events:none;"
                  >
                    {it.sub}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
      <div class="diagram-footer">
        <div class="diagram-legend">
          <span class="legend-item"><span class="legend-sw dia-swap" /> Swappable</span>
          <span class="legend-item"><span class="legend-sw dia-fixed" /> Fixed</span>
          <span class="legend-item"><span class="legend-sw dia-disabled" /> Ablated</span>
        </div>
        <div class="diagram-badges">
          {hasSurgery && <span class="pill pill-warn">Surgery active</span>}
          {hasKV && <span class="pill pill-warn">KV degraded</span>}
          <span class="diagram-tip">Click any block for its explainer</span>
        </div>
      </div>
    </div>
  );
}
