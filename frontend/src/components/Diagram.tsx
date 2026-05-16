import { config, openExplainerId, hoverInfo } from "../state";
import { attentionKind } from "../lib/costs";
import { diagramFfnLabel } from "../lib/labels";

const FIXED = "dia-fixed";
const SWAP = "dia-swap";
const SWAP_DIM = "dia-swap-dim";

const BOX_W = 76;
const BOX_W_SHORT = 52;
const BOX_H = 40;
const PLUS_R = 11;
const GAP = 16;
const TOP_PAD = 64;
const BOT_PAD = 18;
const LABEL_FONT = 10;
const SUB_FONT = 8.5;

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
  const posShort = cfg.pos_embed === "rope" ? "RoPE"
    : cfg.pos_embed === "alibi" ? "ALiBi"
    : cfg.pos_embed === "absolute" ? "Absolute"
    : cfg.pos_embed === "sinusoidal" ? "Sinusoidal"
    : "NoPE";

  let x = 14;
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

  pushBox("Token", `vocab ${cfg.vocab_size}`, FIXED, "misc.tied_embeddings");
  pushBox(posShort, cfg.pos_embed === "absolute" ? `${cfg.ctx}×${cfg.d_model}` : "position", SWAP, `pos.${cfg.pos_embed}`);

  const blockStart = x;
  const preNorm1X = x;
  if (cfg.enable_norms) pushBox(normShort, undefined, SWAP_DIM, `norm.${cfg.norm}`, BOX_W_SHORT);
  const attSub = cfg.attention_type === "mla"
    ? `latent ${Math.max(8, Math.floor(cfg.d_model / 8))}`
    : `${cfg.n_heads}q/${cfg.n_kv_heads}kv${cfg.qk_norm ? "·QKn" : ""}`;
  if (cfg.enable_attention) pushBox(`Attn ${att}`, attSub, SWAP, attentionExplainerId(cfg));
  else pushBox("Attn", "ablated", "dia-disabled", undefined);
  const plus1X = pushPlus();
  const postPlus1X = x;
  if (cfg.enable_norms) pushBox(normShort, undefined, SWAP_DIM, `norm.${cfg.norm}`, BOX_W_SHORT);
  if (cfg.enable_ffn) pushBox(ffn.label, ffn.sub, SWAP, `ffn.${cfg.ffn}`);
  else pushBox("FFN", "ablated", "dia-disabled", undefined);
  const plus2X = pushPlus();
  const blockEnd = x;

  if (cfg.enable_norms) pushBox(`Final ${normShort}`, undefined, FIXED, `norm.${cfg.norm}`, BOX_W_SHORT);
  pushBox("LM Head", cfg.tie_embeddings ? "tied" : "untied", FIXED, "misc.tied_embeddings");

  const totalW = x + 6;
  const totalH = TOP_PAD + BOX_H + BOT_PAD;
  const cy = TOP_PAD + BOX_H / 2;
  const arcTopY = 20;
  const arcTopY2 = 38;

  function onEnter(id: string, e: any) {
    const r = (e.target as SVGElement).getBoundingClientRect();
    hoverInfo.value = { id, x: r.left + r.width / 2, y: r.bottom + 8, placement: "below" };
  }
  function onLeave() {
    hoverInfo.value = null;
  }

  function arc(startX: number, endX: number, topY: number) {
    return `M ${startX} ${TOP_PAD} L ${startX} ${topY} L ${endX} ${topY} L ${endX} ${cy - PLUS_R}`;
  }

  return (
    <div class="diagram-wrap">
      <svg class="diagram" width="100%" height="100%" viewBox={`0 0 ${totalW} ${totalH}`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path class="dia-arrow" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <rect class="dia-block-frame" x={blockStart - GAP / 2} y={TOP_PAD - 12}
              width={blockEnd - blockStart - GAP / 2 + 4}
              height={BOX_H + 24}
              stroke-dasharray="3 2" stroke-width="1" />
        <text class="dia-block-label" x={blockEnd - GAP / 2 - 2} y={TOP_PAD - 15} text-anchor="end"
              font-family="'Inter Tight', sans-serif" font-size="10" font-weight="500"
              font-style="italic">
          ×{cfg.n_layers} layers
        </text>

        <path class="dia-residual" d={arc(preNorm1X - GAP / 2, plus1X, arcTopY)} stroke-width="1.1" />
        <path class="dia-residual" d={arc(postPlus1X - GAP / 2, plus2X, arcTopY2)} stroke-width="1.1" />
        <text class="dia-residual-label" x={(preNorm1X + plus1X) / 2} y={arcTopY - 4} text-anchor="middle"
              font-family="'Inter Tight', sans-serif" font-size="9" font-style="italic">
          residual
        </text>
        <text class="dia-residual-label" x={(postPlus1X + plus2X) / 2} y={arcTopY2 - 4} text-anchor="middle"
              font-family="'Inter Tight', sans-serif" font-size="9" font-style="italic">
          residual
        </text>

        {items.slice(0, -1).map((it, i) => {
          const next = items[i + 1];
          const x1 = it.type === "box" ? it.x + it.w : it.x + PLUS_R;
          const x2 = next.type === "box" ? next.x : next.x - PLUS_R;
          return (
            <line class="dia-line" x1={x1 + 2} y1={cy} x2={x2 - 4} y2={cy}
                  stroke-width="1.6" marker-end="url(#arr)" />
          );
        })}

        {items.map((it, i) => {
          if (it.type === "plus") {
            return (
              <g key={i}>
                <circle class="dia-plus" cx={it.x} cy={cy} r={PLUS_R} stroke-width="1.3" />
                <text class="dia-plus-text" x={it.x} y={cy + 4} text-anchor="middle"
                      font-family="'Inter Tight', sans-serif" font-size="13" font-weight="700">+</text>
              </g>
            );
          }
          const interactive = !!it.explainerId;
          const hasSub = !!it.sub;
          return (
            <g key={i}>
              <rect class={`dia-box ${it.fill}`}
                    x={it.x} y={TOP_PAD} width={it.w} height={BOX_H}
                    stroke-width="1.2"
                    style={{ cursor: interactive ? "pointer" : "default" }}
                    onClick={interactive ? () => (openExplainerId.value = it.explainerId!) : undefined}
                    onMouseEnter={interactive ? (e) => onEnter(it.explainerId!, e) : undefined}
                    onMouseLeave={interactive ? onLeave : undefined}
              />
              <text class="dia-text" x={it.x + it.w / 2} y={hasSub ? cy - 2 : cy + 3} text-anchor="middle"
                    font-family="'Inter Tight', sans-serif" font-size={LABEL_FONT}
                    font-weight="600"
                    style="pointer-events:none;">
                {it.label}
              </text>
              {hasSub && (
                <text class="dia-text-sub" x={it.x + it.w / 2} y={cy + 10} text-anchor="middle"
                      font-family="'Inter Tight', sans-serif" font-size={SUB_FONT}
                      font-weight="500" font-style="italic"
                      style="pointer-events:none;">
                  {it.sub}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
