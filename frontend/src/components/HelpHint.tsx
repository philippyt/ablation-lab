import { useState } from "preact/hooks";

interface Props {
  text: string;
}

const TOOLTIP_WIDTH_PX = 272;
const TOOLTIP_ESTIMATED_HEIGHT_PX = 200;
const EDGE_MARGIN_PX = 12;

export function HelpHint({ text }: Props) {
  const [pos, setPos] = useState<{ left: number; top: number; xSide: "left" | "right"; ySide: "top" | "middle" | "bottom" } | null>(null);

  function show(e: any) {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const xSide: "left" | "right" = r.left < vw / 2 ? "right" : "left";
    const center = r.top + r.height / 2;
    const halfTip = TOOLTIP_ESTIMATED_HEIGHT_PX / 2;
    let ySide: "top" | "middle" | "bottom" = "middle";
    let top = center;
    if (center - halfTip < EDGE_MARGIN_PX) {
      ySide = "top";
      top = EDGE_MARGIN_PX;
    } else if (center + halfTip > vh - EDGE_MARGIN_PX) {
      ySide = "bottom";
      top = vh - EDGE_MARGIN_PX;
    }
    setPos({
      left: xSide === "right" ? r.right + 8 : r.left - 8,
      top,
      xSide,
      ySide,
    });
  }
  function hide() {
    setPos(null);
  }

  return (
    <span class="help-hint" onMouseEnter={show} onMouseLeave={hide}>
      <span class="help-hint-icon">?</span>
      {pos && (() => {
        const xTrans = pos.xSide === "right" ? "0" : "-100%";
        const yTrans = pos.ySide === "middle" ? "-50%" : pos.ySide === "top" ? "0" : "-100%";
        return (
          <span
            class="help-hint-tooltip-fixed"
            style={`left:${pos.left}px;top:${pos.top}px;width:${TOOLTIP_WIDTH_PX}px;transform:translate(${xTrans}, ${yTrans});`}
          >
            {text}
          </span>
        );
      })()}
    </span>
  );
}
