import { useEffect, useRef, useState } from "preact/hooks";

interface Props {
  text: string;
}

const TOOLTIP_WIDTH_PX = 272;
const TOOLTIP_ESTIMATED_HEIGHT_PX = 200;
const EDGE_MARGIN_PX = 12;

type Pos = { left: number; top: number; xSide: "left" | "right"; ySide: "top" | "middle" | "bottom" };

export function HelpHint({ text }: Props) {
  const [pos, setPos] = useState<Pos | null>(null);
  const [pinned, setPinned] = useState(false);
  const rootRef = useRef<HTMLSpanElement | null>(null);

  function compute(el: HTMLElement): Pos {
    const r = el.getBoundingClientRect();
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
    return {
      left: xSide === "right" ? r.right + 8 : r.left - 8,
      top,
      xSide,
      ySide,
    };
  }

  function show(e: any) {
    setPos(compute(e.currentTarget as HTMLElement));
  }
  function hide() {
    if (!pinned) setPos(null);
  }
  function toggle(e: any) {
    e.stopPropagation();
    if (pinned) {
      setPinned(false);
      setPos(null);
    } else {
      setPos(compute(e.currentTarget as HTMLElement));
      setPinned(true);
    }
  }

  useEffect(() => {
    if (!pinned) return;
    function onDoc(e: MouseEvent) {
      const t = e.target as HTMLElement | null;
      if (rootRef.current && t && rootRef.current.contains(t)) return;
      setPinned(false);
      setPos(null);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [pinned]);

  return (
    <span class="help-hint" ref={rootRef} onMouseEnter={show} onMouseLeave={hide} onClick={toggle} role="button" tabIndex={0} aria-label="help">
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
