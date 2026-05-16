import { useState } from "preact/hooks";

interface Props {
  text: string;
}

export function HelpHint({ text }: Props) {
  const [pos, setPos] = useState<{ left: number; top: number; side: "left" | "right" } | null>(null);

  function show(e: any) {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const isLeftHalf = r.left < window.innerWidth / 2;
    setPos({
      left: isLeftHalf ? r.right + 8 : r.left - 8,
      top: r.top + r.height / 2,
      side: isLeftHalf ? "right" : "left",
    });
  }
  function hide() {
    setPos(null);
  }

  return (
    <span class="help-hint" onMouseEnter={show} onMouseLeave={hide}>
      <span class="help-hint-icon">?</span>
      {pos && (
        <span
          class="help-hint-tooltip-fixed"
          style={`left:${pos.left}px;top:${pos.top}px;transform:translate(${pos.side === "right" ? "0" : "-100%"}, -50%);`}
        >
          {text}
        </span>
      )}
    </span>
  );
}
