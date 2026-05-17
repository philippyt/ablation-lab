import { marked } from "marked";
import DOMPurify from "dompurify";
import { useEffect, useRef } from "preact/hooks";
import { getExplainer } from "../content";
import { openExplainerId } from "../state";

marked.setOptions({ gfm: true, breaks: false });
marked.use({
  renderer: {
    link({ href, text }: { href: string; text: string }): string | false {
      const m = /^(concept|component):(.+)$/.exec(href || "");
      if (m) return `<a href="#${m[2]}" data-explainer="${m[2]}" class="explainer-link">${text}</a>`;
      return false;
    },
  },
});

function renderMarkdown(md: string): string {
  const html = marked.parse(md, { async: false }) as string;
  return DOMPurify.sanitize(html, { ADD_ATTR: ["data-explainer"] });
}

function handleBodyClick(e: MouseEvent) {
  const a = (e.target as HTMLElement).closest("a[data-explainer]");
  if (a) {
    e.preventDefault();
    const id = a.getAttribute("data-explainer");
    if (id) openExplainerId.value = id;
  }
}

export function SidePanel() {
  const id = openExplainerId.value;
  const panelRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!id) return;
    function onDocClick(e: MouseEvent) {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      if (panelRef.current && panelRef.current.contains(target)) return;
      if (target.closest("[data-explainer], .diagram-box, .help-hint, .tooltip, .help-hint-tooltip-fixed")) return;
      openExplainerId.value = null;
    }
    function onEsc(e: KeyboardEvent) {
      if (e.key === "Escape") openExplainerId.value = null;
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, [id]);

  if (!id) return null;
  const ex = getExplainer(id);
  if (!ex) return null;

  return (
    <aside class="sidepanel" ref={panelRef as any}>
      <button class="close" onClick={() => (openExplainerId.value = null)} aria-label="close">×</button>
      <div class="meta">{ex.level}</div>
      <h3>{ex.title}</h3>
      <p class="short">{ex.short}</p>
      <div class="body" onClick={handleBodyClick} dangerouslySetInnerHTML={{ __html: renderMarkdown(ex.body) }} />
      {ex.sources && ex.sources.length > 0 && (
        <div class="sources">
          <div class="kv">sources</div>
          <ul>
            {ex.sources.map((s) => (
              <li>{s}</li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  );
}

import { hoverInfo } from "../state";

export function FloatingTooltip() {
  const h = hoverInfo.value;
  if (!h) return null;
  const ex = getExplainer(h.id);
  if (!ex) return null;
  const style = h.placement === "right"
    ? { left: `${h.x}px`, top: `${h.y}px`, transform: "translateY(-50%)" }
    : { left: `${h.x}px`, top: `${h.y}px`, transform: "translateX(-50%)" };
  return (
    <div class="tooltip" style={style}>
      <strong>{ex.title}</strong>
      <div>{ex.short}</div>
      <div class="kv">click for more</div>
    </div>
  );
}
