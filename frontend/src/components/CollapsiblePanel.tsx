import { useState } from "preact/hooks";

interface Props {
  title: string;
  defaultOpen?: boolean;
  action?: any;
  children: any;
}

export function CollapsiblePanel({ title, defaultOpen = true, action, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section class={`panel collapsible ${open ? "is-open" : "is-closed"}`}>
      <header class="panel-summary" onClick={() => setOpen(!open)}>
        <h2>{title}</h2>
        <div class="panel-summary-right" onClick={(e) => e.stopPropagation()}>
          {action}
          <span class="panel-chev" onClick={() => setOpen(!open)}>{open ? "−" : "+"}</span>
        </div>
      </header>
      {open && <div class="panel-body">{children}</div>}
    </section>
  );
}
