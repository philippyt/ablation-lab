import { lastConsequences } from "../state";

export function Consequences() {
  const items = lastConsequences.value;
  if (items.length === 0) return null;
  return (
    <div class="consequence-list">
      <div class="kv consequence-head">latest change</div>
      {items.map((it) => (
        <p class="consequence-line">{it.text}</p>
      ))}
    </div>
  );
}
