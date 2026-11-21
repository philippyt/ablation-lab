import { lastConsequences } from "../state";

export function Consequences() {
  const items = lastConsequences.value;
  if (items.length === 0) return null;
  return (
    <div class="consequence-list">
      <div class="cfg-group-title">Latest change</div>
      {items.map((it) => (
        <p class="consequence-line" key={it.id}>{it.text}</p>
      ))}
    </div>
  );
}
