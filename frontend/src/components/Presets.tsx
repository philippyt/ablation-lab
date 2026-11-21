import { PRESETS, type Preset } from "../lib/presets";
import { config, updateConfig } from "../state";

function matches(cfg: any, p: Preset): boolean {
  for (const k in p.patch) {
    if ((cfg as any)[k] !== (p.patch as any)[k]) return false;
  }
  return true;
}

export function Presets() {
  const cfg = config.value;

  function apply(p: Preset) {
    updateConfig(p.patch, p);
  }

  return (
    <div class="presets-list">
      <p class="panel-note">
        Each preset applies the architectural choices of a real model to this lab toy scale so you can test and ablate them live.
      </p>
      <div class="presets-grid">
        {PRESETS.map((p) => {
          const active = matches(cfg, p);
          return (
            <button
              key={p.id}
              class={`preset ${active ? "preset-active" : ""}`}
              onClick={() => apply(p)}
            >
              <span class="preset-name">{p.name}</span>
              <span class="preset-family">{p.family}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
