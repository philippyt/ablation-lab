import { PRESETS, type Preset } from "../lib/presets";
import { config, updateConfig, name } from "../state";

function matches(cfg: any, p: Preset): boolean {
  for (const k in p.patch) {
    if ((cfg as any)[k] !== (p.patch as any)[k]) return false;
  }
  return true;
}

export function Presets() {
  const cfg = config.value;

  function apply(p: Preset) {
    name.value = p.name;
    updateConfig(p.patch);
  }

  return (
    <div class="presets-list">
      {PRESETS.map((p) => {
        const active = matches(cfg, p);
        return (
          <button
            class={`preset ${active ? "preset-active" : ""}`}
            onClick={() => apply(p)}
          >
            <span class="preset-name">{p.name}</span>
            <span class="preset-family">{p.family}</span>
          </button>
        );
      })}
    </div>
  );
}
