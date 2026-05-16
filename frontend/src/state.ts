import { signal, computed } from "@preact/signals";
import type { ArchConfig, CostPanel, ManifestEntry } from "./types";
import { costPanel } from "./lib/costs";
import { archHash } from "./lib/hash";
export { openExplainerId } from "./content";

const DEFAULT_VOCAB = 72;

export const defaultConfig: ArchConfig = {
  vocab_size: DEFAULT_VOCAB,
  d_model: 128,
  n_layers: 4,
  n_heads: 8,
  n_kv_heads: 8,
  head_dim: 16,
  ctx: 128,
  norm: "layernorm",
  ffn: "dense",
  ffn_mult: 4,
  n_experts: 4,
  top_k: 1,
  tie_embeddings: true,
  pos_embed: "rope",
  activation: "silu",
  qk_norm: false,
  attention_type: "standard",
  enable_attention: true,
  enable_ffn: true,
  enable_norms: true,
};

export const config = signal<ArchConfig>({ ...defaultConfig });
export const name = signal<string>("My model");
export const entries = signal<ManifestEntry[]>([]);
export const previousCost = signal<CostPanel | null>(null);
export const previousConfig = signal<ArchConfig | null>(null);
export const generating = signal<boolean>(false);
export const lastError = signal<string | null>(null);
export const lastConsequences = signal<Array<{ id: string; text: string }>>([]);

export interface CacheStateView {
  positions: number;
  status: "idle" | "streaming" | "done";
  prompt_tokens: number;
}

export const cacheState = signal<CacheStateView>({ positions: 0, status: "idle", prompt_tokens: 0 });

export interface HoverInfo {
  id: string;
  x: number;
  y: number;
  placement: "below" | "right";
}
export const hoverInfo = signal<HoverInfo | null>(null);

const _theme = typeof localStorage !== "undefined" ? localStorage.getItem("ablation-theme") : null;
export const darkMode = signal<boolean>(_theme === "dark");

export function toggleDarkMode(): void {
  darkMode.value = !darkMode.value;
  try {
    localStorage.setItem("ablation-theme", darkMode.value ? "dark" : "light");
  } catch {}
}

export const currentCost = computed<CostPanel>(() => costPanel(config.value));

export const currentHash = signal<string>("");
export const matchingEntry = computed<ManifestEntry | null>(() => {
  const h = currentHash.value;
  return entries.value.find(e => e.hash === h) ?? null;
});

export async function updateConfig(patch: Partial<ArchConfig>): Promise<void> {
  const next = { ...config.value, ...patch };
  if (next.d_model % next.n_heads === 0) {
    next.head_dim = next.d_model / next.n_heads;
  }
  if (next.n_heads % next.n_kv_heads !== 0) {
    next.n_kv_heads = next.n_heads;
  }
  previousCost.value = costPanel(config.value);
  previousConfig.value = { ...config.value };
  config.value = next;
  currentHash.value = await archHash(next);
  const { detectConsequences } = await import("./lib/consequences");
  lastConsequences.value = detectConsequences(previousConfig.value, previousCost.value, next, costPanel(next));
}

export async function setEntries(list: ManifestEntry[]): Promise<void> {
  entries.value = list;
  if (list.length > 0) {
    const baseline = list[0];
    config.value = { ...baseline.arch_config };
    name.value = baseline.name ?? "My model";
    previousCost.value = null;
  }
  currentHash.value = await archHash(config.value);
}
