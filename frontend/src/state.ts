import { signal, computed } from "@preact/signals";
import type {
  ArchConfig,
  CostPanel,
  ManifestEntry,
  Instrumentation,
  SurgeryConfig,
  KVDegradationConfig,
  TelemetryPoint,
  LogitLensStage,
} from "./types";
import { fetchInspect } from "./api";
import { costPanel, kvCachePerToken } from "./lib/costs";
import { archHash } from "./lib/hash";
import { PRESETS, type Preset } from "./lib/presets";
export { openExplainerId } from "./content";

function configMatchesPreset(cfg: ArchConfig, p: Preset): boolean {
  for (const k in p.patch) {
    if ((cfg as any)[k] !== (p.patch as any)[k]) return false;
  }
  return true;
}

const DEFAULT_VOCAB = 72;

export const defaultConfig: ArchConfig = {
  vocab_size: DEFAULT_VOCAB,
  d_model: 128,
  n_layers: 4,
  n_heads: 8,
  n_kv_heads: 2,
  head_dim: 16,
  ctx: 256,
  norm: "rmsnorm",
  ffn: "dense",
  ffn_mult: 4,
  n_experts: 4,
  top_k: 1,
  tie_embeddings: false,
  pos_embed: "rope",
  activation: "silu",
  qk_norm: true,
  attention_type: "standard",
  enable_attention: true,
  enable_ffn: true,
  enable_norms: true,
};

export function emptySurgery(): SurgeryConfig {
  return {
    disabled_heads: {},
    disabled_attn_layers: [],
    disabled_ffn_layers: [],
    disabled_norm_layers: [],
  };
}

export function defaultKV(): KVDegradationConfig {
  return {
    quant: "none",
    window: null,
    sink_mode: "normal",
    sink_tokens: 4,
    drop_prob: 0,
  };
}

export const labMode = signal<"single" | "race">("single");
export const labTab = signal<"surgery" | "kv" | "internals">("surgery");
export const lastRunText = signal<string>("Once upon a time");
export interface LogitLensView {
  chosenChar: string;
  stepIndex: number;
  stages: LogitLensStage[];
  loading: boolean;
  error: string | null;
}
export const logitLens = signal<LogitLensView | null>(null);

export async function inspectTokenLogitLens(prefix: string, chosenChar: string, stepIndex: number): Promise<void> {
  labTab.value = "internals";
  logitLens.value = { chosenChar, stepIndex, stages: [], loading: true, error: null };
  try {
    const res = await fetchInspect({
      config: config.value,
      text: prefix,
      surgery: surgery.value,
      kv: kvDegradation.value,
      include_attn: false,
      include_logit_lens: true,
    });
    logitLens.value = {
      chosenChar,
      stepIndex,
      stages: res.logit_lens ?? [],
      loading: false,
      error: null,
    };
  } catch (e) {
    logitLens.value = {
      chosenChar,
      stepIndex,
      stages: [],
      loading: false,
      error: (e as Error).message,
    };
  }
}

export const config = signal<ArchConfig>({ ...defaultConfig });
export const name = signal<string>("Qwen3 8B");
export const entries = signal<ManifestEntry[]>([]);
export const previousCost = signal<CostPanel | null>(null);
export const previousConfig = signal<ArchConfig | null>(null);
export const generating = signal<boolean>(false);
export const lastError = signal<string | null>(null);
export const lastConsequences = signal<Array<{ id: string; text: string }>>([]);
export const instrumentation = signal<Instrumentation | null>(null);
export const telemetry = signal<TelemetryPoint[]>([]);

export const surgery = signal<SurgeryConfig>(emptySurgery());
export const kvDegradation = signal<KVDegradationConfig>(defaultKV());

export const raceConfigB = signal<ArchConfig>({
  ...defaultConfig,
  qk_norm: false,
});
export const raceNameB = signal<string>("Llama 3 8B");
export const raceKvB = signal<KVDegradationConfig>(defaultKV());
export const raceSurgeryB = signal<SurgeryConfig>(emptySurgery());

export function toggleHead(layer: number, head: number): void {
  const cur = surgery.value;
  const list = cur.disabled_heads[layer] ? [...cur.disabled_heads[layer]] : [];
  const idx = list.indexOf(head);
  if (idx >= 0) list.splice(idx, 1);
  else list.push(head);
  const nextHeads = { ...cur.disabled_heads };
  if (list.length === 0) delete nextHeads[layer];
  else nextHeads[layer] = list.sort((a, b) => a - b);
  surgery.value = { ...cur, disabled_heads: nextHeads };
}

export function toggleLayerPart(part: "attn" | "ffn" | "norm", layer: number): void {
  const cur = surgery.value;
  const key = part === "attn" ? "disabled_attn_layers" : part === "ffn" ? "disabled_ffn_layers" : "disabled_norm_layers";
  const list = [...cur[key]];
  const idx = list.indexOf(layer);
  if (idx >= 0) list.splice(idx, 1);
  else list.push(layer);
  list.sort((a, b) => a - b);
  surgery.value = { ...cur, [key]: list };
}

export function resetSurgery(): void {
  surgery.value = emptySurgery();
}

export function resetKV(): void {
  kvDegradation.value = defaultKV();
}

export function updateKV(patch: Partial<KVDegradationConfig>): void {
  kvDegradation.value = { ...kvDegradation.value, ...patch };
}

export const surgicalCount = computed<number>(() => {
  const s = surgery.value;
  let n = s.disabled_attn_layers.length + s.disabled_ffn_layers.length + s.disabled_norm_layers.length;
  for (const k in s.disabled_heads) n += s.disabled_heads[Number(k)].length;
  return n;
});

export const kvDegradationActive = computed<boolean>(() => {
  const k = kvDegradation.value;
  return k.quant !== "none" || k.window !== null || k.sink_mode !== "normal" || k.drop_prob > 0;
});

export function kvBytesMultiplier(quant: KVDegradationConfig["quant"]): number {
  if (quant === "int8") return 0.5;
  if (quant === "int4") return 0.25;
  if (quant === "int2") return 0.125;
  return 1;
}

export const effectiveKvPerToken = computed<number>(() => {
  const base = kvCachePerToken(config.value);
  return Math.round(base * kvBytesMultiplier(kvDegradation.value.quant));
});

export const effectiveKvFullCtx = computed<number>(() => {
  const cap = kvDegradation.value.window ? Math.min(config.value.ctx, kvDegradation.value.window) : config.value.ctx;
  return effectiveKvPerToken.value * cap;
});

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

const _gpu = typeof localStorage !== "undefined" ? localStorage.getItem("ablation-gpu") : null;
export const selectedGpu = signal<string>(_gpu ?? "vram_141");

export function setSelectedGpu(id: string): void {
  selectedGpu.value = id;
  try { localStorage.setItem("ablation-gpu", id); } catch {}
}

export const currentCost = computed<CostPanel>(() => costPanel(config.value));

export const currentHash = signal<string>("");
export const matchingEntry = computed<ManifestEntry | null>(() => {
  const h = currentHash.value;
  return entries.value.find(e => e.hash === h) ?? null;
});

const KIND_A_STRUCT_FIELDS: (keyof ArchConfig)[] = ["vocab_size", "d_model", "n_layers", "n_heads", "head_dim", "ctx", "ffn_mult"];

function posKindA(p: string): string { return p === "absolute" ? "absolute" : "rope"; }

function kindADiff(req: ArchConfig, entryArch: ArchConfig): string[] {
  const diffs: string[] = [];
  for (const f of KIND_A_STRUCT_FIELDS) if (req[f] !== entryArch[f]) diffs.push(f);
  if (req.norm !== entryArch.norm) diffs.push("norm");
  if (req.attention_type !== entryArch.attention_type) {
    diffs.push("attention_type");
  } else if (req.attention_type !== "mla") {
    if (req.n_kv_heads !== entryArch.n_kv_heads) diffs.push("n_kv_heads");
  }
  if (req.ffn !== entryArch.ffn) diffs.push("ffn");
  else if (req.ffn === "moe") {
    if (req.n_experts !== entryArch.n_experts) diffs.push("n_experts");
    if (req.top_k !== entryArch.top_k) diffs.push("top_k");
  }
  if (req.activation !== entryArch.activation) diffs.push("activation");
  if (Boolean(req.qk_norm) !== Boolean(entryArch.qk_norm)) diffs.push("qk_norm");
  if (posKindA(req.pos_embed) !== posKindA(entryArch.pos_embed)) diffs.push("pos_embed");
  return diffs;
}

export type ResolutionState = "green" | "amber" | "red";
export interface Resolution {
  state: ResolutionState;
  resolvedEntry: ManifestEntry | null;
  differingFields: string[];
  distance: number;
}

export function resolveLocal(cfg: ArchConfig, all: ManifestEntry[], exactHash?: string): Resolution {
  if (exactHash) {
    const exact = all.find(e => e.hash === exactHash) ?? null;
    if (exact) return { state: "green", resolvedEntry: exact, differingFields: [], distance: 0 };
  }
  let best: ManifestEntry | null = null;
  let bestDiffs: string[] = [];
  let bestDist = Infinity;
  for (const e of all) {
    const diffs = kindADiff(cfg, e.arch_config);
    if (diffs.length < bestDist) {
      bestDist = diffs.length;
      best = e;
      bestDiffs = diffs;
    }
  }
  if (best !== null && bestDist === 0) {
    return { state: "green", resolvedEntry: best, differingFields: [], distance: 0 };
  }
  if (best === null || bestDist > 2) {
    return { state: "red", resolvedEntry: null, differingFields: bestDiffs, distance: bestDist === Infinity ? -1 : bestDist };
  }
  return { state: "amber", resolvedEntry: best, differingFields: bestDiffs, distance: bestDist };
}

export const resolution = computed<Resolution>(() => resolveLocal(config.value, entries.value, currentHash.value));

export const kindBActive = computed<string[]>(() => {
  const c = config.value;
  const res = resolution.value;
  const a: string[] = [];
  if (!c.enable_attention) a.push("Attention off");
  if (!c.enable_ffn) a.push("FFN off");
  if (!c.enable_norms) a.push("Norms off");
  if (res.resolvedEntry) {
    const trained = res.resolvedEntry.arch_config.pos_embed;
    if (trained === "rope" && c.pos_embed !== "rope") a.push(`Position ${c.pos_embed}`);
    else if (trained === "absolute" && c.pos_embed === "nope") a.push("Position zeroed");
  }
  if (surgicalCount.value > 0) a.push(`${surgicalCount.value} surgical edits`);
  if (kvDegradationActive.value) a.push("KV degraded");
  return a;
});

export async function updateConfig(patch: Partial<ArchConfig>, fromPreset?: Preset): Promise<void> {
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
  if (fromPreset) {
    name.value = fromPreset.name;
  } else {
    const cur = PRESETS.find(p => p.name === name.value);
    if (cur && !configMatchesPreset(next, cur)) {
      name.value = `Modified ${cur.name}`;
    }
  }
  const { detectConsequences } = await import("./lib/consequences");
  lastConsequences.value = detectConsequences(previousConfig.value, previousCost.value, next, costPanel(next));
}

export async function setEntries(list: ManifestEntry[]): Promise<void> {
  entries.value = list;
  currentHash.value = await archHash(config.value);
}
