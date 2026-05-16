import type { ArchConfig } from "../types";

const ARCH_FIELDS = [
  "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim",
  "ctx", "norm", "ffn", "ffn_mult", "n_experts", "top_k", "tie_embeddings",
  "pos_embed", "activation", "qk_norm", "attention_type",
  "enable_attention", "enable_ffn", "enable_norms",
] as const;

function canonicalJSON(obj: Record<string, unknown>): string {
  const keys = Object.keys(obj).sort();
  const pairs = keys.map(k => `${JSON.stringify(k)}:${JSON.stringify(obj[k])}`);
  return `{${pairs.join(",")}}`;
}

function hashableDict(cfg: ArchConfig): Record<string, unknown> {
  const d: Record<string, unknown> = {};
  for (const k of ARCH_FIELDS) d[k] = cfg[k];
  if (d.ffn !== "moe") {
    delete d.n_experts;
    delete d.top_k;
  }
  return d;
}

export async function archHash(cfg: ArchConfig): Promise<string> {
  const blob = canonicalJSON(hashableDict(cfg));
  const buf = new TextEncoder().encode(blob);
  const digest = await crypto.subtle.digest("SHA-256", buf);
  const bytes = Array.from(new Uint8Array(digest));
  const hex = bytes.map(b => b.toString(16).padStart(2, "0")).join("");
  return hex.slice(0, 12);
}
