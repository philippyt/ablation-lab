import { costPanel } from "../src/lib/costs.ts";
import { archHash } from "../src/lib/hash.ts";

const CONFIGS = [
  { name: "baseline_mha_ln_dense", cfg: base() },
  { name: "gqa", cfg: { ...base(), n_kv_heads: 2 } },
  { name: "mqa", cfg: { ...base(), n_kv_heads: 1 } },
  { name: "rmsnorm", cfg: { ...base(), norm: "rmsnorm" } },
  { name: "moe", cfg: { ...base(), ffn: "moe" } },
];

function base() {
  return {
    vocab_size: 72,
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
  };
}

let failed = 0;
for (const { name, cfg } of CONFIGS) {
  const r = await fetch("http://127.0.0.1:8000/api/cost", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config: cfg }),
  });
  const py = await r.json();
  const ts = costPanel(cfg);
  const keys = Object.keys(py);
  for (const k of keys) {
    if (py[k] !== ts[k]) {
      console.error(`[${name}] mismatch ${k}: py=${py[k]} ts=${ts[k]}`);
      failed++;
    }
  }
  const h = await archHash(cfg);
  console.log(`${name.padEnd(24)} hash=${h}  params=${ts.total_params}  kv@ctx=${ts.kv_cache_full_ctx}`);
}

if (failed > 0) {
  console.error(`\nFAILED: ${failed} mismatch(es)`);
  process.exit(1);
}
console.log("\nall configs match between Python and TS");
