const cfg = { vocab_size:72,d_model:128,n_layers:4,n_heads:8,n_kv_heads:8,head_dim:16,ctx:128,norm:"layernorm",ffn:"dense",ffn_mult:4,n_experts:4,top_k:1,tie_embeddings:true,pos_embed:"rope",activation:"silu",qk_norm:false };
const res = await fetch("http://127.0.0.1:8000/api/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ config: cfg, prompt: "Once", max_tokens: 4, temperature: 0.8 }),
});
const reader = res.body.getReader();
let buf = new Uint8Array(0);
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  const merged = new Uint8Array(buf.length + value.length);
  merged.set(buf); merged.set(value, buf.length);
  buf = merged;
}
console.log("byte length:", buf.length);
console.log("hex of first 200 bytes:");
const hex = Array.from(buf.slice(0, 200)).map(b => b.toString(16).padStart(2,"0")).join(" ");
console.log(hex);
console.log("\ntext (with visible newlines):");
console.log(new TextDecoder().decode(buf).replace(/\n/g, "\\n\n").replace(/\r/g, "\\r"));
