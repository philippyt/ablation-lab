const cfg = {
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

const t0 = Date.now();
const res = await fetch("http://127.0.0.1:8000/api/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    config: cfg,
    prompt: "The little bear walked into the forest and",
    max_tokens: 40,
    temperature: 0.8,
  }),
});
console.log(`HTTP ${res.status} after ${Date.now() - t0}ms`);
if (!res.ok) {
  console.log(await res.text());
  process.exit(1);
}

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buf = "";
let count = 0;
let chunks = 0;
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  chunks++;
  buf += decoder.decode(value, { stream: true });
  const SEP = /\r?\n\r?\n/;
  while (true) {
    const m = SEP.exec(buf);
    if (!m) break;
    const raw = buf.slice(0, m.index);
    buf = buf.slice(m.index + m[0].length);
    let event = "message";
    let data = "";
    for (const line of raw.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    const parsed = data ? JSON.parse(data) : null;
    count++;
    const ms = Date.now() - t0;
    if (event === "token") {
      process.stdout.write(`[${ms}ms tok#${count}] ${JSON.stringify(parsed.text)}  `);
    } else {
      console.log(`\n[${ms}ms] event=${event} data=${JSON.stringify(parsed)}`);
    }
  }
}
console.log(`\n${chunks} reader chunks, ${count} parsed events, ${Date.now() - t0}ms total`);
