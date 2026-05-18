import { config, name, updateConfig } from "../state";
import { attentionKind } from "../lib/costs";
import { ffnLabel, HINTS } from "../lib/labels";
import { HelpHint } from "./HelpHint";

const ATTENTION_OPTIONS: Array<{ value: string; label: string; n_kv_heads: number; attention_type: "standard" | "mla" }> = [
  { value: "mha", label: "MHA - Multi-Head", n_kv_heads: 8, attention_type: "standard" },
  { value: "gqa", label: "GQA - Grouped-Query", n_kv_heads: 2, attention_type: "standard" },
  { value: "mqa", label: "MQA - Multi-Query", n_kv_heads: 1, attention_type: "standard" },
  { value: "mla", label: "MLA - Multi-head Latent", n_kv_heads: 1, attention_type: "mla" },
];

const POS_OPTIONS = [
  { value: "rope", label: "RoPE" },
  { value: "alibi", label: "ALiBi" },
  { value: "absolute", label: "Absolute learned" },
  { value: "sinusoidal", label: "Sinusoidal" },
  { value: "nope", label: "NoPE" },
];

const ACT_OPTIONS = [
  { value: "silu", label: "SiLU / Swish" },
  { value: "gelu", label: "GELU" },
  { value: "relu2", label: "ReLU² (squared)" },
];

interface RowProps {
  label: string;
  hint: string;
  children: any;
}

function Row({ label, hint, children }: RowProps) {
  return (
    <>
      <label>
        {label}
        <HelpHint text={hint} />
      </label>
      {children}
    </>
  );
}

export function ConfigPicker() {
  const cfg = config.value;
  const att = attentionKind(cfg);

  return (
    <>
        <div class="row">
          <Row label="Name" hint={HINTS.name}>
            <input
              type="text"
              value={name.value}
              onInput={(e) => (name.value = (e.target as HTMLInputElement).value)}
            />
          </Row>

          <Row label="Attention" hint={HINTS.attention}>
            <select
              value={att}
              onChange={(e) => {
                const v = (e.target as HTMLSelectElement).value;
                const opt = ATTENTION_OPTIONS.find((o) => o.value === v)!;
                updateConfig({ n_kv_heads: opt.n_kv_heads, attention_type: opt.attention_type });
              }}
            >
              {ATTENTION_OPTIONS.map((o) => (
                <option value={o.value}>{o.label}</option>
              ))}
            </select>
          </Row>

          <Row label="Position" hint={HINTS.position}>
            <select
              value={cfg.pos_embed}
              onChange={(e) => updateConfig({ pos_embed: (e.target as HTMLSelectElement).value as any })}
            >
              {POS_OPTIONS.map((o) => (
                <option value={o.value}>{o.label}</option>
              ))}
            </select>
          </Row>

          <Row label="Norm" hint={HINTS.norm}>
            <select
              value={cfg.norm}
              onChange={(e) => updateConfig({ norm: (e.target as HTMLSelectElement).value as "layernorm" | "rmsnorm" })}
            >
              <option value="layernorm">LayerNorm</option>
              <option value="rmsnorm">RMSNorm</option>
            </select>
          </Row>

          <Row label="QK-Norm" hint={HINTS.qk_norm}>
            <select
              value={cfg.qk_norm ? "on" : "off"}
              onChange={(e) => updateConfig({ qk_norm: (e.target as HTMLSelectElement).value === "on" })}
            >
              <option value="off">off</option>
              <option value="on">on</option>
            </select>
          </Row>

          <Row label="FFN" hint={HINTS.ffn}>
            <select
              value={cfg.ffn}
              onChange={(e) => updateConfig({ ffn: (e.target as HTMLSelectElement).value as "dense" | "moe" })}
            >
              <option value="dense">{ffnLabel({ ...cfg, ffn: "dense" })}</option>
              <option value="moe">{ffnLabel({ ...cfg, ffn: "moe" })}</option>
            </select>
          </Row>

          <Row label="Activation" hint={HINTS.activation}>
            <select
              value={cfg.activation}
              onChange={(e) => updateConfig({ activation: (e.target as HTMLSelectElement).value as any })}
            >
              {ACT_OPTIONS.map((o) => (
                <option value={o.value}>{o.label}</option>
              ))}
            </select>
          </Row>

          <Row label="Tied embed" hint={HINTS.tied}>
            <select
              value={cfg.tie_embeddings ? "tied" : "untied"}
              onChange={(e) => updateConfig({ tie_embeddings: (e.target as HTMLSelectElement).value === "tied" })}
            >
              <option value="tied">tied</option>
              <option value="untied">untied</option>
            </select>
          </Row>

          <Row label="Layers" hint={HINTS.layers}>
            <select
              value={cfg.n_layers}
              onChange={(e) => updateConfig({ n_layers: parseInt((e.target as HTMLSelectElement).value, 10) })}
            >
              {[2, 4, 6, 8].map((n) => (
                <option value={n}>{n}</option>
              ))}
            </select>
          </Row>

          <Row label="d_model" hint={HINTS.d_model}>
            <select
              value={cfg.d_model}
              onChange={(e) => updateConfig({ d_model: parseInt((e.target as HTMLSelectElement).value, 10) })}
            >
              {[64, 128, 256].map((n) => (
                <option value={n}>{n}</option>
              ))}
            </select>
          </Row>

          <Row label="Context" hint={HINTS.ctx}>
            <select
              value={cfg.ctx}
              onChange={(e) => updateConfig({ ctx: parseInt((e.target as HTMLSelectElement).value, 10) })}
            >
              {[64, 128, 256, 512].map((n) => (
                <option value={n}>{n}</option>
              ))}
            </select>
          </Row>

          <Row label="Attn sublayer" hint={HINTS.enable_attention}>
            <select
              value={cfg.enable_attention ? "on" : "off"}
              onChange={(e) => updateConfig({ enable_attention: (e.target as HTMLSelectElement).value === "on" })}
            >
              <option value="on">on</option>
              <option value="off">off (ablate)</option>
            </select>
          </Row>

          <Row label="FFN sublayer" hint={HINTS.enable_ffn}>
            <select
              value={cfg.enable_ffn ? "on" : "off"}
              onChange={(e) => updateConfig({ enable_ffn: (e.target as HTMLSelectElement).value === "on" })}
            >
              <option value="on">on</option>
              <option value="off">off (ablate)</option>
            </select>
          </Row>

          <Row label="Norm layers" hint={HINTS.enable_norms}>
            <select
              value={cfg.enable_norms ? "on" : "off"}
              onChange={(e) => updateConfig({ enable_norms: (e.target as HTMLSelectElement).value === "on" })}
            >
              <option value="on">on</option>
              <option value="off">off (ablate)</option>
            </select>
          </Row>
        </div>
    </>
  );
}
