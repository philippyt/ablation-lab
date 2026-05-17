import type { ArchConfig, CostPanel } from "../types";
import rulesData from "../content/rules.json";
import { fmtBytes, fmtParams } from "./format";

interface Rule {
  id: string;
  trigger: {
    fields_changed: string[];
    metric?: string;
    direction?: "increase" | "decrease" | "any";
    min_relative_change?: number;
    new_value_in?: Record<string, string[]>;
    old_value_in?: Record<string, string[]>;
  };
  template: string;
}

const RULES: Rule[] = rulesData as Rule[];

export interface ConsequenceMatch {
  id: string;
  text: string;
}

function fieldsChanged(oldCfg: any, newCfg: any): Set<string> {
  const out = new Set<string>();
  for (const k of Object.keys(newCfg)) {
    if (oldCfg[k] !== newCfg[k]) out.add(k);
  }
  return out;
}

function fmtPct(rel: number): string {
  const pct = Math.abs(rel) * 100;
  if (pct < 1) return `${pct.toFixed(1)}%`;
  return `${pct.toFixed(0)}%`;
}

function ratioInverse(oldV: number, newV: number): string {
  if (newV === 0) return "∞";
  const r = oldV / newV;
  if (r >= 2) return r.toFixed(0);
  return r.toFixed(1);
}

function buildContext(oldCfg: ArchConfig, oldCost: CostPanel, newCfg: ArchConfig, newCost: CostPanel, metric: string | undefined): Record<string, string> {
  const ctx: Record<string, string> = {};
  for (const k of Object.keys(newCfg)) {
    ctx[`old.${k}`] = String((oldCfg as any)[k]);
    ctx[`new.${k}`] = String((newCfg as any)[k]);
  }
  if (metric) {
    const oldV = (oldCost as any)[metric] as number;
    const newV = (newCost as any)[metric] as number;
    if (oldV !== undefined && newV !== undefined && oldV !== 0) {
      ctx["pct"] = fmtPct((newV - oldV) / oldV);
      ctx["ratio"] = ratioInverse(oldV, newV);
      if (metric.includes("kv") || metric.includes("memory")) {
        ctx["bytes.old"] = fmtBytes(oldV);
        ctx["bytes.new"] = fmtBytes(newV);
      } else {
        ctx["params.old"] = fmtParams(oldV);
        ctx["params.new"] = fmtParams(newV);
        ctx["params.diff"] = fmtParams(Math.abs(newV - oldV));
      }
    }
  }
  ctx["bytes.kv.old"] = fmtBytes(oldCost.kv_cache_per_token);
  ctx["bytes.kv.new"] = fmtBytes(newCost.kv_cache_per_token);
  ctx["pos_size"] = fmtParams(newCfg.ctx * newCfg.d_model);
  ctx["head_size"] = fmtParams(newCfg.vocab_size * newCfg.d_model);
  ctx["qkn_size"] = fmtParams((newCfg.n_heads + newCfg.n_kv_heads) * newCfg.head_dim);
  ctx["tied_state"] = newCfg.tie_embeddings ? "share weights" : "are separate matrices";
  return ctx;
}

function applyTemplate(tpl: string, ctx: Record<string, string>): string {
  return tpl.replace(/\{([^}]+)\}/g, (_, key) => ctx[key] ?? `{${key}}`);
}

export function detectConsequences(
  oldCfg: ArchConfig | null,
  oldCost: CostPanel | null,
  newCfg: ArchConfig,
  newCost: CostPanel,
): ConsequenceMatch[] {
  if (!oldCfg || !oldCost) return [];
  const changed = fieldsChanged(oldCfg, newCfg);
  if (changed.size === 0) return [];

  const matches: ConsequenceMatch[] = [];
  for (const rule of RULES) {
    if (!rule.trigger.fields_changed.every((f) => changed.has(f))) continue;
    if (rule.trigger.new_value_in) {
      let ok = true;
      for (const [k, allowed] of Object.entries(rule.trigger.new_value_in)) {
        if (!allowed.includes(String((newCfg as any)[k]))) { ok = false; break; }
      }
      if (!ok) continue;
    }
    if (rule.trigger.old_value_in) {
      let ok = true;
      for (const [k, allowed] of Object.entries(rule.trigger.old_value_in)) {
        if (!allowed.includes(String((oldCfg as any)[k]))) { ok = false; break; }
      }
      if (!ok) continue;
    }
    if (rule.trigger.metric) {
      const oldV = (oldCost as any)[rule.trigger.metric] as number;
      const newV = (newCost as any)[rule.trigger.metric] as number;
      if (oldV === undefined || newV === undefined) continue;
      const dir = rule.trigger.direction ?? "any";
      if (dir === "decrease" && newV >= oldV) continue;
      if (dir === "increase" && newV <= oldV) continue;
      if (rule.trigger.min_relative_change !== undefined) {
        const rel = oldV === 0 ? Infinity : Math.abs((newV - oldV) / oldV);
        if (rel < rule.trigger.min_relative_change) continue;
      }
    }
    const ctx = buildContext(oldCfg, oldCost, newCfg, newCost, rule.trigger.metric);
    matches.push({ id: rule.id, text: applyTemplate(rule.template, ctx) });
  }
  return matches.slice(0, 3);
}
