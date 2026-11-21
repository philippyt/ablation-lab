import type { ArchConfig, ManifestEntry, GenerateRequest, RaceRequest, TopProb, InspectRequest, InspectResponse } from "./types";

export async function fetchEntries(): Promise<ManifestEntry[]> {
  const res = await fetch("/api/configs");
  if (!res.ok) throw new Error(`Could not load configs (${res.status})`);
  const data = await res.json();
  return data.entries as ManifestEntry[];
}

export type StreamEvent =
  | { event: "start"; prompt_tokens: number; ctx: number; hash: string }
  | { event: "token"; text: string; lm?: number; ent?: number; rn?: number; s?: number }
  | { event: "done"; val_loss: number; hash: string; stop_reason?: "eos" | "max_tokens" | "ctx_limit"; tokens?: number; elapsed_s?: number }
  | { event: "instr"; [k: string]: any }
  | { event: "error"; message: string };

export type RaceEvent =
  | { event: "race_start"; hash_a: string; hash_b: string; val_loss_a: number; val_loss_b: number; state_a: string; state_b: string }
  | { event: "race_step"; step: number; a: string; b: string; diff: boolean; lm_a: number; lm_b: number; ent_a: number; ent_b: number; top_a?: TopProb[]; top_b?: TopProb[] }
  | { event: "race_done"; first_diverge: number | null; stop_reason_a: string; stop_reason_b: string; tokens_a: number; tokens_b: number; elapsed_s: number }
  | { event: "error"; message: string };

async function* streamSSE<T>(url: string, body: unknown, signal?: AbortSignal): AsyncGenerator<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    yield { event: "error", message: humanizeError(res.status, err) } as T;
    return;
  }
  if (!res.body) {
    yield { event: "error", message: "Empty response body" } as T;
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const SEP = /\r?\n\r?\n/;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    while (true) {
      const m = SEP.exec(buf);
      if (!m) break;
      const raw = buf.slice(0, m.index);
      buf = buf.slice(m.index + m[0].length);
      const parsed = parseSSE<T>(raw);
      if (parsed) yield parsed;
    }
  }
}

export function streamGenerate(req: GenerateRequest, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
  return streamSSE<StreamEvent>("/api/generate", req, signal);
}

export function streamRace(req: RaceRequest, signal?: AbortSignal): AsyncGenerator<RaceEvent> {
  return streamSSE<RaceEvent>("/api/race", req, signal);
}

export async function fetchInspect(req: InspectRequest): Promise<InspectResponse> {
  const res = await fetch("/api/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(humanizeError(res.status, err));
  }
  return (await res.json()) as InspectResponse;
}

function humanizeError(status: number, err: any): string {
  if (status === 422 && Array.isArray(err?.detail)) {
    const parts: string[] = [];
    for (const e of err.detail) {
      const field = Array.isArray(e.loc) ? e.loc[e.loc.length - 1] : "value";
      const ctx = e.ctx || {};
      if (e.type === "less_than_equal") parts.push(`${field} must be at most ${ctx.le}`);
      else if (e.type === "greater_than_equal") parts.push(`${field} must be at least ${ctx.ge}`);
      else if (e.type === "less_than") parts.push(`${field} must be less than ${ctx.lt}`);
      else if (e.type === "greater_than") parts.push(`${field} must be greater than ${ctx.gt}`);
      else parts.push(`${field} ${e.msg || "is invalid"}`);
    }
    return parts.join(", ");
  }
  if (typeof err?.detail === "string") return err.detail;
  if (typeof err?.detail?.message === "string") return err.detail.message;
  return `Request failed (${status})`;
}

function parseSSE<T>(block: string): T | null {
  let event = "message";
  let data = "";
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    const parsed = JSON.parse(data);
    return { event, ...parsed } as T;
  } catch {
    return null;
  }
}

export interface CostRequest {
  config: ArchConfig;
  dtype_bytes?: number;
  batch_size?: number;
}
