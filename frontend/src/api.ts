import type { ArchConfig, ManifestEntry, GenerateRequest } from "./types";

export async function fetchEntries(): Promise<ManifestEntry[]> {
  const res = await fetch("/api/configs");
  if (!res.ok) throw new Error(`/api/configs failed: ${res.status}`);
  const data = await res.json();
  return data.entries as ManifestEntry[];
}

export type StreamEvent =
  | { event: "start"; prompt_tokens: number; ctx: number; hash: string }
  | { event: "token"; text: string }
  | { event: "done"; val_loss: number; hash: string; stop_reason?: "eos" | "max_tokens" | "ctx_limit"; tokens?: number; elapsed_s?: number }
  | { event: "instr"; [k: string]: any }
  | { event: "error"; message: string };

export async function* streamGenerate(req: GenerateRequest, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    yield { event: "error", message: humanizeError(res.status, err) };
    return;
  }
  if (!res.body) {
    yield { event: "error", message: "empty body" };
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
      const parsed = parseSSE(raw);
      if (parsed) yield parsed;
    }
  }
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
      else parts.push(`${field}: ${e.msg || "invalid"}`);
    }
    return parts.join(" · ");
  }
  if (typeof err?.detail === "string") return err.detail;
  if (typeof err?.detail?.message === "string") return err.detail.message;
  return `request failed (${status})`;
}

function parseSSE(block: string): StreamEvent | null {
  let event = "message";
  let data = "";
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    const parsed = JSON.parse(data);
    return { event, ...parsed } as StreamEvent;
  } catch {
    return null;
  }
}

export interface CostRequest {
  config: ArchConfig;
  dtype_bytes?: number;
  batch_size?: number;
}
