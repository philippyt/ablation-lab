import {
  labMode,
  config,
  name,
  surgery,
  kvDegradation,
  raceConfigB,
  raceNameB,
  raceKvB,
  updateConfig,
} from "../state";

interface SharePayload {
  m?: "single" | "race";
  n?: string;
  c?: Record<string, unknown>;
  s?: Record<string, unknown>;
  k?: Record<string, unknown>;
  nb?: string;
  cb?: Record<string, unknown>;
  kb?: Record<string, unknown>;
}

function toBase64Url(str: string): string {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromBase64Url(b64: string): string {
  const norm = b64.replace(/-/g, "+").replace(/_/g, "/");
  const pad = norm.length % 4 === 0 ? "" : "=".repeat(4 - (norm.length % 4));
  const bin = atob(norm + pad);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

export function buildShareUrl(): string {
  const payload: SharePayload = {
    m: labMode.value,
    n: name.value,
    c: config.value as unknown as Record<string, unknown>,
    s: surgery.value as unknown as Record<string, unknown>,
    k: kvDegradation.value as unknown as Record<string, unknown>,
    nb: raceNameB.value,
    cb: raceConfigB.value as unknown as Record<string, unknown>,
    kb: raceKvB.value as unknown as Record<string, unknown>,
  };
  const encoded = toBase64Url(JSON.stringify(payload));
  const base = window.location.href.split("#")[0];
  return `${base}#exp=${encoded}`;
}

export function restoreFromUrlHash(): void {
  try {
    const hash = window.location.hash;
    if (!hash.startsWith("#exp=")) return;
    const raw = fromBase64Url(hash.slice(5));
    const p = JSON.parse(raw) as SharePayload;
    if (p.m === "single" || p.m === "race") labMode.value = p.m;
    if (p.c && typeof p.c === "object") {
      updateConfig(p.c as any);
    }
    if (typeof p.n === "string" && p.n.length > 0) name.value = p.n;
    if (p.s && typeof p.s === "object") surgery.value = p.s as any;
    if (p.k && typeof p.k === "object") kvDegradation.value = p.k as any;
    if (typeof p.nb === "string" && p.nb.length > 0) raceNameB.value = p.nb;
    if (p.cb && typeof p.cb === "object") raceConfigB.value = p.cb as any;
    if (p.kb && typeof p.kb === "object") raceKvB.value = p.kb as any;
  } catch {
    // Ignore malformed share hashes
  }
}
