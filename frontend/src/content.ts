import { signal } from "@preact/signals";

export interface Explainer {
  id: string;
  level: string;
  title: string;
  short: string;
  body: string;
  related_concepts: string[];
  sources?: string[];
}

export interface ContentIndex {
  components: Record<string, Explainer>;
  concepts: Record<string, Explainer>;
}

export const content = signal<ContentIndex | null>(null);
export const openExplainerId = signal<string | null>(null);

export async function loadContent(): Promise<void> {
  const res = await fetch("/api/content");
  if (!res.ok) throw new Error(`/api/content → ${res.status}`);
  content.value = (await res.json()) as ContentIndex;
}

export function getExplainer(id: string | null): Explainer | null {
  if (!id || !content.value) return null;
  return content.value.components[id] ?? content.value.concepts[id] ?? null;
}
