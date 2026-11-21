import { signal } from "@preact/signals";
import { useEffect } from "preact/hooks";

export const whatIsThisOpen = signal<boolean>(false);

export function WhatIsThisModal() {
  if (!whatIsThisOpen.value) return null;

  useEffect(() => {
    function onEsc(e: KeyboardEvent) {
      if (e.key === "Escape") whatIsThisOpen.value = false;
    }
    document.addEventListener("keydown", onEsc);
    return () => document.removeEventListener("keydown", onEsc);
  }, []);

  return (
    <div class="modal-backdrop" onClick={() => (whatIsThisOpen.value = false)}>
      <div class="modal wit-modal" onClick={(e) => e.stopPropagation()}>
        <div class="modal-head">
          <div>
            <h2>About Ablation Lab</h2>
            <p class="panel-note">
              A live playground for composing tiny transformers, knocking out individual heads or layers, degrading KV caches, and racing twin models character by character.
            </p>
          </div>
          <button class="close" onClick={() => (whatIsThisOpen.value = false)} aria-label="close">×</button>
        </div>

        <div class="wit-section">
          <h3>What you can test</h3>
          <ul class="wit-list">
            <li><strong>Architecture presets</strong> Apply real model shapes like Llama 3, Qwen3, Mixtral, Gemma 3, and DeepSeek V3.2 at toy scale and inspect exact parameter and KV math.</li>
            <li><strong>Layer and head surgery</strong> Click any cell in the layer by head matrix to zero out individual query heads, attention blocks, FFNs, or norms live during inference.</li>
            <li><strong>KV cache degradation</strong> Quantize KV states down to INT2, shrink the sliding window, keep or drop attention sinks, or drop random cached keys.</li>
            <li><strong>Twin Race</strong> Run two models in lockstep on the same prompt, pinpoint the exact character where they diverge, and compare their top 5 next token probabilities.</li>
          </ul>
        </div>

        <div class="wit-section">
          <h3>Scale</h3>
          <ul class="wit-list">
            <li>Model width 128, 4 layers, 8 heads, context 256, char level vocabulary of 72 characters.</li>
            <li>169 separate checkpoints trained on roughly 1 MB of TinyStories so every architectural axis resolves to real weights.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
