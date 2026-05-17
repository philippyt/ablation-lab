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
    <div class="modal-backdrop wit-backdrop" onClick={() => (whatIsThisOpen.value = false)}>
      <div class="wit-modal" onClick={(e) => e.stopPropagation()}>
        <button class="close wit-close" onClick={() => (whatIsThisOpen.value = false)} aria-label="close">×</button>
        <h2>What is this?</h2>
        <p class="wit-lead">
          A tiny playground for composing tiny transformer architectures and seeing how inference output breaks.
        </p>

        <h3>What it does</h3>
        <ul class="wit-list">
          <li>Pick the attention paradigm (MHA, GQA, MQA, MLA), normalization (LayerNorm, RMSNorm), FFN shape (dense or MoE), activation, positional scheme, and ablation toggles.</li>
          <li>See the exact parameter count, KV cache size, memory footprint, and decode tokens-per-second for the current architecture.</li>
          <li>Load a real trained checkpoint that matches the architecture. Stream generated text back from it.</li>
          <li>Turn attention or the FFN off at inference. Swap the positional scheme at inference. The trained weights stay the same. Output degrades visibly (points to you if you figure out why).</li>
          <li>Export the current architecture as a paste-and-run PyTorch snippet.</li>
        </ul>

        <h3>Scale</h3>
        <ul class="wit-list">
          <li>d_model 128, n_layers 4, ctx 256, char-level vocabulary.</li>
          <li>Trained on roughly 1 MB of TinyStories.</li>
          <li>169 separate trained checkpoints, one per distinct architecture in the lab's matrix.</li>
        </ul>

        <h3>What's in the UI</h3>
        <ul class="wit-list">
          <li><strong>Architecture diagram</strong>: the live block diagram of the current selection. Click any component box for its explainer.</li>
          <li><strong>Component presets</strong>: one-click recipes for real models (Llama 3, Mixtral, DeepSeek V3, Kimi K2.5, etc.) applied at the lab's toy scale.</li>
          <li><strong>Components</strong>: per-axis pickers and ablation toggles.</li>
          <li><strong>Cost</strong>: parameters, KV cache, memory, FFN hidden, decode roofline against a selected VRAM tier.</li>
          <li><strong>Run</strong>: prompt, max tokens, temperature, real streamed output from the matching checkpoint.</li>
          <li><strong>KV cache state</strong>: live cache fill, eviction count, equation breakdown of bytes-per-token, attention pattern viewer.</li>
        </ul>
      </div>
    </div>
  );
}
