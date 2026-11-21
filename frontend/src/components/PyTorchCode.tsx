import { useState } from "preact/hooks";
import { signal } from "@preact/signals";
import { config } from "../state";
import { composeCode, highlightPython } from "../content/code";

export const codeModalOpen = signal<boolean>(false);

export function PyTorchCodeModal() {
  if (!codeModalOpen.value) return null;
  const cfg = config.value;
  const code = composeCode(cfg);
  const html = highlightPython(code);
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div class="modal-backdrop" onClick={() => (codeModalOpen.value = false)}>
      <div class="modal pytorch-modal" onClick={(e) => e.stopPropagation()}>
        <div class="modal-head">
          <div>
            <h3>Architecture in PyTorch</h3>
            <p class="panel-note">Your configured architecture exported as a standalone PyTorch module.</p>
          </div>
          <div class="modal-actions">
            <button class="btn-outline" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
            <button class="close" onClick={() => (codeModalOpen.value = false)}>×</button>
          </div>
        </div>
        <pre class="code"><code dangerouslySetInnerHTML={{ __html: html }} /></pre>
      </div>
    </div>
  );
}
