import { useEffect } from "preact/hooks";
import { fetchEntries } from "./api";
import { setEntries, currentHash, name, lastError } from "./state";
import { loadContent } from "./content";
import { ConfigPicker } from "./components/ConfigPicker";
import { Presets } from "./components/Presets";
import { CollapsiblePanel } from "./components/CollapsiblePanel";
import { Diagram } from "./components/Diagram";
import { CostPanel, CostBasisHint } from "./components/CostPanel";
import { RunPanel } from "./components/RunPanel";
import { KVCacheState } from "./components/KVCacheState";
import { SidePanel, FloatingTooltip } from "./components/Explainer";
import { PyTorchCodeModal, codeModalOpen } from "./components/PyTorchCode";
import { AttentionGridModal } from "./components/AttentionGrid";
import { WhatIsThisModal, whatIsThisOpen } from "./components/WhatIsThisModal";

export function App() {
  useEffect(() => {
    fetchEntries()
      .then((es) => setEntries(es))
      .catch((e) => (lastError.value = e.message));
    loadContent().catch((e) => (lastError.value = e.message));
  }, []);

  return (
    <div class="app">
      <header>
        <h1>Ablation Lab</h1>
        <button class="wit-button" onClick={() => (whatIsThisOpen.value = true)}>What is this?</button>
        <div class="header-right">
          <span class="badge">{name.value}</span>
          <span class="hash">hash {currentHash.value || "—"}</span>
        </div>
      </header>

      <main class="workspace">
        <section class="panel canvas-panel">
          <div class="panel-head">
            <h2>Architecture</h2>
            <div class="panel-actions">
              <button class="ghost ghost-sm" onClick={() => (codeModalOpen.value = true)}>
                show as pytorch code
              </button>
            </div>
          </div>
          <Diagram />
        </section>

        <div class="bottom-row">
          <RunPanel />
          <KVCacheState />
        </div>
      </main>

      <aside class="sidebar">
        <CollapsiblePanel title="Component presets">
          <Presets />
        </CollapsiblePanel>
        <CollapsiblePanel title="Components">
          <ConfigPicker />
        </CollapsiblePanel>
        <CollapsiblePanel title="Cost" action={<CostBasisHint />}>
          <CostPanel />
        </CollapsiblePanel>
      </aside>

      <SidePanel />
      <FloatingTooltip />
      <PyTorchCodeModal />
      <AttentionGridModal />
      <WhatIsThisModal />
    </div>
  );
}
