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
import { InstrumentationPanels } from "./components/InstrumentationPanels";
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
        <a class="gh-link" href="https://github.com/philippyt/ablation-lab" target="_blank" rel="noopener noreferrer" title="View source on GitHub" aria-label="GitHub repository">
          <svg viewBox="0 0 16 16" width="20" height="20" aria-hidden="true">
            <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
          </svg>
        </a>
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
          <InstrumentationPanels />
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
