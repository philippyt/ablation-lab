import { useEffect, useState } from "preact/hooks";
import { fetchEntries } from "./api";
import {
  setEntries,
  name,
  lastError,
  labMode,
  labTab,
  surgicalCount,
  kvDegradationActive,
} from "./state";
import { loadContent } from "./content";
import { buildShareUrl, restoreFromUrlHash } from "./lib/share";
import { ConfigPicker } from "./components/ConfigPicker";
import { Presets } from "./components/Presets";
import { Diagram } from "./components/Diagram";
import { CostPanel } from "./components/CostPanel";
import { RunPanel } from "./components/RunPanel";
import { KVCacheState } from "./components/KVCacheState";
import { InstrumentationPanels } from "./components/InstrumentationPanels";
import { SurgeryPanel } from "./components/SurgeryPanel";
import { TwinRacePanel } from "./components/TwinRacePanel";
import { SidePanel, FloatingTooltip } from "./components/Explainer";
import { PyTorchCodeModal, codeModalOpen } from "./components/PyTorchCode";
import { AttentionGridModal } from "./components/AttentionGrid";
import { WhatIsThisModal, whatIsThisOpen } from "./components/WhatIsThisModal";

export function App() {
  const [sideTab, setSideTab] = useState<"build" | "presets" | "cost">("build");
  const [copiedLink, setCopiedLink] = useState(false);

  useEffect(() => {
    restoreFromUrlHash();
    fetchEntries()
      .then((es) => setEntries(es))
      .catch((e) => (lastError.value = e.message));
    loadContent().catch((e) => (lastError.value = e.message));
  }, []);

  async function shareExperiment() {
    const url = buildShareUrl();
    window.history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url);
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 1600);
    } catch {
      setCopiedLink(false);
    }
  }

  const mode = labMode.value;
  const activeLabTab = labTab.value;
  const sCount = surgicalCount.value;
  const kvOn = kvDegradationActive.value;

  return (
    <div class="app">
      <header class="topbar">
        <div class="brand">
          <h1>Ablation Lab</h1>
          <span class="model-tag">{name.value}</span>
        </div>

        <div class="mode-switch" role="tablist" aria-label="Workspace mode">
          <button
            class={`mode-btn ${mode === "single" ? "is-active" : ""}`}
            onClick={() => (labMode.value = "single")}
          >
            Single Run
          </button>
          <button
            class={`mode-btn ${mode === "race" ? "is-active" : ""}`}
            onClick={() => (labMode.value = "race")}
          >
            Twin Race
          </button>
        </div>

        <div class="topbar-actions">
          <button class="btn-outline" onClick={shareExperiment}>
            {copiedLink ? "Link copied" : "Share link"}
          </button>
          <button class="btn-outline" onClick={() => (codeModalOpen.value = true)}>
            PyTorch code
          </button>
          <button class="btn-outline" onClick={() => (whatIsThisOpen.value = true)}>
            About
          </button>
          <a
            class="icon-btn"
            href="https://github.com/philippyt/ablation-lab"
            target="_blank"
            rel="noopener noreferrer"
            title="View source on GitHub"
            aria-label="GitHub repository"
          >
            <svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true">
              <path
                fill="currentColor"
                fill-rule="evenodd"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
              />
            </svg>
          </a>
        </div>
      </header>

      <main class="workspace">
        <section class="card diagram-card">
          <Diagram />
        </section>

        {mode === "single" ? (
          <div class="stage-grid">
            <RunPanel />

            <section class="card lab-card">
              <div class="tab-bar">
                <button
                  class={`tab-btn ${activeLabTab === "surgery" ? "is-active" : ""}`}
                  onClick={() => (labTab.value = "surgery")}
                >
                  Surgery {sCount > 0 ? `(${sCount})` : ""}
                </button>
                <button
                  class={`tab-btn ${activeLabTab === "kv" ? "is-active" : ""}`}
                  onClick={() => (labTab.value = "kv")}
                >
                  KV Cache {kvOn ? "(mod)" : ""}
                </button>
                <button
                  class={`tab-btn ${activeLabTab === "internals" ? "is-active" : ""}`}
                  onClick={() => (labTab.value = "internals")}
                >
                  Internals
                </button>
              </div>
              <div class="tab-body">
                {activeLabTab === "surgery" && <SurgeryPanel />}
                {activeLabTab === "kv" && <KVCacheState />}
                {activeLabTab === "internals" && <InstrumentationPanels />}
              </div>
            </section>
          </div>
        ) : (
          <section class="card race-card">
            <TwinRacePanel />
          </section>
        )}
      </main>

      <aside class="sidebar">
        <div class="tab-bar sidebar-tabs">
          <button
            class={`tab-btn ${sideTab === "build" ? "is-active" : ""}`}
            onClick={() => setSideTab("build")}
          >
            Build
          </button>
          <button
            class={`tab-btn ${sideTab === "presets" ? "is-active" : ""}`}
            onClick={() => setSideTab("presets")}
          >
            Presets
          </button>
          <button
            class={`tab-btn ${sideTab === "cost" ? "is-active" : ""}`}
            onClick={() => setSideTab("cost")}
          >
            Cost
          </button>
        </div>

        <div class="sidebar-body">
          {sideTab === "build" && <ConfigPicker />}
          {sideTab === "presets" && <Presets />}
          {sideTab === "cost" && <CostPanel />}
        </div>
      </aside>

      <SidePanel />
      <FloatingTooltip />
      <PyTorchCodeModal />
      <AttentionGridModal />
      <WhatIsThisModal />
    </div>
  );
}
