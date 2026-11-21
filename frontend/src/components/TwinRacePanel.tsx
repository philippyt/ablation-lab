import { useRef, useState } from "preact/hooks";
import {
  config,
  name,
  entries,
  surgery,
  kvDegradation,
  raceConfigB,
  raceNameB,
  raceKvB,
  raceSurgeryB,
  defaultConfig,
  defaultKV,
  emptySurgery,
  updateConfig,
  resolveLocal,
} from "../state";
import { PRESETS, type Preset } from "../lib/presets";
import { streamRace } from "../api";
import type { ArchConfig, RaceStep, KVDegradationConfig } from "../types";

function escapeVisible(ch: string): string {
  if (ch === " ") return "space";
  if (ch === "\n") return "newline";
  if (ch === "\t") return "tab";
  return ch;
}

function applyPresetToConfig(base: ArchConfig, p: Preset): ArchConfig {
  return { ...base, ...p.patch };
}

interface OverlaySparkProps {
  title: string;
  seriesA: number[];
  seriesB: number[];
  labelA: string;
  labelB: string;
}

function OverlaySpark({ title, seriesA, seriesB, labelA, labelB }: OverlaySparkProps) {
  const W = 300;
  const H = 64;
  const PAD_X = 6;
  const PAD_Y = 8;
  const all = [...seriesA, ...seriesB];
  const maxV = all.length ? Math.max(...all, 0.01) : 1;
  const minV = all.length ? Math.min(...all, 0) : 0;
  const span = Math.max(maxV - minV, 0.01);
  const maxLen = Math.max(seriesA.length, seriesB.length, 2);

  const toPts = (arr: number[]) =>
    arr
      .map((v, i) => {
        const x = PAD_X + (i / (maxLen - 1)) * (W - PAD_X * 2);
        const y = PAD_Y + (1 - (v - minV) / span) * (H - PAD_Y * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <div class="race-spark-card">
      <div class="race-spark-head">
        <span class="spark-title">{title}</span>
        <div class="race-legend">
          <span class="leg-a">{labelA}</span>
          <span class="leg-b">{labelB}</span>
        </div>
      </div>
      <svg class="spark-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {seriesA.length >= 2 && (
          <polyline points={toPts(seriesA)} fill="none" stroke="#1d4ed8" stroke-width="1.8" />
        )}
        {seriesB.length >= 2 && (
          <polyline points={toPts(seriesB)} fill="none" stroke="#c2410c" stroke-width="1.8" />
        )}
      </svg>
    </div>
  );
}

export function TwinRacePanel() {
  const [prompt, setPrompt] = useState("Once upon a time");
  const [maxTokens, setMaxTokens] = useState(420);
  const [temperature, setTemperature] = useState(0);
  const [racing, setRacing] = useState(false);
  const [steps, setSteps] = useState<RaceStep[]>([]);
  const [firstDiverge, setFirstDiverge] = useState<number | null>(null);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [statusMsg, setStatusMsg] = useState<string>("Pick a matchup or compare your current config against any rival.");
  const [statusErr, setStatusErr] = useState(false);
  const [stopReasons, setStopReasons] = useState<{ a: string; b: string } | null>(null);
  const aborterRef = useRef<AbortController | null>(null);

  const cfgA = config.value;
  const cfgB = raceConfigB.value;
  const kvA = kvDegradation.value;
  const kvB = raceKvB.value;
  const resA = resolveLocal(cfgA, entries.value);
  const resB = resolveLocal(cfgB, entries.value);
  const canRace = resA.state !== "red" && resB.state !== "red";

  function stopRace() {
    aborterRef.current?.abort();
  }

  function applyMatchup(kind: "qknorm" | "mla" | "kvquant" | "sinks") {
    const llama3 = PRESETS.find((p) => p.id === "llama3")!;
    const qwen3 = PRESETS.find((p) => p.id === "qwen3")!;
    const llama1 = PRESETS.find((p) => p.id === "llama1")!;
    const deepseek = PRESETS.find((p) => p.id === "deepseek_v32")!;

    if (kind === "qknorm") {
      updateConfig(llama3.patch, llama3);
      kvDegradation.value = defaultKV();
      surgery.value = emptySurgery();
      raceConfigB.value = applyPresetToConfig(defaultConfig, qwen3);
      raceNameB.value = "Qwen3 8B (QK Norm on)";
      raceKvB.value = defaultKV();
      raceSurgeryB.value = emptySurgery();
      setMaxTokens(420);
    } else if (kind === "mla") {
      updateConfig(llama1.patch, llama1);
      kvDegradation.value = defaultKV();
      surgery.value = emptySurgery();
      raceConfigB.value = applyPresetToConfig(defaultConfig, deepseek);
      raceNameB.value = "DeepSeek V3.2 (MLA)";
      raceKvB.value = defaultKV();
      raceSurgeryB.value = emptySurgery();
      setMaxTokens(320);
    } else if (kind === "kvquant") {
      updateConfig(qwen3.patch, qwen3);
      kvDegradation.value = defaultKV();
      surgery.value = emptySurgery();
      raceConfigB.value = applyPresetToConfig(defaultConfig, qwen3);
      raceNameB.value = "Qwen3 8B (INT2 KV)";
      raceKvB.value = { ...defaultKV(), quant: "int2" };
      raceSurgeryB.value = emptySurgery();
      setMaxTokens(320);
    } else {
      updateConfig(qwen3.patch, qwen3);
      kvDegradation.value = { ...defaultKV(), window: 48, sink_mode: "keep_sinks", sink_tokens: 4 };
      surgery.value = emptySurgery();
      raceConfigB.value = applyPresetToConfig(defaultConfig, qwen3);
      raceNameB.value = "Qwen3 8B (Drop sinks)";
      raceKvB.value = { ...defaultKV(), window: 48, sink_mode: "drop_sinks", sink_tokens: 4 };
      raceSurgeryB.value = emptySurgery();
      setMaxTokens(320);
    }
  }

  function pickPresetB(presetId: string) {
    const p = PRESETS.find((x) => x.id === presetId);
    if (!p) return;
    raceConfigB.value = applyPresetToConfig(defaultConfig, p);
    raceNameB.value = p.name;
    raceKvB.value = defaultKV();
    raceSurgeryB.value = emptySurgery();
  }

  function setKvBQuant(q: KVDegradationConfig["quant"]) {
    raceKvB.value = { ...raceKvB.value, quant: q };
  }

  async function runRace() {
    setSteps([]);
    setFirstDiverge(null);
    setSelectedStep(null);
    setStopReasons(null);
    setStatusErr(false);
    setStatusMsg("Racing both models in lockstep...");
    setRacing(true);
    const aborter = new AbortController();
    aborterRef.current = aborter;

    const acc: RaceStep[] = [];
    let firstDiv: number | null = null;
    try {
      for await (const ev of streamRace(
        {
          config_a: cfgA,
          config_b: cfgB,
          surgery_a: surgery.value,
          surgery_b: raceSurgeryB.value,
          kv_a: kvA,
          kv_b: kvB,
          prompt,
          max_tokens: maxTokens,
          temperature,
        },
        aborter.signal,
      )) {
        if (ev.event === "race_step") {
          const st: RaceStep = {
            step: ev.step,
            a: ev.a,
            b: ev.b,
            diff: ev.diff,
            lm_a: ev.lm_a,
            lm_b: ev.lm_b,
            ent_a: ev.ent_a,
            ent_b: ev.ent_b,
            top_a: ev.top_a,
            top_b: ev.top_b,
          };
          acc.push(st);
          if (st.diff && firstDiv === null) {
            firstDiv = st.step;
            setFirstDiverge(st.step);
            setSelectedStep(st.step);
          }
          if (acc.length % 3 === 0 || st.diff) {
            setSteps([...acc]);
          }
        } else if (ev.event === "race_done") {
          setSteps([...acc]);
          setStopReasons({ a: ev.stop_reason_a, b: ev.stop_reason_b });
          if (ev.first_diverge === null) {
            setStatusMsg(`Identical across ${acc.length} chars in ${ev.elapsed_s.toFixed(2)}s`);
          } else {
            setStatusMsg(`First diverged at char ${ev.first_diverge} (${acc.length} steps in ${ev.elapsed_s.toFixed(2)}s)`);
          }
        } else if (ev.event === "error") {
          setStatusErr(true);
          setStatusMsg(ev.message);
        }
      }
    } catch (e) {
      const err = e as Error;
      if (err.name === "AbortError" || aborter.signal.aborted) {
        setSteps([...acc]);
        setStatusMsg(`Stopped after ${acc.length} steps`);
      } else {
        setStatusErr(true);
        setStatusMsg(err.message);
      }
    } finally {
      setRacing(false);
      aborterRef.current = null;
    }
  }

  const inspected = selectedStep !== null ? steps[selectedStep] : firstDiverge !== null ? steps[firstDiverge] : null;
  const labelA = `${name.value}${kvA.quant !== "none" ? ` (${kvA.quant.toUpperCase()})` : ""}`;
  const labelB = raceNameB.value;

  return (
    <div class="race-stage">
      <div class="race-top-bar">
        <div class="matchup-strip">
          <span class="strip-label">Quick matchups</span>
          <button class="chip-btn" onClick={() => applyMatchup("qknorm")}>QK Norm off vs on</button>
          <button class="chip-btn" onClick={() => applyMatchup("mla")}>MHA vs MLA</button>
          <button class="chip-btn" onClick={() => applyMatchup("kvquant")}>BF16 vs INT2 KV</button>
          <button class="chip-btn" onClick={() => applyMatchup("sinks")}>Keep vs drop sinks</button>
        </div>

        <div class="race-rival-picker">
          <label class="inline-lbl">Model B</label>
          <select
            class="compact-select"
            value={PRESETS.find((p) => p.name === raceNameB.value)?.id ?? ""}
            onChange={(e) => pickPresetB((e.target as HTMLSelectElement).value)}
          >
            <option value="">Custom rival</option>
            {PRESETS.map((p) => (
              <option value={p.id}>{p.name}</option>
            ))}
          </select>
          <select
            class="compact-select"
            value={kvB.quant}
            onChange={(e) => setKvBQuant((e.target as HTMLSelectElement).value as KVDegradationConfig["quant"])}
            title="Model B KV precision"
          >
            <option value="none">KV BF16</option>
            <option value="int8">KV INT8</option>
            <option value="int4">KV INT4</option>
            <option value="int2">KV INT2</option>
          </select>
        </div>
      </div>

      <div class="race-controls">
        <input
          class="prompt-input"
          type="text"
          value={prompt}
          onInput={(e) => setPrompt((e.target as HTMLInputElement).value)}
          placeholder="Enter a story prompt"
        />
        <div class="num-field">
          <span>Max</span>
          <input
            type="number"
            min={20}
            max={2000}
            value={maxTokens}
            onInput={(e) => setMaxTokens(parseInt((e.target as HTMLInputElement).value, 10) || 400)}
          />
        </div>
        <div class="num-field">
          <span>Temp</span>
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onInput={(e) => setTemperature(parseFloat((e.target as HTMLInputElement).value) || 0)}
          />
        </div>
        <button
          class={`btn-primary ${racing ? "btn-stop" : ""}`}
          onClick={racing ? stopRace : runRace}
          disabled={!racing && !canRace}
        >
          {racing ? "Stop" : "Race"}
        </button>
      </div>

      <div class={`race-status ${statusErr ? "is-err" : firstDiverge !== null ? "is-div" : ""}`}>
        <span>{statusMsg}</span>
        {stopReasons && (
          <span class="race-stops">
            A {stopReasons.a === "eos" ? "finished story" : "hit cap"}
            {"   "}
            B {stopReasons.b === "eos" ? "finished story" : "hit cap"}
          </span>
        )}
      </div>

      <div class="race-columns">
        <div class="race-pane">
          <div class="race-pane-head pane-a">
            <strong>Model A</strong>
            <span>{labelA}</span>
          </div>
          <div class="race-output">
            <span class="prompt-echo">{prompt}</span>
            {steps.map((st) => {
              const isFirst = st.step === firstDiverge;
              const isSel = st.step === selectedStep;
              const cls = isFirst ? "ch-first" : st.diff ? "ch-diff" : "";
              return (
                <span
                  key={st.step}
                  class={`race-ch ${cls} ${isSel ? "ch-sel" : ""}`}
                  onClick={() => st.top_a && setSelectedStep(st.step)}
                >
                  {st.a}
                </span>
              );
            })}
          </div>
        </div>

        <div class="race-pane">
          <div class="race-pane-head pane-b">
            <strong>Model B</strong>
            <span>{labelB}</span>
          </div>
          <div class="race-output">
            <span class="prompt-echo">{prompt}</span>
            {steps.map((st) => {
              const isFirst = st.step === firstDiverge;
              const isSel = st.step === selectedStep;
              const cls = isFirst ? "ch-first" : st.diff ? "ch-diff" : "";
              return (
                <span
                  key={st.step}
                  class={`race-ch ${cls} ${isSel ? "ch-sel" : ""}`}
                  onClick={() => st.top_b && setSelectedStep(st.step)}
                >
                  {st.b}
                </span>
              );
            })}
          </div>
        </div>
      </div>

      <div class="race-bottom">
        <div class="diverge-inspector">
          <div class="sub-title">
            <strong>Next token probabilities</strong>
            <span class="pill-count">
              {inspected ? `Step ${inspected.step}` : "Click any highlighted char"}
            </span>
          </div>
          {inspected && inspected.top_a && inspected.top_b ? (
            <div class="prob-compare">
              <div class="prob-col">
                <div class="prob-col-title leg-a">Model A</div>
                {inspected.top_a.map((item, i) => (
                  <div class="prob-row" key={i}>
                    <span class="prob-ch">{escapeVisible(item.ch)}</span>
                    <div class="prob-bar-bg">
                      <div class="prob-bar-fill fill-a" style={{ width: `${(item.p * 100).toFixed(1)}%` }} />
                    </div>
                    <span class="prob-num">{(item.p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <div class="prob-col">
                <div class="prob-col-title leg-b">Model B</div>
                {inspected.top_b.map((item, i) => (
                  <div class="prob-row" key={i}>
                    <span class="prob-ch">{escapeVisible(item.ch)}</span>
                    <div class="prob-bar-bg">
                      <div class="prob-bar-fill fill-b" style={{ width: `${(item.p * 100).toFixed(1)}%` }} />
                    </div>
                    <span class="prob-num">{(item.p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p class="panel-note">
              Run a race to compare the top 5 next character probabilities at the exact point where the two models split.
            </p>
          )}
        </div>

        <div class="race-sparks">
          <OverlaySpark
            title="Max attn logit"
            seriesA={steps.map((s) => s.lm_a)}
            seriesB={steps.map((s) => s.lm_b)}
            labelA="A"
            labelB="B"
          />
          <OverlaySpark
            title="Next token entropy"
            seriesA={steps.map((s) => s.ent_a)}
            seriesB={steps.map((s) => s.ent_b)}
            labelA="A"
            labelB="B"
          />
        </div>
      </div>
    </div>
  );
}
