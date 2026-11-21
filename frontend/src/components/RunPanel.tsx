import { useRef, useState } from "preact/hooks";
import {
  config,
  generating,
  cacheState,
  resolution,
  kindBActive,
  instrumentation,
  telemetry,
  surgery,
  kvDegradation,
  lastRunText,
  logitLens,
  inspectTokenLogitLens,
  updateConfig,
  resetSurgery,
  resetKV,
  updateKV,
} from "../state";
import { PRESETS } from "../lib/presets";
import { streamGenerate } from "../api";
import { HINTS } from "../lib/labels";
import { HelpHint } from "./HelpHint";
import { attnLabels } from "./AttentionGrid";
import { LiveTelemetry } from "./LiveTelemetry";
import type { TelemetryPoint } from "../types";

function surpriseBg(s: number, maxS: number): string {
  if (maxS <= 0) return "transparent";
  const t = Math.min(1, s / Math.max(maxS, 2.5));
  if (t < 0.25) return "transparent";
  return `rgba(217, 119, 6, ${(0.12 + t * 0.42).toFixed(2)})`;
}

export function RunPanel() {
  const [prompt, setPrompt] = useState("Once upon a time");
  const [maxTokens, setMaxTokens] = useState(420);
  const [temperature, setTemperature] = useState(0);
  const [output, setOutput] = useState("");
  const [showSurprise, setShowSurprise] = useState(false);
  const [status, setStatus] = useState<string>("Ready to generate");
  const [statusIsErr, setStatusIsErr] = useState(false);
  const aborterRef = useRef<AbortController | null>(null);

  const res = resolution.value;
  const kbActive = kindBActive.value;
  const runnable = res.state !== "red";
  const pts = telemetry.value;
  const activeLensStep = logitLens.value?.stepIndex ?? null;

  function stop() {
    aborterRef.current?.abort();
  }

  function applyExperiment(kind: "qknorm_loop" | "no_norms" | "drop_sinks" | "int2_kv") {
    const llama3 = PRESETS.find((p) => p.id === "llama3")!;
    const qwen3 = PRESETS.find((p) => p.id === "qwen3")!;
    resetSurgery();
    resetKV();
    if (kind === "qknorm_loop") {
      updateConfig({ ...llama3.patch, enable_attention: true, enable_ffn: true, enable_norms: true }, llama3);
      setMaxTokens(420);
      setTemperature(0);
    } else if (kind === "no_norms") {
      updateConfig({ ...qwen3.patch, enable_attention: true, enable_ffn: true, enable_norms: false }, qwen3);
      setMaxTokens(200);
      setTemperature(0);
    } else if (kind === "drop_sinks") {
      updateConfig({ ...qwen3.patch, enable_attention: true, enable_ffn: true, enable_norms: true }, qwen3);
      updateKV({ window: 48, sink_mode: "drop_sinks", sink_tokens: 4 });
      setMaxTokens(260);
      setTemperature(0);
    } else {
      updateConfig({ ...qwen3.patch, enable_attention: true, enable_ffn: true, enable_norms: true }, qwen3);
      updateKV({ quant: "int2" });
      setMaxTokens(260);
      setTemperature(0);
    }
  }

  async function run() {
    setOutput(prompt);
    lastRunText.value = prompt;
    setStatus("Streaming");
    setStatusIsErr(false);
    generating.value = true;
    instrumentation.value = null;
    logitLens.value = null;
    telemetry.value = [];
    cacheState.value = { positions: 0, status: "streaming", prompt_tokens: 0 };
    attnLabels.value = Array.from(prompt);
    const t0 = performance.now();
    let tokens = 0;
    let acc = prompt;
    let tStart = t0;
    const teleAcc: TelemetryPoint[] = [];
    const aborter = new AbortController();
    aborterRef.current = aborter;
    try {
      for await (const ev of streamGenerate(
        {
          config: config.value,
          prompt,
          max_tokens: maxTokens,
          temperature,
          instrument: true,
          surgery: surgery.value,
          kv: kvDegradation.value,
        },
        aborter.signal,
      )) {
        if (ev.event === "start") {
          cacheState.value = {
            positions: ev.prompt_tokens,
            status: "streaming",
            prompt_tokens: ev.prompt_tokens,
          };
          tStart = performance.now();
        } else if (ev.event === "token") {
          acc += ev.text;
          tokens += 1;
          setOutput(acc);
          lastRunText.value = acc;
          attnLabels.value = [...attnLabels.value, ev.text];
          teleAcc.push({
            ch: ev.text,
            lm: ev.lm ?? 0,
            ent: ev.ent ?? 0,
            rn: ev.rn ?? 0,
            s: ev.s ?? 0,
          });
          if (tokens % 3 === 0 || tokens < 6) {
            telemetry.value = [...teleAcc];
            const dt = (performance.now() - tStart) / 1000;
            const tps = dt > 0 ? tokens / dt : 0;
            setStatus(`Streaming ${tokens} chars at ${tps.toFixed(0)} tok/s`);
          }
          const prev = cacheState.value;
          cacheState.value = { ...prev, positions: prev.positions + 1 };
        } else if (ev.event === "done") {
          telemetry.value = [...teleAcc];
          const dt = (performance.now() - tStart) / 1000;
          const tps = dt > 0 ? tokens / dt : 0;
          const stop =
            ev.stop_reason === "eos"
              ? "Reached story end"
              : ev.stop_reason === "ctx_limit"
                ? "Reached context limit"
                : "Hit token cap";
          setStatus(`${tokens} chars in ${dt.toFixed(2)}s (${tps.toFixed(0)} tok/s) ${stop}`);
          cacheState.value = { ...cacheState.value, status: "done" };
        } else if (ev.event === "instr") {
          const { event: _e, ...payload } = ev as any;
          instrumentation.value = payload;
        } else if (ev.event === "error") {
          setStatus(ev.message);
          setStatusIsErr(true);
        }
      }
    } catch (e) {
      const err = e as Error;
      if (err.name === "AbortError" || aborter.signal.aborted) {
        telemetry.value = [...teleAcc];
        const dt = (performance.now() - tStart) / 1000;
        setStatus(`Stopped at ${tokens} chars (${dt.toFixed(2)}s)`);
        cacheState.value = { ...cacheState.value, status: "done" };
      } else {
        setStatus(err.message);
        setStatusIsErr(true);
      }
    } finally {
      generating.value = false;
      aborterRef.current = null;
    }
  }

  const maxSurprise = pts.length > 0 ? Math.max(...pts.map((p) => p.s)) : 0;

  return (
    <div class="card run-card">
      <div class="run-top">
        <div class="run-status-pills">
          {res.state === "green" && (
            <span class="pill pill-ok">
              Exact checkpoint (loss {res.resolvedEntry!.val_loss.toFixed(3)})
            </span>
          )}
          {res.state === "amber" && (
            <span class="pill pill-warn">
              Closest checkpoint ({res.distance} axis diff)
            </span>
          )}
          {res.state === "red" && (
            <span class="pill pill-err">No checkpoint within 2 axes</span>
          )}
          {kbActive.map((tag) => (
            <span class="pill pill-live" key={tag}>{tag}</span>
          ))}
        </div>

        <label class="toggle-inline" title="Highlight generated characters by token surprise">
          <input
            type="checkbox"
            checked={showSurprise}
            onChange={(e) => setShowSurprise((e.target as HTMLInputElement).checked)}
          />
          <span>Surprise heat</span>
        </label>
      </div>

      <div class="exp-strip">
        <span class="strip-label">Try this</span>
        <button class="chip-btn" onClick={() => applyExperiment("qknorm_loop")}>QK Norm loop</button>
        <button class="chip-btn" onClick={() => applyExperiment("no_norms")}>No norms blowup</button>
        <button class="chip-btn" onClick={() => applyExperiment("drop_sinks")}>Drop 4 sinks</button>
        <button class="chip-btn" onClick={() => applyExperiment("int2_kv")}>INT2 KV stress</button>
      </div>

      <div class="run-controls">
        <input
          class="prompt-input"
          type="text"
          value={prompt}
          onInput={(e) => setPrompt((e.target as HTMLInputElement).value)}
          placeholder="Story prompt"
        />
        <div class="num-field" title={HINTS.max_tokens}>
          <span>Max</span>
          <input
            type="number"
            min={1}
            max={5000}
            value={maxTokens}
            onInput={(e) => setMaxTokens(parseInt((e.target as HTMLInputElement).value, 10) || 400)}
          />
        </div>
        <div class="num-field" title={HINTS.temperature}>
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
          class={`btn-primary ${generating.value ? "btn-stop" : ""}`}
          onClick={generating.value ? stop : run}
          disabled={!generating.value && !runnable}
        >
          {generating.value ? "Stop" : "Run"}
        </button>
      </div>

      <div class={`run-meta ${statusIsErr ? "is-err" : ""}`}>
        <span>{status}</span>
        <span class="meta-tip">
          {pts.length > 0 ? "Click any generated char for Logit Lens" : ""}
          <HelpHint text={HINTS.prompt} />
        </span>
      </div>

      <div class="output-box">
        {!output && pts.length === 0 ? (
          <span class="output-placeholder">Press Run to stream text from the trained TinyStories checkpoint.</span>
        ) : pts.length > 0 ? (
          <>
            <span class="prompt-echo">{prompt}</span>
            {pts.map((p, i) => {
              const isSel = activeLensStep === i;
              return (
                <span
                  key={i}
                  class={`gen-ch ${isSel ? "is-lens-sel" : ""}`}
                  style={showSurprise ? { background: surpriseBg(p.s, maxSurprise) } : undefined}
                  onClick={() => {
                    const prefix = prompt + pts.slice(0, i).map((x) => x.ch).join("");
                    inspectTokenLogitLens(prefix, p.ch, i);
                  }}
                  title={`Click for Logit Lens (surprise ${p.s.toFixed(2)} nats, entropy ${p.ent.toFixed(2)} nats)`}
                >
                  {p.ch}
                </span>
              );
            })}
          </>
        ) : (
          <>
            <span class="prompt-echo">{output.slice(0, prompt.length)}</span>
            <span>{output.slice(prompt.length)}</span>
          </>
        )}
      </div>

      <LiveTelemetry />
    </div>
  );
}
