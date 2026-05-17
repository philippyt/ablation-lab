import { useRef, useState } from "preact/hooks";
import { config, generating, cacheState, resolution, kindBActive, instrumentation } from "../state";
import { streamGenerate } from "../api";
import { HINTS } from "../lib/labels";
import { HelpHint } from "./HelpHint";
import { attnLabels } from "./AttentionGrid";

export function RunPanel() {
  const [prompt, setPrompt] = useState("Once upon a time");
  const [maxTokens, setMaxTokens] = useState(300);
  const [temperature, setTemperature] = useState(0);
  const [output, setOutput] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [statusIsErr, setStatusIsErr] = useState(false);
  const [hoverRun, setHoverRun] = useState(false);
  const aborterRef = useRef<AbortController | null>(null);

  const res = resolution.value;
  const kbActive = kindBActive.value;
  const runnable = res.state !== "red";

  function stop() {
    if (aborterRef.current) {
      aborterRef.current.abort();
    }
  }

  async function run() {
    setOutput(prompt);
    setStatus("streaming...");
    setStatusIsErr(false);
    generating.value = true;
    instrumentation.value = null;
    cacheState.value = { positions: 0, status: "streaming", prompt_tokens: 0 };
    attnLabels.value = Array.from(prompt);
    const t0 = performance.now();
    let tokens = 0;
    let acc = prompt;
    let tStart = t0;
    const aborter = new AbortController();
    aborterRef.current = aborter;
    try {
      for await (const ev of streamGenerate({
        config: config.value,
        prompt,
        max_tokens: maxTokens,
        temperature,
        instrument: true,
      }, aborter.signal)) {
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
          attnLabels.value = [...attnLabels.value, ev.text];
          const prev = cacheState.value;
          cacheState.value = { ...prev, positions: prev.positions + 1 };
          if (tokens % 4 === 0 || tokens < 4) {
            const dt = (performance.now() - tStart) / 1000;
            const tps = dt > 0 ? tokens / dt : 0;
            setStatus(`streaming · ${tokens} tok · ${tps.toFixed(1)} tok/s`);
          }
        } else if (ev.event === "done") {
          const dt = (performance.now() - tStart) / 1000;
          const tps = dt > 0 ? tokens / dt : 0;
          const stop = ev.stop_reason === "eos" ? "stopped at story end" : "hit max_tokens cap";
          setStatus(`${tokens} tok · ${tps.toFixed(1)} tok/s · ${dt.toFixed(2)}s · ${stop} · val_loss=${ev.val_loss.toFixed(3)}`);
          cacheState.value = { ...cacheState.value, status: "done" };
        } else if (ev.event === "instr") {
          const { event: _e, ...payload } = ev as any;
          instrumentation.value = payload;
        } else if (ev.event === "error") {
          setStatus("error: " + ev.message);
          setStatusIsErr(true);
        }
      }
    } catch (e) {
      const err = e as Error;
      if (err.name === "AbortError" || aborter.signal.aborted) {
        const dt = (performance.now() - tStart) / 1000;
        const tps = dt > 0 ? tokens / dt : 0;
        setStatus(`stopped · ${tokens} tok · ${tps.toFixed(1)} tok/s · ${dt.toFixed(2)}s`);
        cacheState.value = { ...cacheState.value, status: "done" };
      } else {
        setStatus("error: " + err.message);
        setStatusIsErr(true);
      }
    } finally {
      generating.value = false;
      aborterRef.current = null;
      setHoverRun(false);
    }
  }

  return (
    <div class="panel">
      <div class="panel-head">
        <button
          class={`run-trigger ${generating.value ? "is-running" : ""} ${generating.value && hoverRun ? "is-stoppable" : ""}`}
          onClick={generating.value ? stop : run}
          onMouseEnter={() => setHoverRun(true)}
          onMouseLeave={() => setHoverRun(false)}
          disabled={generating.value ? false : !runnable}
          title={generating.value ? "Stop streaming and keep the tokens already produced." : (runnable ? "Run inference with the current configuration." : "No trained checkpoint within 2 axes. Cost is still exact, but no real sample can be produced.")}
        >
          {generating.value ? (hoverRun ? "stop" : "running") : "Run"}
        </button>
        <div class="run-state">
          {res.state === "green" && (
            <span class="state-pill ok" title={`The trained checkpoint for this exact architecture is loaded. Hash ${res.resolvedEntry!.hash}, validation loss ${res.resolvedEntry!.val_loss.toFixed(3)}.${kbActive.length > 0 ? " Kind-B interventions are turned on at inference, so " + kbActive.map(s => s.replace("enable_", "")).join(", ") + " is disabled in the forward pass." : ""}`}>
              hash {res.resolvedEntry!.hash} · val={res.resolvedEntry!.val_loss.toFixed(3)}
              {kbActive.length > 0 && <span> · Kind-B: {kbActive.map(s => s.replace("enable_", "")).join(", ")}</span>}
            </span>
          )}
          {res.state === "amber" && (
            <span class="state-pill warn" title={`No trained checkpoint matches this configuration exactly. Running the closest one, which differs on ${res.distance} axis (${res.differingFields.join(", ")}). The output is approximate. The cost panel is still exact.`}>
              approx · {res.distance} axis{res.distance === 1 ? "" : "es"} differ ({res.differingFields.join(", ")})
              {kbActive.length > 0 && <span> · Kind-B: {kbActive.map(s => s.replace("enable_", "")).join(", ")}</span>}
            </span>
          )}
          {res.state === "red" && (
            <span class="state-pill err" title="No trained checkpoint is within 2 axes of this configuration, so we won't fake a sample. The cost panel is still exact. Pick a configuration closer to a trained one to see real output.">
              no checkpoint within 2 axes
            </span>
          )}
        </div>
      </div>
      <div class="row" style="margin-top:0.3rem;">
        <label>Prompt<HelpHint text={HINTS.prompt} /></label>
        <textarea value={prompt} onInput={(e) => setPrompt((e.target as HTMLTextAreaElement).value)} />
        <label>Max tokens<HelpHint text={HINTS.max_tokens} /></label>
        <input type="number" min={1} max={5000} value={maxTokens}
               onInput={(e) => setMaxTokens(parseInt((e.target as HTMLInputElement).value, 10))} />
        <label>Temperature<HelpHint text={HINTS.temperature} /></label>
        <input type="number" min={0} max={2} step={0.1} value={temperature}
               onInput={(e) => setTemperature(parseFloat((e.target as HTMLInputElement).value))} />
      </div>
      <div class={`status ${statusIsErr ? "err" : ""}`}>{status}</div>
      <div class="output">{output}</div>
    </div>
  );
}
