import { useState } from "preact/hooks";
import { config, generating, matchingEntry, cacheState } from "../state";
import { streamGenerate } from "../api";
import { HINTS } from "../lib/labels";
import { HelpHint } from "./HelpHint";
import { attnLabels } from "./AttentionGrid";

export function RunPanel() {
  const [prompt, setPrompt] = useState("Once upon a time");
  const [maxTokens, setMaxTokens] = useState(150);
  const [temperature, setTemperature] = useState(0.8);
  const [output, setOutput] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [statusIsErr, setStatusIsErr] = useState(false);

  const entry = matchingEntry.value;
  const runnable = entry !== null;

  async function run() {
    setOutput(prompt);
    setStatus("streaming...");
    setStatusIsErr(false);
    generating.value = true;
    cacheState.value = { positions: 0, status: "streaming", prompt_tokens: 0 };
    attnLabels.value = Array.from(prompt);
    const t0 = performance.now();
    let tokens = 0;
    let acc = prompt;
    let tStart = t0;
    try {
      for await (const ev of streamGenerate({
        config: config.value,
        prompt,
        max_tokens: maxTokens,
        temperature,
      })) {
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
          cacheState.value = { ...prev, positions: Math.min(prev.positions + 1, config.value.ctx) };
          if (tokens % 4 === 0 || tokens < 4) {
            const dt = (performance.now() - tStart) / 1000;
            const tps = dt > 0 ? tokens / dt : 0;
            setStatus(`streaming · ${tokens} tok · ${tps.toFixed(1)} tok/s`);
          }
        } else if (ev.event === "done") {
          const dt = (performance.now() - tStart) / 1000;
          const tps = dt > 0 ? tokens / dt : 0;
          setStatus(`${tokens} tok · ${tps.toFixed(1)} tok/s · ${dt.toFixed(2)}s · val_loss=${ev.val_loss.toFixed(3)}`);
          cacheState.value = { ...cacheState.value, status: "done" };
        } else if (ev.event === "error") {
          setStatus("error: " + ev.message);
          setStatusIsErr(true);
        }
      }
    } catch (e) {
      setStatus("error: " + (e as Error).message);
      setStatusIsErr(true);
    } finally {
      generating.value = false;
    }
  }

  return (
    <div class="panel">
      <h2>Run</h2>
      {runnable ? (
        <div class="banner ok">checkpoint available · hash {entry!.hash} · val_loss={entry!.val_loss.toFixed(3)}</div>
      ) : (
        <div class="banner">no checkpoint for this architecture yet · pick a different combination or train it offline</div>
      )}
      <div class="row" style="margin-top:0.75rem;">
        <label>Prompt<HelpHint text={HINTS.prompt} /></label>
        <textarea value={prompt} onInput={(e) => setPrompt((e.target as HTMLTextAreaElement).value)} />
        <label>Max tokens<HelpHint text={HINTS.max_tokens} /></label>
        <input type="number" min={1} max={500} value={maxTokens}
               onInput={(e) => setMaxTokens(parseInt((e.target as HTMLInputElement).value, 10))} />
        <label>Temperature<HelpHint text={HINTS.temperature} /></label>
        <input type="number" min={0} max={2} step={0.1} value={temperature}
               onInput={(e) => setTemperature(parseFloat((e.target as HTMLInputElement).value))} />
      </div>
      <button onClick={run} disabled={!runnable || generating.value} style="margin-top:0.75rem;">
        {generating.value ? "running..." : "Run"}
      </button>
      <div class={`status ${statusIsErr ? "err" : ""}`}>{status}</div>
      <div class="output">{output}</div>
    </div>
  );
}
