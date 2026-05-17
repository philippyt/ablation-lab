from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch
from ablation.checkpoints import REPO_ROOT, read_manifest, load_checkpoint, update_entry_field
from ablation.config import arch_config_from_dict
from ablation.model import TinyTransformer
from ablation.tokenizer import CharTokenizer

PROMPT = "Once upon a time"
N_CHARS = 100
DIVERGENCE_THRESHOLD = 0.10
TWIN_AXES = ("norm", "attention_type", "n_kv_heads", "ffn", "n_experts", "top_k", "activation", "pos_embed")

def _reconstruct_source_config(arch: dict) -> str | None:
    norm_short = "ln" if arch.get("norm") == "layernorm" else "rms"
    if arch.get("attention_type") == "mla":
        attn = "mla"
    elif arch.get("n_kv_heads") == 8:
        attn = "mha"
    elif arch.get("n_kv_heads") == 1:
        attn = "mqa"
    else:
        attn = "gqa"
    if arch.get("ffn") == "moe":
        ffn = "moe-small" if arch.get("n_experts") == 4 and arch.get("top_k") == 1 else "moe"
    else:
        ffn = "dense"
    act_short = {"silu": "silu", "gelu": "gelu", "relu2": "rl2"}.get(arch.get("activation"), arch.get("activation"))
    qkn = "qkn1" if arch.get("qk_norm") else "qkn0"
    pos_short = "abs" if arch.get("pos_embed") == "absolute" else "rope"
    candidate = REPO_ROOT / "training" / "configs" / f"{norm_short}_{attn}_{ffn}_{act_short}_{qkn}_{pos_short}.json"
    return str(candidate.relative_to(REPO_ROOT)) if candidate.exists() else None

def _greedy_sample(ckpt: dict, n: int, device: torch.device) -> str:
    cfg = arch_config_from_dict(ckpt["arch_config"])
    tokenizer = CharTokenizer.from_state(ckpt["tokenizer_state"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ids = tokenizer.encode(PROMPT) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = list(ids)
    torch.manual_seed(42)
    for tok in model.generate(idx, max_new_tokens=n, temperature=0.0, top_k=None):
        out_ids.append(int(tok))
        if len(out_ids) >= len(ids) + n:
            break
    return tokenizer.decode(out_ids)[len(PROMPT):]

def _twin_key(cfg_dict: dict) -> tuple:
    return tuple(cfg_dict.get(k) for k in TWIN_AXES)

def _hamming_ratio(a: str, b: str, n: int) -> float:
    la = (a + " " * n)[:n]
    lb = (b + " " * n)[:n]
    diffs = sum(1 for x, y in zip(la, lb) if x != y)
    return diffs / n

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "checkpoints" / "divergence_report.json")
    parser.add_argument("--emit-fails", action="store_true", help="On stdout, print config file paths of failed qkn=on entries (one per line), nothing else.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_manifest()
    entries = manifest.get("entries", {})
    by_twin: dict[tuple, dict[bool, dict]] = {}
    for h, e in entries.items():
        cfg = e["arch_config"]
        key = _twin_key(cfg)
        by_twin.setdefault(key, {})[bool(cfg.get("qk_norm"))] = e
    rows = []
    failed_source_configs: list[str] = []
    for key, pair in sorted(by_twin.items(), key=lambda kv: str(kv[0])):
        if True not in pair or False not in pair:
            continue
        off = pair[False]
        on = pair[True]
        ckpt_off = load_checkpoint(off["hash"], map_location=str(device))
        ckpt_on = load_checkpoint(on["hash"], map_location=str(device))
        s_off = _greedy_sample(ckpt_off, N_CHARS, device)
        s_on = _greedy_sample(ckpt_on, N_CHARS, device)
        ratio = _hamming_ratio(s_off, s_on, N_CHARS)
        collapsed = ratio < DIVERGENCE_THRESHOLD
        verdict = "fail" if collapsed else "pass"
        update_entry_field(on["hash"], "qk_norm_divergence", verdict)
        update_entry_field(off["hash"], "qk_norm_divergence", verdict)
        row = {
            "signature": dict(zip(TWIN_AXES, key)),
            "qkn_off_hash": off["hash"],
            "qkn_on_hash": on["hash"],
            "qkn_off_sample": s_off[:N_CHARS],
            "qkn_on_sample": s_on[:N_CHARS],
            "hamming_ratio_first_100": round(ratio, 4),
            "lesson_collapsed": collapsed,
            "verdict": verdict,
        }
        rows.append(row)
        if collapsed:
            src = on.get("source_config") or _reconstruct_source_config(on["arch_config"])
            if src:
                failed_source_configs.append(src)
            elif not args.emit_fails:
                print(f"  [WARN] collapsed qkn=on hash {on['hash']} has no recoverable source_config — cannot auto-retry", file=sys.stderr)
        if not args.emit_fails:
            mark = "FAIL" if collapsed else "pass"
            sig = "/".join(str(v) for v in key)
            print(f"  [{mark:>4s}]  hamming={ratio:.3f}  {sig}", file=sys.stderr)

    args.report.write_text(json.dumps({
        "threshold": DIVERGENCE_THRESHOLD,
        "pairs_checked": len(rows),
        "collapsed_count": sum(1 for r in rows if r["lesson_collapsed"]),
        "rows": rows,
    }, indent=2))

    if args.emit_fails:
        for p in failed_source_configs:
            print(p)
        return 1 if failed_source_configs else 0

    flagged = [r for r in rows if r["lesson_collapsed"]]
    print(f"\npairs checked: {len(rows)}  collapsed: {len(flagged)}", file=sys.stderr)
    print(f"report: {args.report}", file=sys.stderr)
    if flagged:
        print("\nFLAGGED (lesson collapsed):", file=sys.stderr)
        for r in flagged:
            sig = "/".join(str(v) for v in r["signature"].values())
            print(f"  {sig}  ratio={r['hamming_ratio_first_100']}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
