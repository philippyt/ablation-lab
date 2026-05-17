from __future__ import annotations
import argparse
import torch
from ablation.checkpoints import load_checkpoint
from ablation.config import arch_config_from_dict
from ablation.model import TinyTransformer
from ablation.tokenizer import CharTokenizer

def sample(model: TinyTransformer, tokenizer: CharTokenizer, prompt: str, n: int, runtime: dict | None, device: torch.device) -> str:
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = list(ids)
    for tok in model.generate(idx, max_new_tokens=n, temperature=0.8, top_k=40, runtime=runtime):
        out_ids.append(int(tok))
    return tokenizer.decode(out_ids)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--n", type=int, default=180)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.hash, map_location=str(device))
    cfg = arch_config_from_dict(ckpt["arch_config"])
    tokenizer = CharTokenizer.from_state(ckpt["tokenizer_state"])
    model = TinyTransformer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"hash={args.hash} attn={cfg.attention_kind} norm={cfg.norm} ffn={cfg.ffn} val_loss={ckpt['final_val_loss']:.4f} device={device}")
    print()
    print("=== baseline (all sublayers ON) ===")
    torch.manual_seed(42)
    print(sample(model, tokenizer, args.prompt, args.n, runtime=None, device=device))
    print()
    print("=== Kind-B: enable_attention=False (live ablation, NO retraining) ===")
    torch.manual_seed(42)
    print(sample(model, tokenizer, args.prompt, args.n, runtime={"enable_attention": False}, device=device))
    print()
    print("=== Kind-B: enable_ffn=False (live ablation, NO retraining) ===")
    torch.manual_seed(42)
    print(sample(model, tokenizer, args.prompt, args.n, runtime={"enable_ffn": False}, device=device))
    print()
    print("=== Kind-B: enable_norms=False (live ablation, NO retraining) ===")
    torch.manual_seed(42)
    print(sample(model, tokenizer, args.prompt, args.n, runtime={"enable_norms": False}, device=device))

if __name__ == "__main__":
    main()
