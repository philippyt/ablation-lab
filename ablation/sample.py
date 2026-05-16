from __future__ import annotations
import argparse
import torch
from .checkpoints import load_checkpoint
from .config import arch_config_from_dict
from .model import TinyTransformer
from .tokenizer import CharTokenizer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", required=True, help="Architecture hash from manifest.json")
    parser.add_argument("--prompt", default="\n", help="Prompt to condition on")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    ckpt = load_checkpoint(args.hash, map_location=args.device)
    arch_cfg = arch_config_from_dict(ckpt["arch_config"])
    tokenizer = CharTokenizer.from_state(ckpt["tokenizer_state"])
    model = TinyTransformer(arch_cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    print(args.prompt, end="", flush=True)
    for tok in model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        print(tokenizer.decode([tok]), end="", flush=True)
    print()
    print(f"\n[val_loss={ckpt.get('final_val_loss', float('nan')):.4f}]")

if __name__ == "__main__":
    main()