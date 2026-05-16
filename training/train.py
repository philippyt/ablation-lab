from __future__ import annotations
import argparse
import json
import math
import time
import torch

from ablation.checkpoints import (
    checkpoint_path,
    find_entry_for_config,
    save_checkpoint,
)
from ablation.config import (
    ArchConfig,
    TrainConfig,
    arch_config_from_dict,
    arch_hash,
    train_config_from_dict,
)
from ablation.model import TinyTransformer
from ablation.tokenizer import CharTokenizer
from training.data import get_batch, load_text, make_split
from training.seed import seed_everything

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def cosine_lr(step: int, *, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * 0.1
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

@torch.no_grad()
def estimate_loss(model: TinyTransformer, splits: dict[str, torch.Tensor], cfg: TrainConfig, ctx: int, device) -> dict[str, float]:
    model.eval()
    out = {}
    for split_name, data in splits.items():
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, batch_size=cfg.batch_size, ctx=ctx, device=device)
            _, loss = model(x, targets=y)
            losses.append(loss.item())
        out[split_name] = sum(losses) / len(losses)
    model.train()
    return out

def load_config_file(path: str) -> tuple[dict, dict, str | None]:
    with open(path) as f:
        raw = json.load(f)
    return dict(raw.get("arch", {})), dict(raw.get("train", {})), raw.get("name")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint already exists")
    args = parser.parse_args()

    arch_raw, train_raw, name = load_config_file(args.config)
    train_cfg = train_config_from_dict(train_raw)
    seed_everything(train_cfg.seed)
    device = resolve_device(train_cfg.device)
    print(f"[train] device={device}  seed={train_cfg.seed}  config={args.config}")

    text = load_text(train_cfg.dataset_path)
    tokenizer = CharTokenizer.build_from_text(text)
    print(f"[train] corpus={len(text):,} chars  vocab_size={tokenizer.vocab_size}")

    arch_raw["vocab_size"] = tokenizer.vocab_size
    arch_cfg = arch_config_from_dict(arch_raw)
    h = arch_hash(arch_cfg)
    print(f"[train] arch_hash={h}  attention={arch_cfg.attention_kind}  norm={arch_cfg.norm}  ffn={arch_cfg.ffn}")

    if not args.force and find_entry_for_config(arch_cfg) is not None and checkpoint_path(h).exists():
        print(f"[train] checkpoint already exists at {checkpoint_path(h)}, skipping (use --force to retrain)")
        return

    train_data, val_data = make_split(text, tokenizer)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    print(f"[train] train_tokens={len(train_data):,}  val_tokens={len(val_data):,}")

    model = TinyTransformer(arch_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model_params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    log: list[dict] = []
    best_val = float("inf")
    t0 = time.time()
    model.train()

    for step in range(train_cfg.steps):
        lr = cosine_lr(step, base_lr=train_cfg.lr, warmup=train_cfg.warmup_steps, total=train_cfg.steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, batch_size=train_cfg.batch_size, ctx=arch_cfg.ctx, device=device)
        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if (step + 1) % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, train_cfg, arch_cfg.ctx, device)
            elapsed = time.time() - t0
            print(
                f"[train] step={step+1:5d}/{train_cfg.steps}  lr={lr:.5f}  "
                f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                f"elapsed={elapsed:.1f}s"
            )
            log.append({"step": step + 1, "lr": lr, **losses, "elapsed": elapsed})
            best_val = min(best_val, losses["val"])

    final_losses = estimate_loss(model, {"train": train_data, "val": val_data}, train_cfg, arch_cfg.ctx, device)
    final_val = float(final_losses["val"])
    print(f"[train] complete, final_val_loss={final_val:.4f}, total_time={time.time()-t0:.1f}s")

    save_checkpoint(
        arch_config=arch_cfg,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        tokenizer_state=tokenizer.state(),
        final_val_loss=final_val,
        training_log=log,
        name=name,
    )
    print(f"[train] saved checkpoints/{h}.pt  (manifest updated)")

if __name__ == "__main__":
    main()