from __future__ import annotations
from pathlib import Path
import torch
from ablation.tokenizer import CharTokenizer

def load_text(dataset_path: str) -> str:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path.read_text(encoding="utf-8")

def make_split(text: str, tokenizer: CharTokenizer, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(len(ids) * val_frac)
    n_train = len(ids) - n_val
    return ids[:n_train], ids[n_train:]

def get_batch(data: torch.Tensor, *, batch_size: int, ctx: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(low=0, high=len(data) - ctx - 1, size=(batch_size,))
    x = torch.stack([data[i : i + ctx] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + ctx] for i in ix])
    return x.to(device), y.to(device)