from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass
class CharTokenizer:
    chars: str
    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def _stoi(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.chars)}

    @property
    def _itos(self) -> dict[int, str]:
        return {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def build_from_text(cls, text: str) -> "CharTokenizer":
        return cls(chars="".join(sorted(set(text))))

    def encode(self, s: str) -> list[int]:
        stoi = self._stoi
        return [stoi[c] for c in s if c in stoi]

    def decode(self, ids: Iterable[int]) -> str:
        itos = self._itos
        return "".join(itos[int(i)] for i in ids if int(i) in itos)

    def state(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        return cls(chars=state["chars"])