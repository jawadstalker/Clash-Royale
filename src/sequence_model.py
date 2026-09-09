"""
تعریف مدل و واژه‌نامه (vocab) مشترک بین train_predictor.py و predict_next_card.py.

معماری: Embedding -> LSTM -> Linear
ورودی: توالی id کارت‌های قبلاً دیده‌شده از یک حریف در همان مسابقه
خروجی: توزیع احتمال روی همه کارت‌ها برای "کارت بعدی"
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

PAD_TOKEN = "<PAD>"


class CardVocab:
    """نگاشت بین اسم کارت و عدد (id) برای مدل."""

    def __init__(self, card_names: list[str]):
        self.itos = [PAD_TOKEN] + sorted(set(card_names))
        self.stoi = {name: i for i, name in enumerate(self.itos)}

    def encode(self, card_name: str) -> int:
        return self.stoi.get(card_name, self.stoi[PAD_TOKEN])

    def decode(self, idx: int) -> str:
        return self.itos[idx]

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CardVocab":
        itos = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls.__new__(cls)
        vocab.itos = itos
        vocab.stoi = {name: i for i, name in enumerate(itos)}
        return vocab


class NextCardLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) از idهای کارت
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]  # فقط خروجی آخرین قدم زمانی برای پیش‌بینی "بعدی" لازم است
        logits = self.fc(last_hidden)
        return logits
