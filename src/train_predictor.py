"""
Train مدل پیش‌بینی "کارت بعدی حریف" روی توالی‌های آماده‌شده در prepare_sequences.py.

روش: از هر توالی مسابقه [c1, c2, ..., cn]، نمونه‌های (تاریخچه اخیر -> کارت بعدی) می‌سازیم.
مثلاً با CONTEXT_WINDOW=5: تاریخچه [c1,c2,c3] -> هدف c4, تاریخچه [c1,c2,c3,c4] -> هدف c5, ...

پیش‌نیاز:
    pip install torch
    python src/prepare_sequences.py   (باید data/sequences/sequences.json ساخته باشد)

اجرا:
    python src/train_predictor.py --epochs 30
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import PROJECT_ROOT
from sequence_model import CardVocab, NextCardLSTM

SEQUENCES_PATH = PROJECT_ROOT / "data" / "sequences" / "sequences.json"
MODEL_DIR = PROJECT_ROOT / "models" / "next_card_predictor"
CONTEXT_WINDOW = 5


def build_training_pairs(sequences: list[list[str]], vocab: CardVocab) -> tuple[torch.Tensor, torch.Tensor]:
    contexts = []
    targets = []

    for sequence in sequences:
        encoded = [vocab.encode(card) for card in sequence]
        for i in range(1, len(encoded)):
            start = max(0, i - CONTEXT_WINDOW)
            context = encoded[start:i]
            padding_needed = CONTEXT_WINDOW - len(context)
            context = [0] * padding_needed + context  # پد کردن با PAD (id=0) از چپ
            contexts.append(context)
            targets.append(encoded[i])

    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train(epochs: int, batch_size: int, lr: float) -> None:
    if not SEQUENCES_PATH.exists():
        raise FileNotFoundError(f"{SEQUENCES_PATH} پیدا نشد — اول src/prepare_sequences.py را اجرا کنید.")

    sequences = json.loads(SEQUENCES_PATH.read_text(encoding="utf-8"))
    all_cards = [card for seq in sequences for card in seq]
    vocab = CardVocab(all_cards)

    X, y = build_training_pairs(sequences, vocab)
    print(f"{len(X)} نمونه train از {len(sequences)} مسابقه ساخته شد. اندازه واژه‌نامه: {len(vocab)}")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NextCardLSTM(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()

        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(f"  epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  train_acc={accuracy:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    vocab.save(MODEL_DIR / "vocab.json")
    print(f"\nمدل ذخیره شد در: {MODEL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train مدل پیش‌بینی کارت بعدی حریف")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
