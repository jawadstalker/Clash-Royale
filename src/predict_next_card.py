"""
پیش‌بینی کارت بعدی حریف بر اساس تاریخچه اخیر بازی‌هایش در همین مسابقه.

اجرا (تست دستی با چند کارت آخر):
    python src/predict_next_card.py --history giant fireball knight --top-k 3

استفاده برنامه‌نویسی (مثلاً داخل card_detector.py برای پیش‌بینی زنده حین بازی):
    from predict_next_card import NextCardPredictor
    predictor = NextCardPredictor()
    predictor.predict(["giant", "fireball", "knight"], top_k=3)
"""

import argparse

import torch
import torch.nn.functional as F

from train_predictor import CONTEXT_WINDOW, MODEL_DIR
from sequence_model import CardVocab, NextCardLSTM


class NextCardPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        vocab_path = model_dir / "vocab.json"
        model_path = model_dir / "model.pt"
        if not vocab_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"مدل پیش‌بینی در {model_dir} پیدا نشد — اول src/train_predictor.py را اجرا کنید."
            )

        self.vocab = CardVocab.load(vocab_path)
        self.model = NextCardLSTM(vocab_size=len(self.vocab))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, recent_history: list[str], top_k: int = 3) -> list[tuple[str, float]]:
        """با گرفتن آخرین کارت‌های دیده‌شده از حریف، top_k کارت محتمل بعدی را برمی‌گرداند."""
        encoded = [self.vocab.encode(card) for card in recent_history]
        context = encoded[-CONTEXT_WINDOW:]
        padding_needed = CONTEXT_WINDOW - len(context)
        context = [0] * padding_needed + context

        x = torch.tensor([context], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)

        top_probs, top_ids = torch.topk(probs, k=min(top_k, len(self.vocab)))
        return [(self.vocab.decode(int(idx)), float(p)) for p, idx in zip(top_probs, top_ids)]


def main() -> None:
    parser = argparse.ArgumentParser(description="پیش‌بینی کارت بعدی حریف")
    parser.add_argument("--history", nargs="+", required=True, help="آخرین کارت‌های دیده‌شده، به ترتیب زمان")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    predictor = NextCardPredictor()
    predictions = predictor.predict(args.history, top_k=args.top_k)

    print(f"\nبا توجه به تاریخچه: {args.history}")
    print("محتمل‌ترین کارت‌های بعدی:")
    for card, prob in predictions:
        print(f"  {card:20s}  {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
