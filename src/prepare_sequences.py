"""
تبدیل لاگ‌های خام تشخیص (data/match_logs/*.jsonl) به توالی تمیز از "رویدادهای بازی کارت".

مشکل داده خام: چون هر کارت وقتی روی زمین است چند ثانیه پشت‌سرهم دیده و لاگ می‌شود
(مثلاً هر ۳۰۰ میلی‌ثانیه یک sighting جدید از همان "giant" ثبت می‌شود)، نمی‌شود مستقیم
این را به مدل پیش‌بینی داد — چون همان یک بازی‌کردنِ کارت، ۱۰ بار در توالی تکرار می‌شود.

راه‌حل: sightingهای پشت‌سرهم از یک کارت را که فاصله زمانی‌شان کمتر از COOLDOWN_SECONDS
است، به یک "رویداد بازی" واحد تبدیل می‌کنیم (اولین لحظه دیده‌شدن = لحظه بازی کارت).

اجرا:
    python src/prepare_sequences.py
"""

import json
from pathlib import Path

from config import PROJECT_ROOT

MATCH_LOGS_DIR = PROJECT_ROOT / "data" / "match_logs"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

# اگر یک کارت دوباره ظرف این‌قدر ثانیه دیده شود، همان یک رویداد در نظر گرفته می‌شود
# (چون واحد چند ثانیه روی زمین می‌ماند و هر فریم دوباره تشخیص داده می‌شود)
COOLDOWN_SECONDS = 4.0
MIN_CONFIDENCE = 0.5


def load_raw_sightings(log_path: Path) -> list[dict]:
    sightings = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sightings.append(json.loads(line))
    return sightings


def dedupe_to_play_events(sightings: list[dict]) -> list[dict]:
    """sightingهای پشت‌سرهم یک کارت را به یک رویداد بازی تبدیل می‌کند."""
    sightings = [s for s in sightings if s["conf"] >= MIN_CONFIDENCE]
    sightings.sort(key=lambda s: s["t"])

    last_seen_time: dict[str, float] = {}
    events = []
    for s in sightings:
        card = s["card"]
        t = s["t"]
        if card in last_seen_time and (t - last_seen_time[card]) < COOLDOWN_SECONDS:
            last_seen_time[card] = t
            continue  # ادامه همان رویداد قبلی است، نه بازی جدید
        last_seen_time[card] = t
        events.append({"t": t, "card": card})

    return events


def build_all_sequences() -> list[list[str]]:
    sequences = []
    log_files = sorted(MATCH_LOGS_DIR.glob("match_*.jsonl"))
    if not log_files:
        raise RuntimeError(f"هیچ لاگ مسابقه‌ای در {MATCH_LOGS_DIR} پیدا نشد.")

    for log_path in log_files:
        sightings = load_raw_sightings(log_path)
        events = dedupe_to_play_events(sightings)
        card_sequence = [e["card"] for e in events]
        if len(card_sequence) >= 2:  # توالی با کمتر از ۲ کارت برای train بی‌فایده است
            sequences.append(card_sequence)
        print(f"  {log_path.name}: {len(sightings)} sighting خام -> {len(card_sequence)} رویداد بازی")

    return sequences


def main() -> None:
    print("در حال پردازش لاگ‌های خام...")
    sequences = build_all_sequences()

    out_path = SEQUENCES_DIR / "sequences.json"
    out_path.write_text(json.dumps(sequences, ensure_ascii=False, indent=2), encoding="utf-8")

    total_events = sum(len(s) for s in sequences)
    print(f"\n{len(sequences)} توالی مسابقه ذخیره شد ({total_events} رویداد کارت در مجموع).")
    print(f"خروجی: {out_path}")


if __name__ == "__main__":
    main()
