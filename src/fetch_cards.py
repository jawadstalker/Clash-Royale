"""
دانلود لیست کامل کارت‌ها + تصویر هرکدام از API رسمی Clash Royale.

خروجی:
    data/cards/cards.json      -> متادیتای همه کارت‌ها (اسم، الکسیر، نوع، ...)
    data/cards/<card_name>.png -> تصویر هر کارت (برای train مدل تشخیص)

اجرا:
    python src/fetch_cards.py
"""

import json
import re
import sys

import requests

from config import API_TOKEN, BASE_URL, CARDS_DIR


def slugify(name: str) -> str:
    """تبدیل اسم کارت به نام فایل امن، مثل 'Giant Skeleton' -> 'giant_skeleton'."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def fetch_card_list() -> list[dict]:
    url = f"{BASE_URL}/cards"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()["items"]


def download_card_image(card: dict) -> None:
    # اولویت با تصویر evolution (اگر دارد) نیست - ما همان تصویر اصلی static را می‌خواهیم
    image_url = card.get("iconUrls", {}).get("medium")
    if not image_url:
        print(f"  ! تصویری برای {card['name']} پیدا نشد، رد شد")
        return

    img_resp = requests.get(image_url, timeout=15)
    img_resp.raise_for_status()

    filename = f"{slugify(card['name'])}.png"
    filepath = CARDS_DIR / filename
    filepath.write_bytes(img_resp.content)


def main() -> None:
    if API_TOKEN == "PUT_YOUR_TOKEN_HERE":
        print("خطا: اول توکن API را در src/config.py یا متغیر محیطی CLASH_ROYALE_API_TOKEN قرار دهید.")
        sys.exit(1)

    print("در حال گرفتن لیست کارت‌ها از API...")
    cards = fetch_card_list()
    print(f"{len(cards)} کارت پیدا شد.")

    # ذخیره متادیتا
    metadata_path = CARDS_DIR / "cards.json"
    metadata_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"متادیتا ذخیره شد: {metadata_path}")

    # دانلود تصاویر
    print("در حال دانلود تصاویر کارت‌ها...")
    for i, card in enumerate(cards, start=1):
        print(f"  [{i}/{len(cards)}] {card['name']}")
        download_card_image(card)

    print(f"\nتمام شد. تصاویر در {CARDS_DIR} ذخیره شدند.")


if __name__ == "__main__":
    main()
