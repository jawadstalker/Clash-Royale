"""
تنظیمات پروژه.

توکن API را از https://developer.clashroyale.com بگیرید و اینجا قرار دهید،
یا بهتر: آن را در متغیر محیطی CLASH_ROYALE_API_TOKEN ذخیره کنید تا در گیت کامیت نشود.
"""

import os
from pathlib import Path

# --- API ---
API_TOKEN = os.environ.get("CLASH_ROYALE_API_TOKEN", "PUT_YOUR_TOKEN_HERE")

# اگر IP شما داینامیک است و نمی‌توانید whitelist کنید، از پروکسی RoyaleAPI استفاده کنید:
# BASE_URL = "https://proxy.royaleapi.dev/v1"
BASE_URL = "https://api.clashroyale.com/v1"

# --- مسیرها ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CARDS_DIR = DATA_DIR / "cards"
FRAMES_DIR = DATA_DIR / "frames"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"   # فریم‌های خالی زمین بازی (بدون هیچ واحدی روی آن)
DATASET_DIR = DATA_DIR / "dataset"           # دیتاست نهایی YOLO (images/ + labels/)

for _dir in (CARDS_DIR, FRAMES_DIR, BACKGROUNDS_DIR, DATASET_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- محدوده زمین بازی (Arena) روی تصویر ---
# این مقادیر بسته به رزولوشن گوشی/امولاتور شما فرق می‌کند.
# یک اسکرین‌شات با capture_frame.py بگیرید، آن را در یک ویرایشگر عکس باز کنید
# و مختصات پیکسلی گوشه بالا-چپ و پایین-راست زمین بازی (بدون شامل نوار الکسیر و کارت‌های دست خودتان) را اندازه بگیرید.
ARENA_BBOX = {
    "x_min": 40,    # TODO: با مختصات واقعی گوشی خودتان جایگزین کنید
    "y_min": 120,
    "x_max": 680,
    "y_max": 1150,
}
