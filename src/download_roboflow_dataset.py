"""
دانلود دیتاست واقعی و annotate‌شده Clash Royale از Roboflow Universe.

دیتاست: https://universe.roboflow.com/clash-olbwx/clash-royale-image-dataset
(۱۱۷۶ فریم واقعی از گیم‌پلی، ۵۳ کلاس کارت، لایسنس MIT)

پیش‌نیاز:
    pip install roboflow
    یک API key رایگان از https://app.roboflow.com/settings/api بگیرید
    (نیاز به ساخت اکانت رایگان روبوفلو دارد - این جدا از توکن Clash Royale API است)

اجرا:
    export ROBOFLOW_API_KEY="کلید_شما"
    python src/download_roboflow_dataset.py
"""

import os
from pathlib import Path

from config import PROJECT_ROOT

REAL_DATASET_DIR = PROJECT_ROOT / "data" / "real_dataset"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "متغیر محیطی ROBOFLOW_API_KEY تنظیم نشده. اول از app.roboflow.com/settings/api یک کلید رایگان بگیرید:\n"
            '    export ROBOFLOW_API_KEY="کلید_شما"'
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("clash-olbwx").project("clash-royale-image-dataset")
    version = project.version(1)

    REAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"در حال دانلود دیتاست واقعی به {REAL_DATASET_DIR} ...")
    dataset = version.download("yolov8", location=str(REAL_DATASET_DIR))

    print(f"\nدانلود تمام شد: {dataset.location}")
    print("مرحله بعد: python src/merge_datasets.py")


if __name__ == "__main__":
    main()
