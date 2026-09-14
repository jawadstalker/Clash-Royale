"""
Train مدل YOLOv8 برای تشخیص کارت‌ها با دیتاست ساخته‌شده در generate_dataset.py
(یا دیتاست نهایی ترکیبی از merge_datasets.py، که معمولاً دقت بهتری می‌دهد).

پیش‌نیاز:
    pip install ultralytics
    python src/generate_dataset.py --count 3000
    (اختیاری ولی توصیه‌شده) python src/download_roboflow_dataset.py && python src/merge_datasets.py

اجرا:
    python src/train_yolo.py --epochs 50 --model yolov8n.pt
    python src/train_yolo.py --data data/dataset_final/data.yaml   # با دیتاست ترکیبی real+synthetic
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import DATASET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 روی دیتاست کارت‌های Clash Royale")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="مدل پایه: yolov8n (سریع/سبک) بهترین شروع برای real-time روی CPU/GPU معمولی است",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="مسیر data.yaml (پیش‌فرض: دیتاست synthetic. برای دیتاست ترکیبی: data/dataset_final/data.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_yaml = args.data or (DATASET_DIR / "data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} پیدا نشد — اول python src/generate_dataset.py را اجرا کنید "
            "(یا برای دیتاست ترکیبی: download_roboflow_dataset.py + merge_datasets.py)."
        )

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name="clash_royale_card_detector",
    )

    print("\nآموزش تمام شد. بهترین وزن‌ها در runs/detect/clash_royale_card_detector/weights/best.pt ذخیره شد.")


if __name__ == "__main__":
    main()
