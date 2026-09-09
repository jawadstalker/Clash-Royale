"""
Train مدل YOLOv8 برای تشخیص کارت‌ها با دیتاست ساخته‌شده در generate_dataset.py.

پیش‌نیاز:
    pip install ultralytics
    python src/generate_dataset.py --count 3000

اجرا:
    python src/train_yolo.py --epochs 50 --model yolov8n.pt
"""

import argparse

from ultralytics import YOLO

from config import DATASET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 روی دیتاست کارت‌های Clash Royale")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="مدل پایه: yolov8n (سریع/سبک) بهترین شروع برای real-time روی CPU/GPU معمولی است",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} پیدا نشد — اول python src/generate_dataset.py را اجرا کنید."
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
