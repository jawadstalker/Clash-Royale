"""
تولید دیتاست synthetic برای train مدل YOLO تشخیص کارت.

ایده: چون annotate دستی هزاران فریم واقعی خیلی زمان‌بر است، کارت‌هایی که از API
دانلود کردیم (data/cards/*.png) را به‌صورت تصادفی (اندازه، موقعیت، چرخش کم، شفافیت)
روی فریم‌های خالی زمین بازی (data/backgrounds/*.png) می‌چسبانیم و چون خودمان محل
دقیق چسباندن را می‌دانیم، bounding box به‌صورت خودکار و دقیق ساخته می‌شود.

پیش‌نیاز قبل از اجرا:
    ۱. python src/fetch_cards.py  (برای داشتن تصاویر کارت‌ها)
    ۲. چند فریم "خالی" از زمین بازی (بدون هیچ واحدی، فقط زمین + برج‌ها) با
       capture_frame.py بگیرید و در data/backgrounds/ ذخیره کنید (۱۰-۲۰ تا کافیست،
       هرچه متنوع‌تر — نور/زاویه/رنگ حریف مختلف — بهتر)
    ۳. مقدار ARENA_BBOX را در config.py با مختصات واقعی گوشی خودتان تنظیم کنید

اجرا:
    python src/generate_dataset.py --count 3000 --val-split 0.15
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from config import ARENA_BBOX, BACKGROUNDS_DIR, CARDS_DIR, DATASET_DIR

# بازه اندازه واقعی کارت‌ها نسبت به عرض زمین بازی (تجربی - قابل تنظیم)
MIN_CARD_SCALE = 0.09
MAX_CARD_SCALE = 0.16
MAX_CARDS_PER_FRAME = 4
MIN_CARDS_PER_FRAME = 1


def load_card_files() -> list[Path]:
    files = sorted(p for p in CARDS_DIR.glob("*.png"))
    if not files:
        raise RuntimeError(f"هیچ تصویر کارتی در {CARDS_DIR} پیدا نشد. اول fetch_cards.py را اجرا کنید.")
    return files


def load_background_files() -> list[Path]:
    files = sorted(BACKGROUNDS_DIR.glob("*.png"))
    if not files:
        raise RuntimeError(
            f"هیچ فریم پس‌زمینه‌ای در {BACKGROUNDS_DIR} پیدا نشد. "
            "چند اسکرین‌شات خالی از زمین بازی با capture_frame.py بگیرید و آنجا بگذارید."
        )
    return files


def paste_card_on_background(
    background: Image.Image, card_img: Image.Image
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """یک کارت را در نقطه تصادفی داخل ARENA_BBOX می‌چسباند و bbox نرمال‌شده YOLO برمی‌گرداند."""
    bg_w, bg_h = background.size

    scale = random.uniform(MIN_CARD_SCALE, MAX_CARD_SCALE)
    card_w = int(bg_w * scale)
    ratio = card_img.height / card_img.width
    card_h = int(card_w * ratio)
    resized_card = card_img.resize((card_w, card_h), Image.LANCZOS)

    x_min, y_min = ARENA_BBOX["x_min"], ARENA_BBOX["y_min"]
    x_max, y_max = ARENA_BBOX["x_max"], ARENA_BBOX["y_max"]

    max_x = max(x_min, x_max - card_w)
    max_y = max(y_min, y_max - card_h)
    paste_x = random.randint(x_min, max_x)
    paste_y = random.randint(y_min, max_y)

    background.paste(resized_card, (paste_x, paste_y), resized_card)

    # bbox نرمال‌شده به فرمت YOLO: x_center, y_center, width, height (نسبت ۰ تا ۱)
    x_center = (paste_x + card_w / 2) / bg_w
    y_center = (paste_y + card_h / 2) / bg_h
    norm_w = card_w / bg_w
    norm_h = card_h / bg_h
    return background, (x_center, y_center, norm_w, norm_h)


def generate_samples(count: int, val_split: float) -> None:
    card_files = load_card_files()
    background_files = load_background_files()
    class_names = [p.stem for p in card_files]  # اسم فایل بدون پسوند = اسم کلاس

    images_train = DATASET_DIR / "images" / "train"
    images_val = DATASET_DIR / "images" / "val"
    labels_train = DATASET_DIR / "labels" / "train"
    labels_val = DATASET_DIR / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    n_val = int(count * val_split)

    for i in range(count):
        split = "val" if i < n_val else "train"
        images_dir = images_val if split == "val" else images_train
        labels_dir = labels_val if split == "val" else labels_train

        bg_path = random.choice(background_files)
        canvas = Image.open(bg_path).convert("RGBA").copy()

        n_cards = random.randint(MIN_CARDS_PER_FRAME, MAX_CARDS_PER_FRAME)
        chosen_cards = random.sample(card_files, k=min(n_cards, len(card_files)))

        label_lines = []
        for card_path in chosen_cards:
            card_img = Image.open(card_path).convert("RGBA")
            class_id = class_names.index(card_path.stem)
            canvas, (xc, yc, w, h) = paste_card_on_background(canvas, card_img)
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        out_name = f"synth_{i:05d}"
        canvas.convert("RGB").save(images_dir / f"{out_name}.jpg", quality=90)
        (labels_dir / f"{out_name}.txt").write_text("\n".join(label_lines), encoding="utf-8")

        if (i + 1) % 200 == 0:
            print(f"  ساخته شد: {i + 1}/{count}")

    # فایل data.yaml برای ultralytics
    data_yaml = DATASET_DIR / "data.yaml"
    yaml_content = (
        f"path: {DATASET_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(class_names, ensure_ascii=False)}\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")

    print(f"\nتمام شد. {count} فریم ساخته شد ({count - n_val} train / {n_val} val).")
    print(f"تعداد کلاس‌ها (کارت‌ها): {len(class_names)}")
    print(f"فایل کانفیگ برای train: {data_yaml}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ساخت دیتاست synthetic برای train YOLO")
    parser.add_argument("--count", type=int, default=2000, help="تعداد کل فریم‌های ساختگی")
    parser.add_argument("--val-split", type=float, default=0.15, help="نسبت داده validation")
    args = parser.parse_args()

    generate_samples(args.count, args.val_split)


if __name__ == "__main__":
    main()
