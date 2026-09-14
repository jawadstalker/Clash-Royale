"""
ترکیب دیتاست واقعی (دانلودشده از Roboflow) با دیتاست synthetic خودمان
برای یک دیتاست نهایی قوی‌تر: synthetic زیاد (تنوع بالا، حجم زیاد) + real کم
(واقع‌گرایانه، برای پر کردن sim-to-real gap).

پیش‌نیاز:
    python src/generate_dataset.py             (باید data/dataset/ ساخته باشد)
    python src/download_roboflow_dataset.py    (باید data/real_dataset/ ساخته باشد)

اجرا:
    python src/merge_datasets.py

خروجی:
    data/dataset_final/images/{train,val}/*.jpg|png
    data/dataset_final/labels/{train,val}/*.txt   (idهای کلاس با واژه‌نامه یکپارچه remap شده‌اند)
    data/dataset_final/data.yaml
"""

import re
import shutil
from pathlib import Path

import yaml

from config import PROJECT_ROOT
from fetch_cards import slugify

SYNTHETIC_DIR = PROJECT_ROOT / "data" / "dataset"
REAL_DIR = PROJECT_ROOT / "data" / "real_dataset"
FINAL_DIR = PROJECT_ROOT / "data" / "dataset_final"

# اسپلیت‌های احتمالی دیتاست دانلودی از Roboflow و این‌که به کدام اسپلیت نهایی ما نگاشت شوند
REAL_SPLIT_MAP = {"train": "train", "valid": "val", "test": "val"}


def normalize_roboflow_class(raw_name: str) -> str:
    """'Card Archers' -> 'archers'  (همان قرارداد slugify فایل fetch_cards.py)."""
    name = re.sub(r"^card[\s_]+", "", raw_name.strip(), flags=re.IGNORECASE)
    return slugify(name)


def load_yaml_classes(yaml_path: Path) -> list[str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return data["names"] if isinstance(data["names"], list) else list(data["names"].values())


def build_unified_vocab(synthetic_classes: list[str], real_classes_raw: list[str]) -> list[str]:
    """واژه‌نامه یکپارچه: ترتیب کلاس‌های synthetic حفظ می‌شود (چون لیبل‌های آن از قبل با این ترتیب نوشته شده‌اند)،
    و کلاس‌های جدیدی که فقط در دیتاست واقعی هستند به انتها اضافه می‌شوند."""
    unified = list(synthetic_classes)
    for raw in real_classes_raw:
        normalized = normalize_roboflow_class(raw)
        if normalized not in unified:
            unified.append(normalized)
    return unified


def copy_split(
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    prefix: str,
    class_id_remap: dict[int, int] | None = None,
) -> int:
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    if not src_images_dir.exists():
        return 0

    for img_path in src_images_dir.iterdir():
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = src_labels_dir / f"{img_path.stem}.txt"

        new_stem = f"{prefix}_{img_path.stem}"
        shutil.copy2(img_path, dst_images_dir / f"{new_stem}{img_path.suffix}")

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            if class_id_remap is not None:
                remapped_lines = []
                for line in lines:
                    parts = line.split()
                    if not parts:
                        continue
                    old_id = int(parts[0])
                    new_id = class_id_remap.get(old_id, old_id)
                    remapped_lines.append(" ".join([str(new_id)] + parts[1:]))
                lines = remapped_lines
            (dst_labels_dir / f"{new_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        else:
            (dst_labels_dir / f"{new_stem}.txt").write_text("", encoding="utf-8")

        count += 1
    return count


def main() -> None:
    synthetic_yaml = SYNTHETIC_DIR / "data.yaml"
    real_yaml = REAL_DIR / "data.yaml"

    if not synthetic_yaml.exists():
        raise FileNotFoundError(f"{synthetic_yaml} پیدا نشد — اول src/generate_dataset.py را اجرا کنید.")
    if not real_yaml.exists():
        raise FileNotFoundError(f"{real_yaml} پیدا نشد — اول src/download_roboflow_dataset.py را اجرا کنید.")

    synthetic_classes = load_yaml_classes(synthetic_yaml)
    real_classes_raw = load_yaml_classes(real_yaml)

    unified_vocab = build_unified_vocab(synthetic_classes, real_classes_raw)
    print(f"واژه‌نامه یکپارچه: {len(unified_vocab)} کلاس ({len(synthetic_classes)} از synthetic + "
          f"{len(unified_vocab) - len(synthetic_classes)} کلاس جدید فقط در دیتاست واقعی)")

    # نگاشت idهای کلاس دیتاست واقعی به idهای واژه‌نامه یکپارچه
    real_id_remap = {}
    for old_id, raw_name in enumerate(real_classes_raw):
        normalized = normalize_roboflow_class(raw_name)
        real_id_remap[old_id] = unified_vocab.index(normalized)

    total_images = 0

    # کپی دیتاست synthetic (idهای آن از قبل با unified_vocab یکی است، پس remap لازم نیست)
    for split in ("train", "val"):
        n = copy_split(
            SYNTHETIC_DIR / "images" / split,
            SYNTHETIC_DIR / "labels" / split,
            FINAL_DIR / "images" / split,
            FINAL_DIR / "labels" / split,
            prefix="synth",
        )
        print(f"  synthetic/{split}: {n} تصویر کپی شد")
        total_images += n

    # کپی دیتاست واقعی با remap idهای کلاس
    for real_split, final_split in REAL_SPLIT_MAP.items():
        n = copy_split(
            REAL_DIR / real_split / "images",
            REAL_DIR / real_split / "labels",
            FINAL_DIR / "images" / final_split,
            FINAL_DIR / "labels" / final_split,
            prefix=f"real_{real_split}",
            class_id_remap=real_id_remap,
        )
        print(f"  real/{real_split} -> {final_split}: {n} تصویر کپی شد")
        total_images += n

    # نوشتن data.yaml نهایی
    final_yaml = FINAL_DIR / "data.yaml"
    final_yaml.write_text(
        f"path: {FINAL_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(unified_vocab)}\n"
        f"names: {unified_vocab!r}\n".replace("'", '"'),
        encoding="utf-8",
    )

    print(f"\nدیتاست نهایی ترکیبی آماده شد: {FINAL_DIR}  ({total_images} تصویر در مجموع)")
    print(f"برای train: python src/train_yolo.py  (با تغییر مسیر data.yaml به {final_yaml})")


if __name__ == "__main__":
    main()
