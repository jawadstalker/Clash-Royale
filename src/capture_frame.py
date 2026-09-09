"""
گرفتن فریم زنده از امولاتور اندروید با ADB.

پیش‌نیاز: ADB باید نصب باشد و امولاتور/موبایل با `adb devices` قابل دیدن باشد.

اجرا (یک اسکرین‌شات تک):
    python src/capture_frame.py

اجرا (گرفتن پیوسته با فریم‌ریت مشخص، برای تست pipeline):
    python src/capture_frame.py --loop --fps 5
"""

import argparse
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np

from config import FRAMES_DIR


def capture_single_frame() -> np.ndarray:
    """یک فریم را با adb exec-out screencap می‌گیرد و به آرایه OpenCV تبدیل می‌کند."""
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        capture_output=True,
        check=True,
    )
    img_array = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("نتوانستم فریم را دیکود کنم — مطمئن شوید adb به دستگاه وصل است (adb devices).")
    return frame


def save_frame(frame: np.ndarray) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = FRAMES_DIR / f"frame_{timestamp}.png"
    cv2.imwrite(str(filepath), frame)
    print(f"ذخیره شد: {filepath}")


def loop_capture(fps: float) -> None:
    interval = 1.0 / fps
    print(f"گرفتن فریم پیوسته با {fps} FPS شروع شد. برای توقف Ctrl+C بزنید.")
    try:
        while True:
            start = time.time()
            frame = capture_single_frame()
            save_frame(frame)
            elapsed = time.time() - start
            time.sleep(max(0.0, interval - elapsed))
    except KeyboardInterrupt:
        print("\nمتوقف شد.")


def main() -> None:
    parser = argparse.ArgumentParser(description="گرفتن فریم زنده از امولاتور Clash Royale")
    parser.add_argument("--loop", action="store_true", help="گرفتن مداوم فریم به‌جای یک عکس تک")
    parser.add_argument("--fps", type=float, default=5.0, help="فریم بر ثانیه در حالت loop")
    args = parser.parse_args()

    if args.loop:
        loop_capture(args.fps)
    else:
        frame = capture_single_frame()
        save_frame(frame)


if __name__ == "__main__":
    main()
