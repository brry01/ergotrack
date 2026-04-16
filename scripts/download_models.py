"""Download the MediaPipe PoseLandmarker model file (~30MB) before first run."""
import os
import sys
import urllib.request

MODELS = {
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    ),
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
        sys.stdout.write(f"\r  [{bar}] {pct}%")
        sys.stdout.flush()


def download(name: str):
    url = MODELS[name]
    dest = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(dest):
        print(f"  {name} already exists, skipping.")
        return
    print(f"Downloading {name} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n  Saved to {dest}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pose_landmarker_full.task"
    if target not in MODELS:
        print(f"Unknown model: {target}. Choose from: {list(MODELS)}")
        sys.exit(1)
    download(target)
    print("Done.")
