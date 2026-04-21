"""Download pose model files before first run."""
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
    # MoveNet Lightning — used by ai-edge-litert backend (avoids MediaPipe's
    # remap crash on RPi5 / Cortex-A76).  Input: 192×192 RGB uint8.
    # Output: [1,1,17,3] → (y_norm, x_norm, confidence) per keypoint.
    # TFHub serves the file via HTTP redirect; we set a browser User-Agent
    # so the server accepts the request.
    "movenet_lightning.tflite": (
        "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning"
        "/tflite/int8/4?lite-format=tflite"
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
    # Use a browser-like User-Agent so TFHub and Google Storage accept the request.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent",
                          "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n  Saved to {dest}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "movenet_lightning.tflite"
    if target == "all":
        for name in MODELS:
            download(name)
    elif target not in MODELS:
        print(f"Unknown model: {target}. Choose from: {list(MODELS)} or 'all'")
        sys.exit(1)
    else:
        download(target)
    print("Done.")
