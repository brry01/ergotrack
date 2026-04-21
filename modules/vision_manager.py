"""Camera capture and pose landmark detection.

Inference backend priority:
  1. ai-edge-litert + MoveNet Lightning TFLite  (RPi5 / ARM64 — avoids the
     MediaPipe remap crash caused by XNNPACK's SVE probe on Cortex-A76)
  2. mediapipe mp.solutions.pose               (fallback for x86 / PC dev)

Camera priority:
  1. picamera2 (RPi Camera Module 3, BGR888 format)
  2. GStreamer libcamerasrc pipeline (RPi OS, Camera Module 3 in pyenv venv)
  3. OpenCV V4L2 / AUTO (USB webcam or built-in)

Privacy guarantee: frames are deleted from memory immediately after
landmark detection in headless (capture_and_detect) mode and are never
written to disk in any mode.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import glob
import re

import cv2
import numpy as np

from modules.config_profile import VisionConfig
from modules.posture_logic import PostureLandmarks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend 1: ai-edge-litert + MoveNet (preferred on ARM64 / RPi5)
#
# MediaPipe's bundled OpenCV 4.5.5 crashes in remap() on Cortex-A76 because
# XNNPACK's SVE probe (prctl PR_SVE_GET_VL) fails silently and corrupts the
# bounding-box outputs used by MediaPipe's internal image-crop step.
# ai-edge-litert runs TFLite inference directly — no MediaPipe C++ graph,
# no bundled OpenCV, no remap call.
# ---------------------------------------------------------------------------
_HAS_LITERT = False
try:
    from ai_edge_litert.interpreter import Interpreter as _LiteRTInterpreter
    _HAS_LITERT = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Backend 2: MediaPipe legacy solutions API (fallback)
# ---------------------------------------------------------------------------
_HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    _mp_pose_test = mp.solutions.pose.Pose   # validate attribute exists
    del _mp_pose_test
    _HAS_MEDIAPIPE = True
except (ImportError, AttributeError):
    pass

if not _HAS_LITERT and not _HAS_MEDIAPIPE:
    logger.warning("Neither ai-edge-litert nor mediapipe found — VisionManager unavailable.")


# ---------------------------------------------------------------------------
# MoveNet landmark adapter
# ---------------------------------------------------------------------------

class _LM:
    """Minimal landmark with x, y, z, visibility (matches MediaPipe's API)."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x: float = 0.5, y: float = 0.5,
                 z: float = 0.0, visibility: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _LMList:
    """Wraps a list as .landmark attribute (matches mp.solutions result)."""
    def __init__(self, lms):
        self.landmark = lms


class _MoveNetResult:
    """Mimics mp.solutions.pose.Pose result structure."""
    def __init__(self, lms):
        self.pose_landmarks = _LMList(lms) if lms else None
        self.pose_world_landmarks = None


class _MoveNetDetector:
    """TFLite/MoveNet inference using ai-edge-litert.

    MoveNet keypoints (COCO order, 17 total):
        0=nose  1=l_eye  2=r_eye  3=l_ear  4=r_ear
        5=l_sho 6=r_sho  7=l_elbow 8=r_elbow
        9=l_wrist 10=r_wrist 11=l_hip 12=r_hip
        13=l_knee 14=r_knee 15=l_ankle 16=r_ankle

    Mapping to MediaPipe indices used by math_utils:
        MoveNet 3 → MP 7  (LEFT_EAR)
        MoveNet 4 → MP 8  (RIGHT_EAR)
        MoveNet 5 → MP 11 (LEFT_SHOULDER)
        MoveNet 6 → MP 12 (RIGHT_SHOULDER)
        MoveNet 11 → MP 23 (LEFT_HIP)
        MoveNet 12 → MP 24 (RIGHT_HIP)
    """

    _KP_MAP = {3: 7, 4: 8, 5: 11, 6: 12, 11: 23, 12: 24}
    _NEEDED = list(_KP_MAP.keys())           # MoveNet indices we must see
    _CONF_THRESHOLD = 0.25                   # min keypoint confidence
    _N_MP = 33                               # MediaPipe landmark count

    def __init__(self, model_path: str):
        self._interp = _LiteRTInterpreter(model_path=model_path)
        self._interp.allocate_tensors()
        inp = self._interp.get_input_details()[0]
        self._inp_idx = inp["index"]
        self._inp_dtype = inp["dtype"]
        self._inp_size = int(inp["shape"][1])   # 192 (Lightning) or 256 (Thunder)
        self._out_idx = self._interp.get_output_details()[0]["index"]

    def process(self, rgb: np.ndarray) -> _MoveNetResult:
        size = self._inp_size
        img = cv2.resize(rgb, (size, size))
        if self._inp_dtype == np.uint8:
            tensor = img[np.newaxis]                          # uint8 int8 quant
        else:
            tensor = (img.astype(np.float32) / 255.0)[np.newaxis]

        self._interp.set_tensor(self._inp_idx, tensor)
        self._interp.invoke()

        # Output shape: [1, 1, 17, 3]  →  [y_norm, x_norm, confidence]
        kps = self._interp.get_tensor(self._out_idx)[0, 0]  # (17, 3)

        # Reject frame if none of the key landmarks are confidently visible
        if all(kps[k, 2] < self._CONF_THRESHOLD for k in self._NEEDED):
            return _MoveNetResult(None)

        lms = [_LM() for _ in range(self._N_MP)]
        for mn_idx, mp_idx in self._KP_MAP.items():
            y, x, conf = kps[mn_idx]   # MoveNet outputs (y, x), not (x, y)
            lms[mp_idx] = _LM(x=float(x), y=float(y), z=0.0, visibility=float(conf))
        return _MoveNetResult(lms)

    def close(self):
        pass   # LiteRT interpreter has no explicit close

# ---------------------------------------------------------------------------
# picamera2 availability
# ---------------------------------------------------------------------------
_HAS_PICAMERA2 = False
try:
    from picamera2 import Picamera2   # type: ignore
    _HAS_PICAMERA2 = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# rpicam-vid subprocess capture (RPi OS Camera Module 3 without GStreamer)
# ---------------------------------------------------------------------------

import shutil
import subprocess
import threading
import queue


class _RpicamCapture:
    """Stream MJPEG frames from rpicam-vid stdout.

    This completely avoids v4l2loopback, GStreamer, and any kernel module
    setup.  rpicam-vid writes a raw MJPEG stream to stdout; we parse JPEG
    boundaries (FF D8 … FF D9) and decode each frame with cv2.imdecode.

    Works from any Python version since it uses only subprocess + OpenCV.
    """

    _SOI = b"\xff\xd8"   # JPEG Start Of Image marker
    _EOI = b"\xff\xd9"   # JPEG End Of Image marker
    _CHUNK = 65536        # read chunk size (64 KB)

    def __init__(self, width: int, height: int, fps: int):
        # Try rpicam-vid first (newer alias), fall back to libcamera-vid
        exe = "rpicam-vid" if shutil.which("rpicam-vid") else "libcamera-vid"
        self._proc = subprocess.Popen(
            [
                exe, "--nopreview", "-t", "0",
                "--width", str(width), "--height", str(height),
                "--framerate", str(fps),
                "--codec", "mjpeg", "--output", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._buf = b""

    # ------------------------------------------------------------------
    def isOpened(self) -> bool:
        return self._proc.poll() is None

    def read(self):
        """Return (True, BGR frame) or (False, None)."""
        while self.isOpened():
            chunk = self._proc.stdout.read(self._CHUNK)
            if not chunk:
                return False, None
            self._buf += chunk

            start = self._buf.find(self._SOI)
            if start == -1:
                self._buf = b""
                continue
            if start > 0:
                self._buf = self._buf[start:]

            end = self._buf.find(self._EOI, 2)
            if end == -1:
                continue   # incomplete frame — read more data

            jpeg = self._buf[: end + 2]
            self._buf = self._buf[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is not None:
                return True, frame
        return False, None

    def release(self):
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            self._proc.kill()


class _ThreadedCamera:
    """Wraps any camera (.read() / .release()) and captures frames in a
    background thread so the GUI/inference thread never blocks on I/O.

    Always returns the *latest* available frame instantly.
    """

    def __init__(self, cam):
        self._cam = cam
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self._cam.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            frame = self._frame
        return (frame is not None), frame

    def isOpened(self) -> bool:
        return not self._stop.is_set()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=3)
        self._cam.release()


class VisionManager:
    """Captures frames and detects pose landmarks.

    Parameters
    ----------
    config:
        Vision parameters (fps, resolution, model path).
    simulate:
        When True, the manager is not initialised — callers should use
        CameraSimulator instead and should never call this class.

    Context manager usage::

        with VisionManager(config) as vm:
            lms = vm.capture_and_detect()
    """

    def __init__(self, config: VisionConfig, simulate: bool = False):
        if simulate:
            raise RuntimeError(
                "VisionManager should not be instantiated in simulation mode. "
                "Use CameraSimulator instead."
            )
        if not _HAS_LITERT and not _HAS_MEDIAPIPE:
            raise ImportError(
                "No pose inference backend found.\n"
                "Install ai-edge-litert:  pip install ai-edge-litert\n"
                "Then download the model: python scripts/download_models.py movenet_lightning.tflite"
            )

        self._config = config
        self._camera = None
        self._use_picamera2 = False
        self._use_gstreamer = False
        self._use_rpicam = False
        self._landmarker: Optional[PoseLandmarker] = None
        self._start_ns = time.perf_counter_ns()

        # Background inference cache — updated by _inference_loop thread.
        # GUI reads these instantly without blocking on inference.
        self._inf_frame: Optional[np.ndarray] = None
        self._inf_landmarks = PostureLandmarks(normalized=[], world=[], is_valid=False)
        self._inf_lock = threading.Lock()
        self._inf_stop = threading.Event()
        self._inf_thread: Optional[threading.Thread] = None

        self._init_camera()
        self._init_landmarker()

        # Start inference loop after both camera and landmarker are ready
        self._inf_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="ergo-inference"
        )
        self._inf_thread.start()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        self.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_and_detect(self) -> PostureLandmarks:
        """Headless mode: detect landmarks, discard frame immediately.

        The BGR frame is deleted as soon as landmarks are extracted —
        no image data persists in memory beyond this call.
        """
        with self._inf_lock:
            return self._inf_landmarks

    def capture_with_frame(self) -> Tuple[Optional[np.ndarray], PostureLandmarks]:
        """GUI mode: return latest cached (BGR frame, PostureLandmarks).

        Returns instantly — inference runs in the background thread.
        """
        with self._inf_lock:
            frame = self._inf_frame
            lms = self._inf_landmarks
        return frame, lms

    def release(self):
        """Release camera and inference resources."""
        self._inf_stop.set()
        if self._inf_thread is not None:
            self._inf_thread.join(timeout=3)

        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

        if self._camera is not None:
            try:
                if self._use_picamera2:
                    self._camera.stop()
                    self._camera.close()
                else:
                    self._camera.release()
            except Exception:
                pass
            self._camera = None

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _inference_loop(self):
        """Continuously capture frames and run pose inference.

        Stores latest (frame, landmarks) in the cache; GUI reads from it
        without ever blocking on I/O or inference.

        Design notes
        ------------
        * ``_inf_frame`` is updated *before* inference so the video panel
          refreshes at camera FPS (10 Hz) even while inference runs at its
          own pace (~3–5 Hz on RPi5).
        * Same-frame deduplication: inference is skipped when the camera
          hasn't produced a new frame yet, avoiding wasted CPU cycles when
          the inference loop is faster than the camera.
        """
        _last_frame_id: int = -1   # id() of the last frame we ran inference on

        while not self._inf_stop.is_set():
            frame_bgr = self._grab_frame()
            if frame_bgr is None:
                time.sleep(0.05)
                continue

            # ── 1. Always push the latest frame to the GUI immediately ──────
            with self._inf_lock:
                self._inf_frame = frame_bgr

            # ── 2. Skip inference if the camera hasn't delivered a new frame ─
            fid = id(frame_bgr)
            if fid == _last_frame_id:
                # Inference is faster than the camera; yield briefly and wait.
                time.sleep(0.02)
                continue
            _last_frame_id = fid

            # ── 3. Run inference and update landmark cache ───────────────────
            landmarks = self._detect(frame_bgr)
            with self._inf_lock:
                self._inf_landmarks = landmarks

    # ------------------------------------------------------------------
    # Internal — camera
    # ------------------------------------------------------------------

    def _init_camera(self):
        w, h = self._config.resolution

        # --- 0. rpicam-vid subprocess (RPi OS Camera Module 3, no GStreamer) -
        # This is the most reliable path on RPi OS when running in a pyenv
        # venv (where picamera2/libcamera bindings aren't importable).
        # rpicam-vid writes raw MJPEG to stdout; we parse JPEG boundaries.
        if shutil.which("rpicam-vid") or shutil.which("libcamera-vid"):
            try:
                cap = _RpicamCapture(w, h, self._config.fps)
                # Give rpicam-vid ~1 s to start streaming
                time.sleep(1.0)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    # Wrap in threaded buffer so GUI never blocks on I/O
                    self._camera = _ThreadedCamera(cap)
                    self._use_rpicam = True
                    logger.info("Camera: rpicam-vid MJPEG subprocess (%dx%d).", w, h)
                    return
                cap.release()
                logger.debug("rpicam-vid opened but gave no frames.")
            except Exception:
                logger.debug("rpicam-vid init error.", exc_info=True)

        # --- 1. picamera2 (if importable in this Python env) ----------------
        if _HAS_PICAMERA2:
            try:
                self._camera = Picamera2()
                cam_cfg = self._camera.create_preview_configuration(
                    main={"size": (w, h), "format": "BGR888"}
                )
                self._camera.configure(cam_cfg)
                self._camera.start()
                self._use_picamera2 = True
                logger.info("Camera: picamera2 (%dx%d).", w, h)
                return
            except Exception:
                logger.exception("picamera2 init failed — falling back to OpenCV.")
                self._camera = None

        # OpenCV fallback — try backends in priority order until frames arrive.
        #
        # On RPi OS with Camera Module 3 (rp1-cfe raw CSI device), standard
        # V4L2 capture devices are NOT available to OpenCV.  libcamera owns the
        # sensor exclusively.  The only way to read frames from a non-system
        # Python (e.g., pyenv venv) is via GStreamer's libcamerasrc element,
        # which bridges libcamera → GStreamer → OpenCV appsink.
        #
        # Candidate order:
        #   1. GStreamer / libcamerasrc  — RPi OS with Camera Module 3
        #   2. V4L2 + MJPEG             — USB webcams and some CSI cameras
        #   3. V4L2 + YUYV              — another common USB format
        #   4. OpenCV AUTO              — PC built-in webcams

        fps = self._config.fps

        # --- 1. GStreamer libcamerasrc (RPi Camera Module 3 on RPi OS) -------
        # Requires: sudo apt install gstreamer1.0-libcamera gstreamer1.0-plugins-good
        #           and OpenCV built with GStreamer support (default in apt opencv)
        gst_pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        try:
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self._camera = cap
                    self._use_gstreamer = True
                    logger.info("Camera: GStreamer/libcamerasrc (%dx%d).", w, h)
                    return
                cap.release()
                logger.debug("GStreamer pipeline opened but gave no frames.")
            else:
                logger.debug("GStreamer pipeline failed to open (libcamerasrc not available?).")
        except Exception:
            logger.debug("GStreamer init error.", exc_info=True)

        # --- 2-4. Standard OpenCV backends (scan all /dev/video* devices) -----
        #
        # On RPi OS the Camera Module 3 (rp1-cfe raw CSI) cannot be opened
        # directly by OpenCV.  A v4l2loopback virtual device fed by libcamera-vid
        # is the cleanest workaround when OpenCV is built without GStreamer:
        #
        #   sudo apt install v4l2loopback-dkms
        #   sudo modprobe v4l2loopback devices=1 video_nr=42 card_label=ErgoCamera exclusive_caps=1
        #   echo "options v4l2loopback devices=1 video_nr=42 card_label=ErgoCamera exclusive_caps=1" \
        #     | sudo tee /etc/modprobe.d/v4l2loopback.conf
        #   echo "v4l2loopback" | sudo tee -a /etc/modules
        #   sudo systemctl enable --now ergo-camera.service   # see scripts/setup_rpi.sh
        #
        # Scanning all /dev/video* devices lets us find the loopback device
        # regardless of which index the kernel assigned it.

        # Collect device indices from /dev/video* — sort numerically.
        # IMPORTANT: on RPi OS with PiSP ISP, devices like /dev/video20-35
        # belong to the pispbe backend and may return a single stale frame
        # then nothing.  We require _PROBE_FRAMES consecutive successes to
        # confirm a device is truly streaming (filters out ISP devices).
        _PROBE_FRAMES = 3

        dev_paths = sorted(
            glob.glob("/dev/video*"),
            key=lambda p: int(re.search(r"\d+", p).group()),
        )
        if not dev_paths:
            dev_paths = []   # Windows / no V4L2 → fall through to AUTO

        fourccs = [
            (cv2.VideoWriter_fourcc(*"MJPG"), "MJPEG"),
            (cv2.VideoWriter_fourcc(*"YUYV"), "YUYV"),
            (None,                            "default"),
        ]

        for dev_path in dev_paths:
            dev_idx = int(re.search(r"\d+", dev_path).group())
            for fourcc, fmt_label in fourccs:
                cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
                if fourcc is not None:
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)

                if not cap.isOpened():
                    cap.release()
                    break   # device not accessible — skip remaining fourccs

                # Read multiple frames to confirm continuous streaming.
                # Devices like pispbe return one stale frame then stall.
                good = sum(
                    1 for _ in range(_PROBE_FRAMES)
                    if cap.read()[0]
                )
                if good == _PROBE_FRAMES:
                    self._camera = _ThreadedCamera(cap)
                    logger.info("Camera: V4L2 %s fmt=%s (%dx%d).",
                                dev_path, fmt_label, w, h)
                    return

                cap.release()
                logger.debug("V4L2 %s fmt=%s: only %d/%d probe frames — skipping.",
                             dev_path, fmt_label, good, _PROBE_FRAMES)

        # Last resort: OpenCV AUTO (works on Windows/macOS with built-in webcam)
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                self._camera = cap
                logger.info("Camera: OpenCV AUTO (%dx%d).", w, h)
                return
        cap.release()

        raise RuntimeError(
            "No camera backend could deliver frames.\n"
            "\n"
            "RPi OS + Camera Module 3 — set up the v4l2loopback virtual device:\n"
            "  sudo apt install v4l2loopback-dkms ffmpeg\n"
            "  sudo modprobe v4l2loopback devices=1 video_nr=42 \\\n"
            "      card_label=ErgoCamera exclusive_caps=1\n"
            "  libcamera-vid --nopreview -t 0 --width 640 --height 480 \\\n"
            "      --framerate 15 --codec yuv420 --output - 2>/dev/null | \\\n"
            "  ffmpeg -f rawvideo -pix_fmt yuv420p -s 640x480 -r 15 -i - \\\n"
            "      -f v4l2 -pix_fmt yuv420p /dev/video42 &\n"
            "\n"
            "Then re-run ErgoTrack, or use: python main.py --simulate"
        )

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Capture a single BGR frame from the active camera."""
        try:
            if self._use_picamera2:
                frame = self._camera.capture_array()   # BGR888 numpy array
            else:
                ret, frame = self._camera.read()       # works for both OpenCV cap and _RpicamCapture
                if not ret or frame is None:
                    return None
            # Guarantee C-contiguous layout.
            return np.ascontiguousarray(frame)
        except Exception:
            logger.exception("Frame capture error.")
            return None

    # ------------------------------------------------------------------
    # Internal — MediaPipe
    # ------------------------------------------------------------------

    def _init_landmarker(self):
        # --- Backend 1: ai-edge-litert + MoveNet (preferred on ARM64) --------
        if _HAS_LITERT:
            movenet_path = os.path.join(
                os.path.dirname(self._config.model_path),
                "movenet_lightning.tflite",
            )
            if os.path.exists(movenet_path):
                self._landmarker = _MoveNetDetector(movenet_path)
                logger.info("Pose estimator: MoveNet Lightning (ai-edge-litert) — %s", movenet_path)
                return
            else:
                logger.warning(
                    "ai-edge-litert available but %s not found. "
                    "Run: python scripts/download_models.py movenet_lightning.tflite",
                    movenet_path,
                )

        # --- Backend 2: MediaPipe legacy solutions API (fallback) ------------
        if not _HAS_MEDIAPIPE:
            raise ImportError(
                "No pose inference backend available.\n"
                "Install ai-edge-litert and download the MoveNet model:\n"
                "  pip install ai-edge-litert\n"
                "  python scripts/download_models.py movenet_lightning.tflite"
            )
        self._landmarker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("Pose estimator: mp.solutions.pose (complexity=1).")

    def _ts_ms(self) -> int:  # kept for potential future VIDEO-mode restoration
        """Monotonically increasing timestamp in milliseconds.

        Uses perf_counter (not wall clock) to guarantee strict monotonicity
        even across NTP corrections.
        """
        return (time.perf_counter_ns() - self._start_ns) // 1_000_000

    def _detect(self, frame_bgr: np.ndarray) -> PostureLandmarks:
        """Run pose estimation on a BGR frame and return PostureLandmarks."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Ensure C-contiguous uint8 — MJPEG frames from v4l2loopback can
            # have strided layouts that confuse both LiteRT and MediaPipe.
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            result = self._landmarker.process(rgb)
        except Exception:
            logger.exception("Landmark detection error.")
            return PostureLandmarks(normalized=[], world=[], is_valid=False)

        if not result.pose_landmarks:
            return PostureLandmarks(normalized=[], world=[], is_valid=False)

        # Both backends expose .pose_landmarks.landmark (list with .x/.y/.z/.visibility)
        # and .pose_world_landmarks (None for MoveNet, set for MediaPipe).
        world = []
        if result.pose_world_landmarks is not None:
            world = result.pose_world_landmarks.landmark

        return PostureLandmarks(
            normalized=result.pose_landmarks.landmark,
            world=world,
            is_valid=True,
        )
