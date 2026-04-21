"""Camera capture and MediaPipe pose landmark detection.

Uses the MediaPipe legacy solutions API (mp.solutions.pose.Pose).
The Tasks API (PoseLandmarker) crashes on RPi5 (Cortex-A76) with a fatal
cv::remap assertion in MediaPipe's bundled OpenCV 4.5.5 due to XNNPACK's
SVE probe failing silently and corrupting bounding-box outputs.

Camera priority:
  1. picamera2 (RPi Camera Module 3, BGR888 format)
  2. OpenCV VideoCapture (USB webcam or built-in — PC development)

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
# MediaPipe — prefer the legacy solutions API.
#
# The Tasks API (PoseLandmarker) crashes on RPi5 (Cortex-A76) with a fatal
# assertion in MediaPipe's bundled OpenCV 4.5.5 remap() call regardless of
# running mode, model size, or thread count.  Root cause: XNNPACK's SVE
# probe fails on Cortex-A76 (no SVE) and corrupts bounding-box outputs used
# by the internal image-cropping step.
#
# The legacy mp.solutions.pose API uses a different internal graph that does
# not go through the same remap code path and runs stably on ARM64.
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    _MP_POSE_CLS = mp.solutions.pose.Pose   # validate attribute exists
    _HAS_MEDIAPIPE = True
except (ImportError, AttributeError):
    _HAS_MEDIAPIPE = False
    logger.warning("mediapipe not installed — VisionManager unavailable.")

# ---------------------------------------------------------------------------
# picamera2 availability
# ---------------------------------------------------------------------------
_HAS_PICAMERA2 = False
try:
    from picamera2 import Picamera2   # type: ignore
    _HAS_PICAMERA2 = True
except ImportError:
    pass


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
        if not _HAS_MEDIAPIPE:
            raise ImportError("mediapipe is required but not installed.")

        self._config = config
        self._camera = None
        self._use_picamera2 = False
        self._use_gstreamer = False
        self._landmarker: Optional[PoseLandmarker] = None
        self._start_ns = time.perf_counter_ns()

        self._init_camera()
        self._init_landmarker()

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
        frame_bgr = self._grab_frame()
        if frame_bgr is None:
            return PostureLandmarks(normalized=[], world=[], is_valid=False)

        result = self._detect(frame_bgr)

        # Privacy: delete frame immediately
        del frame_bgr

        return result

    def capture_with_frame(self) -> Tuple[Optional[np.ndarray], PostureLandmarks]:
        """GUI mode: return (BGR frame, PostureLandmarks).

        The frame is never written to disk — it lives only for the current
        GUI render cycle.
        """
        frame_bgr = self._grab_frame()
        if frame_bgr is None:
            return None, PostureLandmarks(normalized=[], world=[], is_valid=False)

        landmarks = self._detect(frame_bgr)
        return frame_bgr, landmarks

    def release(self):
        """Release camera and MediaPipe resources."""
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
    # Internal — camera
    # ------------------------------------------------------------------

    def _init_camera(self):
        w, h = self._config.resolution
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

        # Collect device indices from /dev/video* — sort numerically so lower
        # indices (real sensor or loopback near 0) are tried first.
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

                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self._camera = cap
                    logger.info("Camera: V4L2 %s fmt=%s (%dx%d).",
                                dev_path, fmt_label, w, h)
                    return

                cap.release()
                logger.debug("V4L2 %s fmt=%s: opened but no frames.", dev_path, fmt_label)

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
                ret, frame = self._camera.read()
                if not ret or frame is None:
                    return None
            # Guarantee C-contiguous layout; some decoders (MJPEG via V4L2)
            # return strided views that crash MediaPipe's internal remap().
            return np.ascontiguousarray(frame)
        except Exception:
            logger.exception("Frame capture error.")
            return None

    # ------------------------------------------------------------------
    # Internal — MediaPipe
    # ------------------------------------------------------------------

    def _init_landmarker(self):
        # Legacy solutions API — avoids the Tasks API remap crash on RPi5.
        # model_complexity: 0=lite, 1=full, 2=heavy
        self._landmarker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("Pose estimator ready (mp.solutions.pose, complexity=1).")

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
            # Ensure C-contiguous uint8 layout — MJPEG-decoded frames from
            # v4l2loopback can have strided views that MediaPipe rejects.
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            result = self._landmarker.process(rgb)
        except Exception:
            logger.exception("Landmark detection error.")
            return PostureLandmarks(normalized=[], world=[], is_valid=False)

        if not result.pose_landmarks:
            return PostureLandmarks(normalized=[], world=[], is_valid=False)

        # Legacy API returns landmark lists directly (not wrapped in a list-of-poses).
        # Each landmark has .x, .y, .z, .visibility — same attribute names as the
        # Tasks API NormalizedLandmark, so posture_logic/math_utils need no changes.
        return PostureLandmarks(
            normalized=result.pose_landmarks.landmark,
            world=result.pose_world_landmarks.landmark if result.pose_world_landmarks else [],
            is_valid=True,
        )
