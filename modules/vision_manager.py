"""Camera capture and MediaPipe pose landmark detection.

Uses the MediaPipe Tasks API (mediapipe >= 0.10) with PoseLandmarker in
VIDEO running mode, which applies temporal smoothing between frames.

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

import cv2
import numpy as np

from modules.config_profile import VisionConfig
from modules.posture_logic import PostureLandmarks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe Tasks API imports
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions
    _HAS_MEDIAPIPE = True
except ImportError:
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

        # --- 2-4. Standard OpenCV backends (USB webcams / PC built-ins) ------
        candidates = [
            (cv2.CAP_V4L2,  cv2.VideoWriter_fourcc(*"MJPG"), "V4L2/MJPEG"),
            (cv2.CAP_V4L2,  cv2.VideoWriter_fourcc(*"YUYV"), "V4L2/YUYV"),
            (cv2.CAP_ANY,   None,                            "AUTO"),
        ]
        for backend, fourcc, label in candidates:
            cap = cv2.VideoCapture(0, backend)
            if fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)

            if not cap.isOpened():
                cap.release()
                continue

            # Verify that frames actually arrive (not just that the device opened)
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                self._camera = cap
                logger.info("Camera: OpenCV %s (%dx%d).", label, w, h)
                return

            logger.debug("Camera backend %s opened but gave no frames — trying next.", label)
            cap.release()

        raise RuntimeError(
            "No camera backend could deliver frames.\n"
            "\n"
            "RPi OS + Camera Module 3: install GStreamer support and retry:\n"
            "  sudo apt install gstreamer1.0-libcamera gstreamer1.0-plugins-good\n"
            "\n"
            "USB webcam: check v4l2-ctl --list-devices\n"
            "\n"
            "Or use simulation mode: python main.py --simulate"
        )

    def _grab_frame(self) -> Optional[np.ndarray]:
        """Capture a single BGR frame from the active camera."""
        try:
            if self._use_picamera2:
                return self._camera.capture_array()   # BGR888 numpy array
            else:
                ret, frame = self._camera.read()
                return frame if ret else None
        except Exception:
            logger.exception("Frame capture error.")
            return None

    # ------------------------------------------------------------------
    # Internal — MediaPipe
    # ------------------------------------------------------------------

    def _init_landmarker(self):
        model_path = self._config.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Run:  python scripts/download_models.py"
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,   # temporal smoothing between frames
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        logger.info("PoseLandmarker loaded: %s", model_path)

    def _ts_ms(self) -> int:
        """Monotonically increasing timestamp in milliseconds.

        Uses perf_counter (not wall clock) to guarantee strict monotonicity
        even across NTP corrections.
        """
        return (time.perf_counter_ns() - self._start_ns) // 1_000_000

    def _detect(self, frame_bgr: np.ndarray) -> PostureLandmarks:
        """Run PoseLandmarker on a BGR frame and return PostureLandmarks."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, self._ts_ms())
        except Exception:
            logger.exception("Landmark detection error.")
            return PostureLandmarks(normalized=[], world=[], is_valid=False)

        if not result.pose_landmarks:
            return PostureLandmarks(normalized=[], world=[], is_valid=False)

        return PostureLandmarks(
            normalized=result.pose_landmarks[0],
            world=result.pose_world_landmarks[0] if result.pose_world_landmarks else [],
            is_valid=True,
        )
