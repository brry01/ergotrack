"""Synthetic camera and landmark generator for PC simulation.

Produces MediaPipe-compatible NormalizedLandmark objects and synthetic BGR
frames without requiring a physical camera or the full Tasks API model.

Usage::

    sim = CameraSimulator(scenario="cycling")
    lms  = sim.get_landmarks()   # PostureLandmarks-compatible
    frame = sim.get_frame()      # 640×480 BGR numpy array
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Minimal landmark stand-in (mirrors mediapipe NormalizedLandmark interface)
# ---------------------------------------------------------------------------

@dataclass
class _FakeLandmark:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0
    visibility: float = 1.0


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

class SimulationScenario(enum.Enum):
    GOOD_POSTURE = "good"
    MILD_FHP = "mild_fhp"
    SEVERE_NECK = "severe_neck"
    SHOULDER_TILT = "shoulder_tilt"
    CYCLING = "cycling"


# Baseline good-posture coordinates (normalized image space, 640×480 camera)
_BASELINE: dict[int, tuple[float, float]] = {
    0:  (0.50, 0.12),   # NOSE
    7:  (0.44, 0.18),   # LEFT_EAR
    8:  (0.56, 0.18),   # RIGHT_EAR
    11: (0.38, 0.35),   # LEFT_SHOULDER
    12: (0.62, 0.35),   # RIGHT_SHOULDER
    13: (0.32, 0.52),   # LEFT_ELBOW
    14: (0.68, 0.52),   # RIGHT_ELBOW
    23: (0.40, 0.65),   # LEFT_HIP
    24: (0.60, 0.65),   # RIGHT_HIP
    25: (0.40, 0.82),   # LEFT_KNEE
    26: (0.60, 0.82),   # RIGHT_KNEE
}

# Skeleton connections to draw on the synthetic frame
_CONNECTIONS = [
    (7, 11), (8, 12),           # ear → shoulder
    (11, 12),                   # shoulders
    (11, 13), (12, 14),         # shoulder → elbow
    (11, 23), (12, 24),         # shoulder → hip
    (23, 24),                   # hips
    (23, 25), (24, 26),         # hip → knee
]

# Alert-level colours used for the stick figure border
_SCENARIO_COLORS = {
    SimulationScenario.GOOD_POSTURE:  (50, 205, 50),    # green
    SimulationScenario.MILD_FHP:      (0, 200, 255),    # yellow-ish
    SimulationScenario.SEVERE_NECK:   (0, 0, 220),      # red
    SimulationScenario.SHOULDER_TILT: (0, 165, 255),    # orange
}

_CYCLE_ORDER = [
    SimulationScenario.GOOD_POSTURE,
    SimulationScenario.MILD_FHP,
    SimulationScenario.SEVERE_NECK,
    SimulationScenario.SHOULDER_TILT,
]
_FRAMES_PER_SCENARIO = 75   # 5 s at 15 FPS


class CameraSimulator:
    """Generates synthetic pose landmarks and video frames.

    Parameters
    ----------
    scenario:
        One of the ``SimulationScenario`` enum values (or its string alias).
    frame_size:
        ``(width, height)`` of the generated BGR frame.
    noise_std:
        Standard deviation of Gaussian noise added to landmark coordinates.
    """

    def __init__(
        self,
        scenario: str | SimulationScenario = SimulationScenario.CYCLING,
        frame_size: tuple[int, int] = (640, 480),
        noise_std: float = 0.005,
    ):
        if isinstance(scenario, str):
            # Accept both "cycling" and "SimulationScenario.CYCLING"
            scenario_map = {s.value: s for s in SimulationScenario}
            scenario_map.update({s.name.lower(): s for s in SimulationScenario})
            scenario = scenario_map.get(scenario.lower(), SimulationScenario.CYCLING)
        self._base_scenario = scenario
        self._frame_size = frame_size
        self._noise_std = noise_std
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_scenario(self) -> SimulationScenario:
        if self._base_scenario != SimulationScenario.CYCLING:
            return self._base_scenario
        idx = (self._frame_count // _FRAMES_PER_SCENARIO) % len(_CYCLE_ORDER)
        return _CYCLE_ORDER[idx]

    def get_landmarks(self):
        """Return a list of 33 synthetic NormalizedLandmark-compatible objects.

        The returned list is compatible with the ``PostureLandmarks.normalized``
        field and can be passed directly to any function in ``utils.math_utils``.
        """
        coords = self._build_coords(self.current_scenario)
        lms = self._make_landmarks(coords)
        self._frame_count += 1
        return lms

    def get_frame(self) -> np.ndarray:
        """Return a synthetic 640×480 BGR frame with a stick figure overlay."""
        scenario = self.current_scenario
        coords = self._build_coords(scenario)
        w, h = self._frame_size

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)   # dark background

        color = _SCENARIO_COLORS.get(scenario, (180, 180, 180))

        # Draw connections
        for (a, b) in _CONNECTIONS:
            if a in coords and b in coords:
                ax, ay = int(coords[a][0] * w), int(coords[a][1] * h)
                bx, by = int(coords[b][0] * w), int(coords[b][1] * h)
                cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

        # Draw keypoints
        for idx, (nx, ny) in coords.items():
            px, py = int(nx * w), int(ny * h)
            cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)

        # Scenario label
        label = scenario.name.replace("_", " ").title()
        cv2.putText(frame, f"SIM: {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_coords(self, scenario: SimulationScenario) -> dict[int, tuple[float, float]]:
        """Return a dict mapping landmark index → (x, y) with noise applied."""
        coords: dict[int, tuple[float, float]] = {}

        # Start from the baseline
        for idx, (x, y) in _BASELINE.items():
            coords[idx] = (x, y)

        # Apply scenario-specific deformations
        if scenario == SimulationScenario.MILD_FHP:
            # Ears shift forward (in x) relative to shoulder centre
            for idx in (7, 8):
                x, y = coords[idx]
                coords[idx] = (x + 0.06, y + 0.03)

        elif scenario == SimulationScenario.SEVERE_NECK:
            # Ears drop toward shoulder level — severe neck flexion
            ear_y_target = coords[11][1] - 0.02   # just above shoulder
            for idx in (7, 8):
                x, _ = coords[idx]
                coords[idx] = (x + 0.04, ear_y_target)
            # Nose follows
            nx, _ = coords[0]
            coords[0] = (nx + 0.03, ear_y_target - 0.04)

        elif scenario == SimulationScenario.SHOULDER_TILT:
            # Left shoulder raised, right shoulder lowered
            lx, ly = coords[11]
            rx, ry = coords[12]
            coords[11] = (lx, ly - 0.06)
            coords[12] = (rx, ry + 0.06)

        # Add Gaussian noise to all keypoints
        rng = np.random.default_rng()
        for idx, (x, y) in coords.items():
            nx = float(np.clip(x + rng.normal(0, self._noise_std), 0.0, 1.0))
            ny = float(np.clip(y + rng.normal(0, self._noise_std), 0.0, 1.0))
            coords[idx] = (nx, ny)

        return coords

    def _make_landmarks(self, coords: dict[int, tuple[float, float]]) -> list:
        """Build a 33-element list of _FakeLandmark objects."""
        lms: list[_FakeLandmark] = []
        for i in range(33):
            if i in coords:
                x, y = coords[i]
                lms.append(_FakeLandmark(x=x, y=y, z=0.0, visibility=1.0))
            else:
                lms.append(_FakeLandmark(x=0.5, y=0.5, z=0.0, visibility=0.1))
        return lms
