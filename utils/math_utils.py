"""Ergonomic metric calculations from MediaPipe pose landmarks.

All functions accept a list of 33 NormalizedLandmark objects as returned by
the MediaPipe Tasks API (mediapipe.tasks.vision.PoseLandmarker).

Coordinate convention: (0,0) = top-left, y increases downward, values in [0,1].
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Landmark index constants (MediaPipe Pose, 33 keypoints)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_neck_flexion(landmarks: list) -> float:
    """Return neck flexion in degrees.

    Measured as the angle between the ear→shoulder vector and the hip→shoulder
    vector, remapped so that 0° = perfectly upright and positive values
    indicate forward flexion.

    Uses the mean of left/right landmarks for robustness when one side is
    partially occluded.
    """
    ear = _mean_xy(landmarks, LEFT_EAR, RIGHT_EAR)
    shoulder = _mean_xy(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip = _mean_xy(landmarks, LEFT_HIP, RIGHT_HIP)

    v_ear = ear - shoulder
    v_hip = hip - shoulder

    norm_product = np.linalg.norm(v_ear) * np.linalg.norm(v_hip) + 1e-9
    cos_a = np.dot(v_ear, v_hip) / norm_product
    angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    # 180° = perfectly upright (ear above shoulder, hip below)
    # Subtract from 180 so 0° = good, positive = flexion
    return float(180.0 - angle_deg)


def compute_fhp_ratio(landmarks: list) -> float:
    """Return Forward Head Posture ratio (dimensionless).

    Measures how far the ear's x-coordinate is displaced from the shoulder
    mid-point, normalized by shoulder width.  Returns 0 when the ear is
    directly above the shoulder; positive values indicate forward displacement.
    Returns 0.0 if the shoulder width is too small to be reliable.
    """
    mean_ear_x = (landmarks[LEFT_EAR].x + landmarks[RIGHT_EAR].x) / 2.0
    mean_shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2.0
    shoulder_width = abs(landmarks[LEFT_SHOULDER].x - landmarks[RIGHT_SHOULDER].x)

    if shoulder_width < 1e-6:
        return 0.0

    return float(abs(mean_ear_x - mean_shoulder_x) / shoulder_width)


def compute_shoulder_asymmetry(landmarks: list) -> float:
    """Return shoulder tilt angle in degrees.

    0° = perfectly level shoulders; positive values indicate tilt.
    """
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    dy = ls.y - rs.y
    dx = ls.x - rs.x
    return float(abs(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9))))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean_xy(landmarks: list, idx_a: int, idx_b: int) -> np.ndarray:
    """Return the mean (x, y) of two landmarks as a numpy array."""
    a = landmarks[idx_a]
    b = landmarks[idx_b]
    return np.array([(a.x + b.x) / 2.0, (a.y + b.y) / 2.0])
