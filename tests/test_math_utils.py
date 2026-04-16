"""Unit tests for utils/math_utils.py.

All tests use synthetic landmark lists created by make_landmarks() so there
is no dependency on MediaPipe or a camera.
"""
from __future__ import annotations

import sys
import os
import math
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.math_utils import (
    compute_fhp_ratio,
    compute_neck_flexion,
    compute_shoulder_asymmetry,
    LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _LM:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0
    visibility: float = 1.0


def make_landmarks(overrides: dict[int, tuple[float, float]]) -> list[_LM]:
    """Build a 33-element list with defaults and apply per-index (x, y) overrides."""
    lms = [_LM() for _ in range(33)]
    for idx, (x, y) in overrides.items():
        lms[idx] = _LM(x=x, y=y)
    return lms


# Shared good-posture landmark set.
#
# Design intent:
#   - Front-facing camera → wide shoulders (width ≈ 0.24) so FHP stays small.
#   - Ears are slightly offset in x from mean-shoulder-x (0.03 units) so that
#     the ear→shoulder vector has a non-zero x component, making neck_flexion
#     non-degenerate (~9° — well below the 25° warn threshold).
#   - Verified values:
#       FHP          = abs(0.47 - 0.50) / 0.24 ≈ 0.125  (< warn 0.15)
#       neck_flexion ≈ 9.2°  (< warn 25°)
#       shoulder_asym ≈ 0°
_GOOD = {
    LEFT_EAR:       (0.41, 0.16),
    RIGHT_EAR:      (0.53, 0.16),
    LEFT_SHOULDER:  (0.38, 0.35),
    RIGHT_SHOULDER: (0.62, 0.35),
    LEFT_HIP:       (0.40, 0.65),
    RIGHT_HIP:      (0.60, 0.65),
}


# ---------------------------------------------------------------------------
# compute_neck_flexion
# ---------------------------------------------------------------------------

class TestNeckFlexion:

    def test_upright_posture_near_zero(self):
        lms = make_landmarks(_GOOD)
        angle = compute_neck_flexion(lms)
        assert angle < 25.0, f"Expected below warn threshold (25°), got {angle:.1f}°"

    def test_severe_flexion_above_threshold(self):
        # Ears drop to exactly shoulder y-level → ear vector is purely horizontal
        # With asymmetric x-coords, this gives ~94° flexion (well above 40° threshold).
        overrides = dict(_GOOD)
        shoulder_y = (_GOOD[LEFT_SHOULDER][1] + _GOOD[RIGHT_SHOULDER][1]) / 2.0
        overrides[LEFT_EAR]  = (_GOOD[LEFT_EAR][0],  shoulder_y)
        overrides[RIGHT_EAR] = (_GOOD[RIGHT_EAR][0], shoulder_y)
        lms = make_landmarks(overrides)
        angle = compute_neck_flexion(lms)
        assert angle > 40.0, f"Expected severe flexion (>40°), got {angle:.1f}°"

    def test_mild_flexion_in_range(self):
        # Ear midway between its good position and shoulder level → moderate flexion
        overrides = dict(_GOOD)
        mid_y = (_GOOD[LEFT_EAR][1] + _GOOD[LEFT_SHOULDER][1]) / 2.0
        overrides[LEFT_EAR]  = (_GOOD[LEFT_EAR][0],  mid_y)
        overrides[RIGHT_EAR] = (_GOOD[RIGHT_EAR][0], mid_y)
        lms = make_landmarks(overrides)
        angle = compute_neck_flexion(lms)
        assert 10.0 < angle < 80.0, f"Expected mild-moderate flexion, got {angle:.1f}°"

    def test_returns_float(self):
        lms = make_landmarks(_GOOD)
        assert isinstance(compute_neck_flexion(lms), float)

    def test_non_negative(self):
        lms = make_landmarks(_GOOD)
        assert compute_neck_flexion(lms) >= 0.0


# ---------------------------------------------------------------------------
# compute_fhp_ratio
# ---------------------------------------------------------------------------

class TestFHPRatio:

    def test_ear_over_shoulder_near_zero(self):
        # Ear x == shoulder x → FHP ≈ 0
        overrides = dict(_GOOD)
        shoulder_cx = (_GOOD[LEFT_SHOULDER][0] + _GOOD[RIGHT_SHOULDER][0]) / 2.0
        half = (_GOOD[RIGHT_SHOULDER][0] - _GOOD[LEFT_SHOULDER][0]) / 2.0
        overrides[LEFT_EAR]  = (shoulder_cx - half * 0.1, _GOOD[LEFT_EAR][1])
        overrides[RIGHT_EAR] = (shoulder_cx + half * 0.1, _GOOD[RIGHT_EAR][1])
        lms = make_landmarks(overrides)
        ratio = compute_fhp_ratio(lms)
        assert ratio < 0.15, f"Expected near-zero FHP ratio, got {ratio:.4f}"

    def test_forward_head_produces_nonzero_ratio(self):
        overrides = dict(_GOOD)
        overrides[LEFT_EAR]  = (_GOOD[LEFT_EAR][0]  + 0.08, _GOOD[LEFT_EAR][1])
        overrides[RIGHT_EAR] = (_GOOD[RIGHT_EAR][0] + 0.08, _GOOD[RIGHT_EAR][1])
        lms = make_landmarks(overrides)
        ratio = compute_fhp_ratio(lms)
        assert ratio > 0.0, "Expected positive FHP ratio for forward-displaced ears"

    def test_zero_shoulder_width_returns_zero(self):
        overrides = dict(_GOOD)
        # Collapse shoulders to the same x
        overrides[LEFT_SHOULDER]  = (0.50, _GOOD[LEFT_SHOULDER][1])
        overrides[RIGHT_SHOULDER] = (0.50, _GOOD[RIGHT_SHOULDER][1])
        lms = make_landmarks(overrides)
        assert compute_fhp_ratio(lms) == 0.0

    def test_returns_non_negative(self):
        lms = make_landmarks(_GOOD)
        assert compute_fhp_ratio(lms) >= 0.0


# ---------------------------------------------------------------------------
# compute_shoulder_asymmetry
# ---------------------------------------------------------------------------

class TestShoulderAsymmetry:

    def test_level_shoulders_near_zero(self):
        # Left and right shoulder at the same y
        lms = make_landmarks(_GOOD)
        asym = compute_shoulder_asymmetry(lms)
        assert asym < 2.0, f"Expected ~0° asymmetry, got {asym:.2f}°"

    def test_tilted_shoulders_produce_angle(self):
        overrides = dict(_GOOD)
        overrides[LEFT_SHOULDER]  = (_GOOD[LEFT_SHOULDER][0],  _GOOD[LEFT_SHOULDER][1]  - 0.06)
        overrides[RIGHT_SHOULDER] = (_GOOD[RIGHT_SHOULDER][0], _GOOD[RIGHT_SHOULDER][1] + 0.06)
        lms = make_landmarks(overrides)
        asym = compute_shoulder_asymmetry(lms)
        assert asym > 5.0, f"Expected significant asymmetry, got {asym:.2f}°"

    def test_returns_non_negative(self):
        lms = make_landmarks(_GOOD)
        assert compute_shoulder_asymmetry(lms) >= 0.0

    def test_returns_float(self):
        lms = make_landmarks(_GOOD)
        assert isinstance(compute_shoulder_asymmetry(lms), float)
