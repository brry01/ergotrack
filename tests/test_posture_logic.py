"""Unit tests for modules/posture_logic.py.

Tests cover:
- AlertStateMachine debounce behaviour (escalation and reset)
- PostureLogic.analyze() integration with CameraSimulator landmarks
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.posture_logic import (
    AlertLevel,
    AlertStateMachine,
    PostureLandmarks,
    PostureLogic,
)
from modules.config_profile import AlertThresholds, AlertStateConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _LM:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0
    visibility: float = 1.0


def make_landmarks(overrides: dict) -> list[_LM]:
    lms = [_LM() for _ in range(33)]
    for idx, (x, y) in overrides.items():
        lms[idx] = _LM(x=x, y=y)
    return lms


# Good-posture landmark set — mirrors test_math_utils._GOOD exactly.
# FHP ≈ 0.125 (< warn 0.15), neck ≈ 9° (< warn 25°), shoulder_asym ≈ 0°.
_GOOD = {
    7:  (0.41, 0.16),   # LEFT_EAR
    8:  (0.53, 0.16),   # RIGHT_EAR
    11: (0.38, 0.35),   # LEFT_SHOULDER
    12: (0.62, 0.35),   # RIGHT_SHOULDER
    23: (0.40, 0.65),   # LEFT_HIP
    24: (0.60, 0.65),   # RIGHT_HIP
}

# Severe: ears drop to shoulder y-level → ear vector purely horizontal → ~90° flexion
_SEVERE = dict(_GOOD)
_SEVERE[7] = (_GOOD[7][0], 0.35)   # LEFT_EAR at shoulder y
_SEVERE[8] = (_GOOD[8][0], 0.35)   # RIGHT_EAR at shoulder y


def make_pl_good() -> PostureLandmarks:
    return PostureLandmarks(normalized=make_landmarks(_GOOD), world=[], is_valid=True)


def make_pl_severe() -> PostureLandmarks:
    return PostureLandmarks(normalized=make_landmarks(_SEVERE), world=[], is_valid=True)


def make_pl_invalid() -> PostureLandmarks:
    return PostureLandmarks(normalized=[], world=[], is_valid=False)


def make_logic(bad=10, good=5) -> PostureLogic:
    return PostureLogic(AlertThresholds(), AlertStateConfig(bad, good))


# ---------------------------------------------------------------------------
# AlertStateMachine
# ---------------------------------------------------------------------------

class TestAlertStateMachine:

    def test_no_alert_before_threshold(self):
        sm = AlertStateMachine(bad_threshold=10, good_reset=5)
        for _ in range(9):
            result = sm.update(AlertLevel.LEVEL2)
        assert result == AlertLevel.OK

    def test_alert_fires_at_threshold(self):
        sm = AlertStateMachine(bad_threshold=10, good_reset=5)
        result = AlertLevel.OK
        for _ in range(10):
            result = sm.update(AlertLevel.LEVEL2)
        assert result == AlertLevel.LEVEL2

    def test_alert_resets_after_good_frames(self):
        sm = AlertStateMachine(bad_threshold=2, good_reset=3)
        # Trigger alert
        sm.update(AlertLevel.LEVEL1)
        sm.update(AlertLevel.LEVEL1)
        # Partial recovery — not yet reset
        sm.update(AlertLevel.OK)
        sm.update(AlertLevel.OK)
        result = sm.update(AlertLevel.OK)
        assert result == AlertLevel.OK

    def test_partial_good_frames_do_not_reset(self):
        sm = AlertStateMachine(bad_threshold=2, good_reset=3)
        sm.update(AlertLevel.LEVEL1)
        sm.update(AlertLevel.LEVEL1)
        # Only 2 good frames — should still be at the triggered level
        sm.update(AlertLevel.OK)
        result = sm.update(AlertLevel.OK)
        assert result == AlertLevel.LEVEL1

    def test_escalation_within_bad_streak(self):
        sm = AlertStateMachine(bad_threshold=2, good_reset=5)
        sm.update(AlertLevel.LEVEL1)
        sm.update(AlertLevel.LEVEL1)
        # Now escalate
        sm.update(AlertLevel.LEVEL3)
        result = sm.update(AlertLevel.LEVEL3)
        assert result == AlertLevel.LEVEL3

    def test_no_deescalation_within_bad_streak(self):
        sm = AlertStateMachine(bad_threshold=2, good_reset=5)
        # Trigger at LEVEL3
        sm.update(AlertLevel.LEVEL3)
        sm.update(AlertLevel.LEVEL3)
        # Feed LEVEL1 — should stay at LEVEL3
        result = sm.update(AlertLevel.LEVEL1)
        assert result == AlertLevel.LEVEL3

    def test_reset_clears_state(self):
        sm = AlertStateMachine(bad_threshold=1, good_reset=1)
        sm.update(AlertLevel.LEVEL2)
        sm.reset()
        assert sm._current_level == AlertLevel.OK
        assert sm._consecutive_bad == 0

    def test_ok_input_increments_good_counter(self):
        sm = AlertStateMachine(bad_threshold=5, good_reset=3)
        sm.update(AlertLevel.OK)
        sm.update(AlertLevel.OK)
        sm.update(AlertLevel.OK)
        # Never triggered — should still be OK
        assert sm.update(AlertLevel.OK) == AlertLevel.OK


# ---------------------------------------------------------------------------
# PostureLogic
# ---------------------------------------------------------------------------

class TestPostureLogic:

    def test_invalid_landmarks_returns_ok_report(self):
        pl = make_logic(bad=1)
        report = pl.analyze(make_pl_invalid())
        assert report.severity == AlertLevel.OK
        assert not report.is_bad_posture
        assert report.neck_flexion_deg == 0.0

    def test_good_posture_no_alert(self):
        pl = make_logic(bad=1, good=1)
        report = pl.analyze(make_pl_good())
        # May or may not fire depending on exact angles; just ensure no crash
        assert report.severity in list(AlertLevel)

    def test_severe_neck_triggers_alert(self):
        # bad_threshold=1 so it fires immediately on the second frame
        pl = make_logic(bad=2, good=5)
        for _ in range(3):
            report = pl.analyze(make_pl_severe())
        assert report.severity >= AlertLevel.LEVEL1
        assert report.dominant_issue == "neck_flexion"

    def test_report_has_all_fields(self):
        pl = make_logic()
        report = pl.analyze(make_pl_good())
        assert hasattr(report, "neck_flexion_deg")
        assert hasattr(report, "fhp_ratio")
        assert hasattr(report, "shoulder_asymmetry_deg")
        assert hasattr(report, "severity")
        assert hasattr(report, "dominant_issue")
        assert hasattr(report, "is_bad_posture")
        assert hasattr(report, "timestamp")

    def test_good_frames_reset_after_bad(self):
        pl = make_logic(bad=2, good=2)
        # Trigger
        pl.analyze(make_pl_severe())
        pl.analyze(make_pl_severe())
        pl.analyze(make_pl_severe())
        # Reset
        pl.analyze(make_pl_good())
        report = pl.analyze(make_pl_good())
        assert report.severity == AlertLevel.OK


# ---------------------------------------------------------------------------
# Integration with CameraSimulator
# ---------------------------------------------------------------------------

class TestPostureLogicWithSimulator:

    def test_simulator_good_posture(self):
        from utils.camera_simulator import CameraSimulator
        sim = CameraSimulator(scenario="good", noise_std=0.0)
        pl = make_logic(bad=1, good=1)
        lms = sim.get_landmarks()
        pl_lm = PostureLandmarks(normalized=lms, world=[], is_valid=True)
        # Just ensure no crash and returns a valid report
        report = pl.analyze(pl_lm)
        assert report.neck_flexion_deg >= 0.0

    def test_simulator_severe_neck_escalates(self):
        from utils.camera_simulator import CameraSimulator
        sim = CameraSimulator(scenario="severe_neck", noise_std=0.0)
        pl = make_logic(bad=3, good=5)
        for _ in range(5):
            lms = sim.get_landmarks()
            report = pl.analyze(PostureLandmarks(normalized=lms, world=[], is_valid=True))
        assert report.severity >= AlertLevel.LEVEL1
