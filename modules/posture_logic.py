"""Posture analysis and alert state machine.

PostureLogic.analyze() is the single public entry point: it consumes a
PostureLandmarks object, computes ergonomic metrics, runs them through a
debounce state machine, and returns a PostureReport.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional

from utils.math_utils import (
    compute_fhp_ratio,
    compute_neck_flexion,
    compute_shoulder_asymmetry,
)
from modules.config_profile import AlertThresholds, AlertStateConfig


# ---------------------------------------------------------------------------
# AlertLevel
# ---------------------------------------------------------------------------

class AlertLevel(enum.IntEnum):
    OK = 0
    LEVEL1 = 1   # mild
    LEVEL2 = 2   # moderate
    LEVEL3 = 3   # severe


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PostureLandmarks:
    """Wrapper around MediaPipe landmark lists.

    Both ``normalized`` and ``world`` hold 33 elements.  In simulation mode
    ``world`` is an empty list; consumers must guard against this.
    """
    normalized: list
    world: list = field(default_factory=list)
    is_valid: bool = True


@dataclass
class PostureReport:
    neck_flexion_deg: float = 0.0
    fhp_ratio: float = 0.0
    shoulder_asymmetry_deg: float = 0.0
    is_bad_posture: bool = False
    dominant_issue: str = "none"
    severity: AlertLevel = AlertLevel.OK
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Alert state machine
# ---------------------------------------------------------------------------

class AlertStateMachine:
    """Debounces raw per-frame alert levels to prevent flicker.

    A non-OK level only becomes the *current* level after ``bad_threshold``
    consecutive bad frames.  The level resets to OK only after ``good_reset``
    consecutive OK frames.  Within a bad streak the level can escalate
    (e.g., LEVEL1 → LEVEL3) but never de-escalate.
    """

    def __init__(self, bad_threshold: int = 10, good_reset: int = 5):
        self._bad_threshold = bad_threshold
        self._good_reset = good_reset
        self._consecutive_bad = 0
        self._consecutive_good = 0
        self._current_level = AlertLevel.OK

    def update(self, raw_severity: AlertLevel) -> AlertLevel:
        if raw_severity > AlertLevel.OK:
            self._consecutive_bad += 1
            self._consecutive_good = 0
            if self._consecutive_bad >= self._bad_threshold:
                # Escalate — never de-escalate within a bad streak
                self._current_level = max(self._current_level, raw_severity)
        else:
            self._consecutive_good += 1
            self._consecutive_bad = 0
            if self._consecutive_good >= self._good_reset:
                self._current_level = AlertLevel.OK

        return self._current_level

    def reset(self):
        self._consecutive_bad = 0
        self._consecutive_good = 0
        self._current_level = AlertLevel.OK


# ---------------------------------------------------------------------------
# PostureLogic
# ---------------------------------------------------------------------------

class PostureLogic:
    """Converts raw pose landmarks into a PostureReport.

    Parameters
    ----------
    thresholds:
        Alert level thresholds loaded from config.
    alert_state_config:
        Debounce parameters for AlertStateMachine.
    """

    def __init__(
        self,
        thresholds: AlertThresholds,
        alert_state_config: AlertStateConfig,
    ):
        self._thr = thresholds
        self._state_machine = AlertStateMachine(
            bad_threshold=alert_state_config.bad_frame_threshold,
            good_reset=alert_state_config.good_frame_reset,
        )

    def analyze(self, landmarks: PostureLandmarks) -> PostureReport:
        """Analyze landmarks and return a debounced PostureReport."""
        if not landmarks.is_valid or not landmarks.normalized:
            return PostureReport()

        neck = compute_neck_flexion(landmarks.normalized)
        fhp = compute_fhp_ratio(landmarks.normalized)
        asym = compute_shoulder_asymmetry(landmarks.normalized)

        raw_severity, dominant = self._classify(neck, fhp, asym)
        final_severity = self._state_machine.update(raw_severity)

        return PostureReport(
            neck_flexion_deg=round(neck, 2),
            fhp_ratio=round(fhp, 4),
            shoulder_asymmetry_deg=round(asym, 2),
            is_bad_posture=final_severity > AlertLevel.OK,
            dominant_issue=dominant,
            severity=final_severity,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self, neck: float, fhp: float, asym: float
    ) -> tuple[AlertLevel, str]:
        """Return (raw AlertLevel, dominant_issue string).

        Priority: neck_flexion > fhp > shoulder_asymmetry.
        """
        t = self._thr

        # Neck flexion (highest priority)
        if neck >= t.neck_flexion_severe:
            return AlertLevel.LEVEL3, "neck_flexion"
        if neck >= t.neck_flexion_warn:
            return AlertLevel.LEVEL2, "neck_flexion"   # moderate → motor

        # FHP
        if fhp >= t.fhp_severe:
            return AlertLevel.LEVEL3, "fhp"
        if fhp >= t.fhp_warn:
            return AlertLevel.LEVEL1, "fhp"

        # Shoulder asymmetry (lowest priority)
        if asym >= t.shoulder_asymmetry_severe:
            return AlertLevel.LEVEL2, "shoulder_asymmetry"
        if asym >= t.shoulder_asymmetry_warn:
            return AlertLevel.LEVEL1, "shoulder_asymmetry"

        return AlertLevel.OK, "none"
