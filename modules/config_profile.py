"""Configuration loader with hot-reload support.

Loads config/default.yaml and optionally merges a user override file.
All modules receive strongly-typed dataclass instances instead of raw dicts.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AlertThresholds:
    neck_flexion_warn: float = 25.0
    neck_flexion_severe: float = 40.0
    fhp_warn: float = 0.15
    fhp_severe: float = 0.30
    shoulder_asymmetry_warn: float = 5.0
    shoulder_asymmetry_severe: float = 10.0


@dataclass
class VisionConfig:
    fps: int = 15
    resolution: tuple = (640, 480)
    model_path: str = "models/pose_landmarker_full.task"


@dataclass
class HardwareConfig:
    # Buzzer (passive, PWM 2 kHz) — fires only at LEVEL3
    buzzer_pin: int = 18
    buzzer_active: bool = False  # False = passive buzzer (PWM); True = active (DC)
    buzzer_invert: bool = False  # True = active-low module (LOW=ON, HIGH=OFF)
    # Vibration motor (DC) — fires at LEVEL2
    motor_pin: int = 17
    motor_active: bool = True    # True = DC HIGH/LOW; False = PWM
    motor_invert: bool = False   # True = active-low module
    # OLED SSD1306 (I2C) — updates at LEVEL1+, cycles face/data every 5 s
    oled_address: int = 0x3C
    oled_width: int = 128
    oled_height: int = 64


@dataclass
class AlertStateConfig:
    bad_frame_threshold: int = 10
    good_frame_reset: int = 5


@dataclass
class LoggingConfig:
    level: str = "INFO"
    csv_output: bool = True
    csv_path: str = "ergotrack_log.csv"


# ---------------------------------------------------------------------------
# ConfigProfile
# ---------------------------------------------------------------------------

class ConfigProfile:
    """Loads YAML config and exposes typed dataclass attributes.

    Call ``maybe_reload()`` at the top of each processing cycle to pick up
    changes without restarting the process. The check is a single
    ``os.stat()`` call and is essentially free at 15 FPS.
    """

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        user_path: Optional[str] = None,
    ):
        self._config_path = config_path
        self._user_path = user_path
        self._mtime: float = 0.0
        self._user_mtime: float = 0.0

        # Public typed attributes (populated by _parse)
        self.thresholds = AlertThresholds()
        self.vision = VisionConfig()
        self.hardware = HardwareConfig()
        self.alert_state = AlertStateConfig()
        self.logging = LoggingConfig()

        self._load_and_parse()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_reload(self) -> bool:
        """Return True and reload if any config file was modified."""
        try:
            new_mtime = os.stat(self._config_path).st_mtime
        except OSError:
            return False

        new_user_mtime: float = 0.0
        if self._user_path:
            try:
                new_user_mtime = os.stat(self._user_path).st_mtime
            except OSError:
                pass

        if new_mtime != self._mtime or new_user_mtime != self._user_mtime:
            logger.info("Config changed — reloading.")
            self._load_and_parse()
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_parse(self):
        raw = self._read_yaml(self._config_path)
        if self._user_path:
            user_raw = self._read_yaml(self._user_path)
            raw = _deep_merge(raw, user_raw)
        self._parse(raw)

        try:
            self._mtime = os.stat(self._config_path).st_mtime
        except OSError:
            pass
        if self._user_path:
            try:
                self._user_mtime = os.stat(self._user_path).st_mtime
            except OSError:
                pass

    def _read_yaml(self, path: str) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data or {}
        except FileNotFoundError:
            logger.warning("Config file not found: %s — using defaults.", path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("YAML parse error in %s: %s — keeping previous config.", path, exc)
            return {}

    def _parse(self, raw: dict):
        thr = raw.get("thresholds", {})
        self.thresholds = AlertThresholds(
            neck_flexion_warn=float(thr.get("neck_flexion_warn", 25.0)),
            neck_flexion_severe=float(thr.get("neck_flexion_severe", 40.0)),
            fhp_warn=float(thr.get("fhp_warn", 0.15)),
            fhp_severe=float(thr.get("fhp_severe", 0.30)),
            shoulder_asymmetry_warn=float(thr.get("shoulder_asymmetry_warn", 5.0)),
            shoulder_asymmetry_severe=float(thr.get("shoulder_asymmetry_severe", 10.0)),
        )

        vis = raw.get("vision", {})
        res_raw = vis.get("resolution", [640, 480])
        self.vision = VisionConfig(
            fps=int(vis.get("fps", 15)),
            resolution=tuple(res_raw),
            model_path=str(vis.get("model_path", "models/pose_landmarker_full.task")),
        )

        hw = raw.get("hardware", {})
        oled_addr_raw = hw.get("oled_address", "0x3C")
        self.hardware = HardwareConfig(
            buzzer_pin=int(hw.get("buzzer_pin", 18)),
            buzzer_active=bool(hw.get("buzzer_active", True)),
            buzzer_invert=bool(hw.get("buzzer_invert", False)),
            motor_pin=int(hw.get("motor_pin", 17)),
            motor_active=bool(hw.get("motor_active", True)),
            motor_invert=bool(hw.get("motor_invert", False)),
            oled_address=int(str(oled_addr_raw), 16),
            oled_width=int(hw.get("oled_width", 128)),
            oled_height=int(hw.get("oled_height", 64)),
        )

        als = raw.get("alert_state", {})
        self.alert_state = AlertStateConfig(
            bad_frame_threshold=int(als.get("bad_frame_threshold", 10)),
            good_frame_reset=int(als.get("good_frame_reset", 5)),
        )

        log = raw.get("logging", {})
        self.logging = LoggingConfig(
            level=str(log.get("level", "INFO")).upper(),
            csv_output=bool(log.get("csv_output", True)),
            csv_path=str(log.get("csv_path", "ergotrack_log.csv")),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result
