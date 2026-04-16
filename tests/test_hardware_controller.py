"""Unit tests for modules/hardware_controller.py.

All tests run in simulation mode (no GPIO or OLED required) and verify
the console-output contract and safe operation of the controller.
"""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.hardware_controller import HardwareController, _HAS_GPIO, _HAS_OLED
from modules.posture_logic import AlertLevel, PostureReport
from modules.config_profile import HardwareConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hw(sim_mode: bool = True) -> HardwareController:
    return HardwareController(HardwareConfig(), sim_mode=sim_mode)


def make_report(severity: AlertLevel = AlertLevel.LEVEL3) -> PostureReport:
    return PostureReport(
        neck_flexion_deg=45.0,
        fhp_ratio=0.32,
        shoulder_asymmetry_deg=8.0,
        is_bad_posture=True,
        dominant_issue="neck_flexion",
        severity=severity,
    )


# ---------------------------------------------------------------------------
# Simulation mode tests
# ---------------------------------------------------------------------------

class TestSimulationMode:

    def test_trigger_level1_prints_alert(self, capsys):
        hw = make_hw()
        hw.trigger_alert(AlertLevel.LEVEL1)
        captured = capsys.readouterr()
        assert "LEVEL1" in captured.out

    def test_trigger_level2_prints_alert(self, capsys):
        hw = make_hw()
        hw.trigger_alert(AlertLevel.LEVEL2)
        captured = capsys.readouterr()
        assert "LEVEL2" in captured.out

    def test_trigger_level3_prints_alert(self, capsys):
        hw = make_hw()
        hw.trigger_alert(AlertLevel.LEVEL3)
        captured = capsys.readouterr()
        assert "LEVEL3" in captured.out

    def test_trigger_ok_produces_no_output(self, capsys):
        hw = make_hw()
        hw.trigger_alert(AlertLevel.OK)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_cleanup_does_not_raise(self):
        hw = make_hw()
        hw.cleanup()   # must not raise

    def test_update_oled_does_not_raise(self):
        hw = make_hw()
        report = make_report()
        hw.update_oled(report)   # no-op in sim mode, must not raise

    def test_sim_mode_flag_is_set(self):
        hw = make_hw(sim_mode=True)
        assert hw._sim_mode is True


# ---------------------------------------------------------------------------
# Auto-simulation detection
# ---------------------------------------------------------------------------

class TestAutoSimDetection:

    def test_auto_sim_on_pc(self):
        # On a PC without RPi.GPIO or adafruit_ssd1306, sim_mode should auto-enable
        hw = HardwareController(HardwareConfig())   # no explicit sim_mode
        if not (_HAS_GPIO or _HAS_OLED):
            assert hw._sim_mode is True
        else:
            # On actual RPi with hardware, this may be False — accept both
            assert isinstance(hw._sim_mode, bool)

    def test_explicit_sim_mode_overrides_hardware(self):
        hw = HardwareController(HardwareConfig(), sim_mode=True)
        assert hw._sim_mode is True


# ---------------------------------------------------------------------------
# Multiple alerts
# ---------------------------------------------------------------------------

class TestMultipleAlerts:

    def test_repeated_alerts_do_not_crash(self, capsys):
        hw = make_hw()
        for level in [AlertLevel.LEVEL1, AlertLevel.LEVEL2, AlertLevel.LEVEL3, AlertLevel.OK]:
            hw.trigger_alert(level)
        hw.cleanup()

    def test_ok_after_alert_produces_no_output(self, capsys):
        hw = make_hw()
        hw.trigger_alert(AlertLevel.LEVEL2)
        capsys.readouterr()   # consume previous output
        hw.trigger_alert(AlertLevel.OK)
        captured = capsys.readouterr()
        assert captured.out == ""
