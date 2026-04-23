"""Hardware output controller: GPIO buzzer/motor + OLED display.

Auto-detects available hardware at import time.  If GPIO or OLED libraries
are absent (e.g., on a development PC), the controller transparently enters
Simulation Mode and prints alerts to stdout instead.

Real hardware wiring (RPi5):
  - Buzzer / vibration motor: GPIO buzzer_pin (BCM 18)
  - OLED: I2C SSD1306 128×64 at address 0x3C (I2C bus 1)

Buzzer types (set buzzer_active in config/default.yaml):
  buzzer_active: true  — active buzzer / vibration motor: driven with DC
                         HIGH/LOW.  The device has an internal oscillator.
  buzzer_active: false — passive buzzer: driven with 2 kHz PWM.

GPIO library priority (RPi OS Bookworm):
  1. RPi.GPIO  — works on RPi5 via the lgpio backend included in RPi OS
  2. rpi-lgpio — drop-in replacement if RPi.GPIO is absent
  Simulation mode activates automatically when neither is available (PC).

OLED library: luma.oled (pip install luma.oled)
"""
from __future__ import annotations

import logging
import threading
import time

from modules.config_profile import HardwareConfig
from modules.posture_logic import AlertLevel, PostureReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPIO detection — try RPi.GPIO first, then rpi-lgpio (RPi5 fallback)
# ---------------------------------------------------------------------------

_HAS_GPIO = False
_GPIO = None

try:
    # RPi.GPIO works on RPi5 when either of these is installed:
    #   sudo apt install python3-rpi.gpio   (RPi OS system Python only)
    #   pip install rpi-lgpio               (venv — also installs as RPi.GPIO)
    import RPi.GPIO as _GPIO        # type: ignore
    _HAS_GPIO = True
    logger.debug("GPIO: RPi.GPIO available")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# OLED detection — luma.oled (no Blinka layer needed)
# ---------------------------------------------------------------------------

_HAS_OLED = False
_luma_i2c = None
_luma_ssd1306 = None

try:
    from luma.core.interface.serial import i2c as _luma_i2c      # type: ignore
    from luma.oled.device import ssd1306 as _luma_ssd1306        # type: ignore
    _HAS_OLED = True
    logger.debug("OLED: luma.oled available")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# PIL (for OLED rendering)
# ---------------------------------------------------------------------------

_HAS_PIL = False
try:
    from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw, ImageFont as _PIL_Font
    _HAS_PIL = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Buzzer alert patterns  (duration_ms, beep_count)
# ---------------------------------------------------------------------------
_BUZZER_FREQ_HZ = 2000
_BUZZER_PATTERNS = {
    AlertLevel.LEVEL1: (100, 1),   # 1 short beep
    AlertLevel.LEVEL2: (100, 2),   # 2 short beeps
    AlertLevel.LEVEL3: (300, 3),   # 3 long beeps
}

# OLED is updated at most this often (seconds).  I2C transfers at 100 kHz
# take ~100 ms for a 128×64 frame — too slow to call every inference cycle.
_OLED_MIN_INTERVAL_S = 2.0


class HardwareController:
    """Manages physical alert outputs with automatic simulation fallback.

    Parameters
    ----------
    config:
        Hardware configuration (pin numbers, OLED address, etc.).
    sim_mode:
        Force simulation mode regardless of detected hardware.
        Defaults to True automatically when neither GPIO nor OLED is found.
    """

    def __init__(self, config: HardwareConfig, sim_mode: bool = False):
        self._config = config
        self._sim_mode: bool = sim_mode or not (_HAS_GPIO or _HAS_OLED)
        self._pwm = None          # only used when buzzer_active=False (passive buzzer)
        self._oled = None
        self._buzzer_lock = threading.Lock()
        self._last_oled_t: float = 0.0

        if self._sim_mode:
            logger.info("HardwareController: simulation mode active.")
        else:
            self._init_gpio()
            self._init_oled()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trigger_alert(self, level: AlertLevel):
        """Fire the appropriate alert output for the given AlertLevel."""
        if self._sim_mode:
            if level != AlertLevel.OK:
                print(f"[ALERT] Level {level.name}", flush=True)
            return

        if level == AlertLevel.OK:
            return

        pattern = _BUZZER_PATTERNS.get(level)
        if pattern:
            duration_ms, count = pattern
            threading.Thread(
                target=self._beep,
                args=(duration_ms, count),
                daemon=True,
            ).start()

    def update_oled(self, report: PostureReport):
        """Render a PostureReport summary to the OLED display.

        Internally throttled to at most once every ``_OLED_MIN_INTERVAL_S``
        seconds — I2C transfers take ~100 ms and must not block the main loop
        on every inference cycle.  Safe to call on every frame.
        """
        if self._sim_mode or self._oled is None or not _HAS_PIL:
            return
        now = time.monotonic()
        if now - self._last_oled_t < _OLED_MIN_INTERVAL_S:
            return
        self._last_oled_t = now
        try:
            self._render_oled(report)
        except Exception:
            logger.exception("OLED render error.")

    def cleanup(self):
        """Release all hardware resources. Call before process exit."""
        if self._sim_mode:
            return
        try:
            if self._pwm is not None:
                self._pwm.stop()
                self._pwm = None
            if _HAS_GPIO and _GPIO is not None:
                _GPIO.output(self._config.buzzer_pin, _GPIO.LOW)   # ensure off
                _GPIO.cleanup()
        except Exception:
            logger.exception("GPIO cleanup error.")

    # ------------------------------------------------------------------
    # Internal — GPIO
    # ------------------------------------------------------------------

    def _init_gpio(self):
        if not _HAS_GPIO or _GPIO is None:
            return
        try:
            _GPIO.setmode(_GPIO.BCM)
            _GPIO.setwarnings(False)
            _GPIO.setup(self._config.buzzer_pin, _GPIO.OUT, initial=_GPIO.LOW)

            if not self._config.buzzer_active:
                # Passive buzzer — needs PWM to produce a tone
                self._pwm = _GPIO.PWM(self._config.buzzer_pin, _BUZZER_FREQ_HZ)

            btype = "activo" if self._config.buzzer_active else f"pasivo {_BUZZER_FREQ_HZ} Hz"
            logger.info("GPIO initialised — Buzzer BCM%d (%s).",
                        self._config.buzzer_pin, btype)
        except Exception:
            logger.exception("GPIO init failed — outputs disabled.")
            self._pwm = None

    def _beep(self, duration_ms: int, count: int):
        """Emit a buzzer/motor pattern in a background thread.

        Active buzzer / vibration motor (buzzer_active=True):
            Driven with GPIO HIGH/LOW — the device generates its own tone.
        Passive buzzer (buzzer_active=False):
            Driven with 2 kHz PWM — the GPIO signal creates the tone.
        """
        with self._buzzer_lock:
            dur = duration_ms / 1000.0
            for i in range(count):
                try:
                    if self._config.buzzer_active:
                        # Active buzzer / motor: simple DC on/off
                        _GPIO.output(self._config.buzzer_pin, _GPIO.HIGH)
                        time.sleep(dur)
                        _GPIO.output(self._config.buzzer_pin, _GPIO.LOW)
                    else:
                        # Passive buzzer: PWM tone
                        if self._pwm is None:
                            return
                        self._pwm.start(50)
                        time.sleep(dur)
                        self._pwm.stop()
                    if i < count - 1:
                        time.sleep(0.08)   # gap between beeps
                except Exception:
                    logger.exception("Buzzer error.")
                    break

    # ------------------------------------------------------------------
    # Internal — OLED  (luma.oled + PIL)
    # ------------------------------------------------------------------

    def _init_oled(self):
        if not (_HAS_OLED and _luma_i2c and _luma_ssd1306):
            if not _HAS_OLED:
                logger.warning(
                    "luma.oled not installed — OLED disabled. "
                    "Fix: pip install luma.oled"
                )
            return
        try:
            serial = _luma_i2c(port=1, address=self._config.oled_address)
            self._oled = _luma_ssd1306(
                serial,
                width=self._config.oled_width,
                height=self._config.oled_height,
            )
            logger.info("OLED initialised (SSD1306 128×64 at I2C 0x%02X).",
                        self._config.oled_address)
        except Exception as exc:
            exc_name = type(exc).__name__
            exc_str  = str(exc)
            if "DeviceNotFoundError" in exc_name or "Remote I/O" in exc_str:
                logger.warning(
                    "OLED not found at 0x%02X on I2C bus 1. "
                    "Check wiring and run: python scripts/test_oled.py",
                    self._config.oled_address,
                )
            elif "Permission" in exc_str or "Access" in exc_str:
                logger.error(
                    "OLED I2C permission denied. "
                    "Fix: sudo usermod -aG i2c %s  (then re-login)",
                    __import__("os").getenv("USER", "$USER"),
                )
            else:
                logger.error(
                    "OLED init failed: %s: %s — run scripts/test_oled.py "
                    "for full diagnosis.",
                    exc_name, exc_str,
                )
            self._oled = None

    def _render_oled(self, report: PostureReport):
        """Draw PostureReport data onto the OLED using luma.oled canvas API."""
        if not (_HAS_PIL and self._oled is not None):
            return

        from luma.core.render import canvas   # type: ignore  # local import — only on RPi

        status = report.severity.name
        lines = [
            f"ErgoTrack [{status}]",
            f"Neck: {report.neck_flexion_deg:.1f}deg",
            f"FHP:  {report.fhp_ratio:.2f}",
            f"Asym: {report.shoulder_asymmetry_deg:.1f}deg",
        ]

        try:
            font = _PIL_Font.load_default()
        except Exception:
            font = None

        with canvas(self._oled) as draw:
            for i, line in enumerate(lines):
                draw.text((0, i * 14), line, font=font, fill="white")
