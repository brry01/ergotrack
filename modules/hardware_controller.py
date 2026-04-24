"""Hardware output controller: GPIO buzzer/motor + OLED display.

Alert routing
─────────────
  LEVEL1  → motor vibrates  (short pulse)
  LEVEL2  → motor vibrates  (double pulse)
  LEVEL3  → motor vibrates  (triple pulse) + buzzer sounds

Auto-detects available hardware at import time.  If GPIO or OLED libraries
are absent (e.g., on a development PC), the controller transparently enters
Simulation Mode and prints alerts to stdout instead.

Real hardware wiring (RPi5):
  - Vibration motor : GPIO motor_pin  (BCM 17)  ← active at LEVEL1+
  - Buzzer (KY-012) : GPIO buzzer_pin (BCM 18)  ← active at LEVEL3 only
  - OLED            : I2C SSD1306 128×64 at 0x3C (I2C bus 1)

Buzzer/motor types (buzzer_active / motor_active in config/default.yaml):
  *_active: true  — active device driven with DC HIGH/LOW.
  *_active: false — passive buzzer driven with 2 kHz PWM.

*_invert: true    — active-low module (LOW=ON, HIGH=OFF).

GPIO library priority (RPi OS Bookworm):
  1. RPi.GPIO  — works on RPi5 via the lgpio backend in RPi OS
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
# Alert patterns  (duration_ms, pulse_count)
# Motor fires on every alert level; buzzer fires only at LEVEL3.
# ---------------------------------------------------------------------------
_BUZZER_FREQ_HZ = 2000

_MOTOR_PATTERNS: dict[AlertLevel, tuple[int, int]] = {
    AlertLevel.LEVEL1: (150, 1),   # 1 short vibration
    AlertLevel.LEVEL2: (150, 2),   # 2 short vibrations
    AlertLevel.LEVEL3: (300, 3),   # 3 long vibrations
}

_BUZZER_PATTERNS: dict[AlertLevel, tuple[int, int]] = {
    # Only LEVEL3 triggers the buzzer
    AlertLevel.LEVEL3: (300, 3),   # 3 long beeps
}

# OLED is updated at most this often (seconds).  I2C transfers at 100 kHz
# take ~100 ms for a 128×64 frame — too slow to call every inference cycle.
_OLED_MIN_INTERVAL_S = 2.0


class HardwareController:
    """Manages physical alert outputs with automatic simulation fallback.

    Alert routing
    ─────────────
      LEVEL1 → motor only
      LEVEL2 → motor only
      LEVEL3 → motor + buzzer

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
        self._pwm_buzzer = None   # passive buzzer PWM (buzzer_active=False)
        self._pwm_motor  = None   # passive motor  PWM (motor_active=False)
        self._oled = None
        self._motor_lock  = threading.Lock()
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
        """Fire the appropriate alert outputs for the given AlertLevel.

        Routing:
          LEVEL1/2 → motor only
          LEVEL3   → motor + buzzer
        """
        if self._sim_mode:
            if level != AlertLevel.OK:
                print(f"[ALERT] Level {level.name}", flush=True)
            return

        if level == AlertLevel.OK:
            return

        # Motor — fires for every non-OK level
        motor_pattern = _MOTOR_PATTERNS.get(level)
        if motor_pattern:
            duration_ms, count = motor_pattern
            threading.Thread(
                target=self._activate_motor,
                args=(duration_ms, count),
                daemon=True,
            ).start()

        # Buzzer — fires ONLY at LEVEL3
        buzzer_pattern = _BUZZER_PATTERNS.get(level)
        if buzzer_pattern:
            duration_ms, count = buzzer_pattern
            threading.Thread(
                target=self._beep_buzzer,
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
            if self._pwm_buzzer is not None:
                self._pwm_buzzer.stop()
                self._pwm_buzzer = None
            if self._pwm_motor is not None:
                self._pwm_motor.stop()
                self._pwm_motor = None
            if _HAS_GPIO and _GPIO is not None:
                cfg = self._config
                # Drive both pins to idle (off) before cleanup
                _GPIO.output(cfg.buzzer_pin,
                             _GPIO.HIGH if cfg.buzzer_invert else _GPIO.LOW)
                _GPIO.output(cfg.motor_pin,
                             _GPIO.HIGH if cfg.motor_invert else _GPIO.LOW)
                _GPIO.cleanup()
        except Exception:
            logger.exception("GPIO cleanup error.")

    # ------------------------------------------------------------------
    # Internal — GPIO init
    # ------------------------------------------------------------------

    def _init_gpio(self):
        if not _HAS_GPIO or _GPIO is None:
            return
        try:
            _GPIO.setmode(_GPIO.BCM)
            _GPIO.setwarnings(False)

            cfg = self._config

            # Buzzer pin
            buzzer_idle = _GPIO.HIGH if cfg.buzzer_invert else _GPIO.LOW
            _GPIO.setup(cfg.buzzer_pin, _GPIO.OUT, initial=buzzer_idle)
            if not cfg.buzzer_active:
                self._pwm_buzzer = _GPIO.PWM(cfg.buzzer_pin, _BUZZER_FREQ_HZ)

            # Motor pin
            motor_idle = _GPIO.HIGH if cfg.motor_invert else _GPIO.LOW
            _GPIO.setup(cfg.motor_pin, _GPIO.OUT, initial=motor_idle)
            if not cfg.motor_active:
                self._pwm_motor = _GPIO.PWM(cfg.motor_pin, _BUZZER_FREQ_HZ)

            logger.info(
                "GPIO initialised — Buzzer BCM%d (%s%s), Motor BCM%d (%s%s).",
                cfg.buzzer_pin,
                "activo" if cfg.buzzer_active else f"pasivo {_BUZZER_FREQ_HZ} Hz",
                " inv" if cfg.buzzer_invert else "",
                cfg.motor_pin,
                "activo" if cfg.motor_active else f"pasivo {_BUZZER_FREQ_HZ} Hz",
                " inv" if cfg.motor_invert else "",
            )
        except Exception:
            logger.exception("GPIO init failed — outputs disabled.")
            self._pwm_buzzer = None
            self._pwm_motor  = None

    # ------------------------------------------------------------------
    # Internal — motor output
    # ------------------------------------------------------------------

    def _activate_motor(self, duration_ms: int, count: int):
        """Emit a vibration motor pattern in a background thread."""
        cfg = self._config
        on  = _GPIO.LOW  if cfg.motor_invert else _GPIO.HIGH
        off = _GPIO.HIGH if cfg.motor_invert else _GPIO.LOW

        with self._motor_lock:
            dur = duration_ms / 1000.0
            for i in range(count):
                try:
                    if cfg.motor_active:
                        _GPIO.output(cfg.motor_pin, on)
                        time.sleep(dur)
                        _GPIO.output(cfg.motor_pin, off)
                    else:
                        if self._pwm_motor is None:
                            return
                        self._pwm_motor.start(50)
                        time.sleep(dur)
                        self._pwm_motor.stop()
                    if i < count - 1:
                        time.sleep(0.08)
                except Exception:
                    logger.exception("Motor error.")
                    break

    # ------------------------------------------------------------------
    # Internal — buzzer output
    # ------------------------------------------------------------------

    def _beep_buzzer(self, duration_ms: int, count: int):
        """Emit a buzzer alert pattern in a background thread (LEVEL3 only)."""
        cfg = self._config
        on  = _GPIO.LOW  if cfg.buzzer_invert else _GPIO.HIGH
        off = _GPIO.HIGH if cfg.buzzer_invert else _GPIO.LOW

        with self._buzzer_lock:
            dur = duration_ms / 1000.0
            for i in range(count):
                try:
                    if cfg.buzzer_active:
                        _GPIO.output(cfg.buzzer_pin, on)
                        time.sleep(dur)
                        _GPIO.output(cfg.buzzer_pin, off)
                    else:
                        if self._pwm_buzzer is None:
                            return
                        self._pwm_buzzer.start(50)
                        time.sleep(dur)
                        self._pwm_buzzer.stop()
                    if i < count - 1:
                        time.sleep(0.08)
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
