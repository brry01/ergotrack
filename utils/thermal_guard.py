"""CPU thermal throttle guard.

Runs as a daemon thread, polling the Linux thermal sysfs node every
POLL_INTERVAL_S seconds.  When the temperature exceeds THROTTLE_TEMP_C,
it sets a threading.Event so the main loop can skip processing cycles and
let the CPU cool down.

On non-Linux platforms (e.g., Windows development) the sysfs path does not
exist; _read_temp() returns None silently and throttling never triggers.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

THERMAL_PATH = "/sys/class/thermal/thermal_zone0/temp"
POLL_INTERVAL_S = 10.0
THROTTLE_TEMP_C = 82.0   # RPi5 firmware throttles at 85°C; 82°C gives a 3°C buffer
PAUSE_AFTER_THROTTLE_S = 5.0


class ThermalGuard:
    """Thread-safe CPU temperature monitor.

    Usage::

        guard = ThermalGuard()
        guard.start()
        ...
        if guard.should_throttle:
            time.sleep(0.5)
            continue
        ...
        guard.stop()
    """

    def __init__(
        self,
        thermal_path: str = THERMAL_PATH,
        throttle_temp: float = THROTTLE_TEMP_C,
        poll_interval: float = POLL_INTERVAL_S,
        pause_duration: float = PAUSE_AFTER_THROTTLE_S,
    ):
        self._thermal_path = thermal_path
        self._throttle_temp = throttle_temp
        self._poll_interval = poll_interval
        self._pause_duration = pause_duration

        self._throttle_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="ThermalGuard",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def should_throttle(self) -> bool:
        """True when the CPU temperature is above the throttle threshold."""
        return self._throttle_event.is_set()

    def start(self):
        """Start the background polling thread."""
        self._thread.start()
        logger.debug("ThermalGuard started (threshold=%.1f°C).", self._throttle_temp)

    def stop(self):
        """Signal the polling thread to exit."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_temp(self) -> Optional[float]:
        """Return CPU temperature in °C, or None if unavailable."""
        try:
            with open(self._thermal_path) as f:
                raw = f.read().strip()
            return int(raw) / 1000.0
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _poll_loop(self):
        while not self._stop_event.wait(self._poll_interval):
            temp = self._read_temp()
            if temp is None:
                continue

            if temp > self._throttle_temp:
                if not self._throttle_event.is_set():
                    logger.warning(
                        "CPU temperature %.1f°C exceeds %.1f°C — throttling for %.0fs.",
                        temp, self._throttle_temp, self._pause_duration,
                    )
                self._throttle_event.set()
                time.sleep(self._pause_duration)
            else:
                if self._throttle_event.is_set():
                    logger.info("CPU temperature %.1f°C — throttle cleared.", temp)
                self._throttle_event.clear()
