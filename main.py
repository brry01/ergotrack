"""ErgoTrack — entry point.

Usage:
  python main.py                          # headless, auto-simulate on non-RPi
  python main.py --gui                    # GUI dashboard
  python main.py --simulate               # force simulation mode
  python main.py --gui --simulate --scenario severe_neck
  python main.py --config myconfig.yaml   # custom config file
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# ARM64 / XNNPACK fix — must be set before any mediapipe/TFLite import.
#
# On RPi5 (Cortex-A76) the kernel returns EINVAL for prctl(PR_SVE_GET_VL)
# because the core has no SVE unit.  XNNPACK's multi-threaded SVE probe
# then produces NaN outputs that corrupt the pose-landmarker's remap() call,
# triggering a fatal assertion in MediaPipe's bundled OpenCV 4.5.5.
# Forcing single-threaded mode avoids the multi-threaded SVE code path.
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("TFLITE_NUM_THREADS", "1")

import argparse
import csv
import logging
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from modules.config_profile import ConfigProfile
from modules.hardware_controller import HardwareController
from modules.posture_logic import AlertLevel, PostureLogic, PostureReport
from utils.thermal_guard import ThermalGuard
from utils.terminal_display import TerminalMonitor


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def is_raspberry_pi() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            return "Raspberry Pi" in f.read()
    except (FileNotFoundError, OSError):
        return False


def should_simulate(args: argparse.Namespace) -> bool:
    return args.simulate or not is_raspberry_pi()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ErgoTrack — real-time ergonomic postural monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true",
                        help="Launch the ErgoDashboard GUI")
    parser.add_argument("--simulate", action="store_true",
                        help="Use camera simulator (auto-enabled on non-RPi)")
    parser.add_argument("--config", default="config/default.yaml",
                        metavar="PATH", help="Path to YAML config (default: config/default.yaml)")
    parser.add_argument("--user-config", default=None, metavar="PATH",
                        help="Optional user override config file")
    parser.add_argument("--scenario", default="cycling",
                        choices=["good", "mild_fhp", "severe_neck", "shoulder_tilt", "cycling"],
                        help="Simulator scenario (default: cycling)")
    parser.add_argument("--monitor", action="store_true",
                        help="Print live coloured metrics to the terminal (headless mode only)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

_CSV_HEADER = [
    "timestamp", "neck_flexion_deg", "fhp_ratio",
    "shoulder_asymmetry_deg", "severity", "dominant_issue",
]


def init_csv(path: str):
    """Open CSV file and write header if new. Returns (file_obj, csv.writer)."""
    is_new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(_CSV_HEADER)
    return f, writer


def write_csv_row(writer, report: PostureReport):
    writer.writerow([
        time.strftime("%Y-%m-%dT%H:%M:%S"),
        f"{report.neck_flexion_deg:.2f}",
        f"{report.fhp_ratio:.4f}",
        f"{report.shoulder_asymmetry_deg:.2f}",
        report.severity.name,
        report.dominant_issue,
    ])


# ---------------------------------------------------------------------------
# Headless loop
# ---------------------------------------------------------------------------

def run_headless(vm, pl: PostureLogic, hw: HardwareController,
                 config: ConfigProfile, thermal: ThermalGuard,
                 monitor=None):
    """Main headless loop.

    Parameters
    ----------
    monitor:
        Optional ``TerminalMonitor`` instance.  When provided, each cycle
        prints a coloured metric line instead of the plain INFO log entry.
    """
    logger = logging.getLogger("headless")
    frame_interval = 1.0 / config.vision.fps

    csv_file, csv_writer = (None, None)
    if config.logging.csv_output:
        csv_file, csv_writer = init_csv(config.logging.csv_path)
        logger.info("CSV logging → %s", config.logging.csv_path)

    if monitor:
        logger.info("ErgoTrack monitor mode — Ctrl+C para salir.")
    else:
        logger.info("ErgoTrack headless mode started. Press Ctrl+C to stop.")

    _last_throttle_log: float = 0.0   # timestamp of last throttle warning

    try:
        while True:
            cycle_start = time.perf_counter()

            # Hot-reload config (cheap mtime check)
            config.maybe_reload()

            # Thermal throttle — log at most once every 30 s to avoid spam
            if thermal.should_throttle:
                now = time.perf_counter()
                if now - _last_throttle_log >= 30.0:
                    logger.warning("ThermalGuard: CPU caliente — esperando.")
                    _last_throttle_log = now
                time.sleep(2.0)
                continue

            landmarks = vm.capture_and_detect()
            report = pl.analyze(landmarks)
            hw.trigger_alert(report.severity)

            if monitor:
                # Coloured terminal display — replaces per-frame log entries
                monitor.update(report)
            elif report.is_bad_posture:
                logger.info(
                    "BAD POSTURE | %s | neck=%.1f° fhp=%.3f asym=%.1f°",
                    report.severity.name,
                    report.neck_flexion_deg,
                    report.fhp_ratio,
                    report.shoulder_asymmetry_deg,
                )
            else:
                logger.debug(
                    "OK | neck=%.1f° fhp=%.3f asym=%.1f°",
                    report.neck_flexion_deg, report.fhp_ratio, report.shoulder_asymmetry_deg,
                )

            if csv_writer:
                write_csv_row(csv_writer, report)

            elapsed = time.perf_counter() - cycle_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > frame_interval * 1.5:
                logger.warning("Cycle overrun: %.1f ms (budget %.0f ms)",
                               elapsed * 1000, frame_interval * 1000)

    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        if csv_file:
            csv_file.close()


# ---------------------------------------------------------------------------
# Simulation adapter for headless mode
# ---------------------------------------------------------------------------

class _HeadlessSimAdapter:
    """Wraps CameraSimulator to provide capture_and_detect() for headless loop."""

    def __init__(self, simulator, posture_landmarks_cls):
        self._sim = simulator
        self._PostureLandmarks = posture_landmarks_cls

    def capture_and_detect(self):
        lms = self._sim.get_landmarks()
        return self._PostureLandmarks(normalized=lms, world=[], is_valid=True)

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    simulate = should_simulate(args)

    # 1. Load config
    config = ConfigProfile(args.config, args.user_config)
    setup_logging(config.logging.level)
    logger = logging.getLogger("main")

    logger.info(
        "ErgoTrack starting | mode=%s | simulate=%s",
        "gui" if args.gui else "headless",
        simulate,
    )

    # 2. Thermal guard (daemon thread — safe to start even on Windows)
    thermal = ThermalGuard()
    thermal.start()

    # 3. Vision source
    if simulate:
        from utils.camera_simulator import CameraSimulator
        from modules.posture_logic import PostureLandmarks
        sim = CameraSimulator(scenario=args.scenario)
        logger.info("Simulation mode: scenario=%s", args.scenario)
    else:
        from modules.vision_manager import VisionManager
        try:
            vm_real = VisionManager(config=config.vision)
        except (FileNotFoundError, RuntimeError, ImportError) as exc:
            logger.error("VisionManager init failed: %s", exc)
            logger.error("Tip: run 'python scripts/download_models.py' to fetch the model, "
                         "or use --simulate for development without a camera.")
            sys.exit(1)

    # 4. PostureLogic
    pl = PostureLogic(config.thresholds, config.alert_state)

    # 5. Hardware controller
    hw = HardwareController(config.hardware, sim_mode=simulate)

    # 6. Launch mode
    try:
        if args.gui:
            from gui.ergo_dashboard import ErgoDashboard, SimVisionSource
            from modules.posture_logic import PostureLandmarks

            if simulate:
                vision_source = SimVisionSource(sim, PostureLandmarks)
            else:
                # Wrap VisionManager to also expose last_landmarks
                vision_source = _VisionManagerGUIAdapter(vm_real)

            dashboard = ErgoDashboard(vision_source, pl, hw, config)
            dashboard.run()

        else:
            if simulate:
                from modules.posture_logic import PostureLandmarks
                vm = _HeadlessSimAdapter(sim, PostureLandmarks)
            else:
                vm = vm_real

            monitor = TerminalMonitor(config.thresholds) if args.monitor else None
            run_headless(vm, pl, hw, config, thermal, monitor=monitor)

    finally:
        thermal.stop()
        hw.cleanup()
        if not simulate and not args.gui:
            try:
                vm_real.release()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GUI adapter for real VisionManager (exposes last_landmarks)
# ---------------------------------------------------------------------------

class _VisionManagerGUIAdapter:
    """Thin wrapper that caches last_landmarks for overlay drawing."""

    def __init__(self, vision_manager):
        self._vm = vision_manager
        self.last_landmarks = None

    def capture_with_frame(self):
        frame, lms = self._vm.capture_with_frame()
        if lms.is_valid:
            self.last_landmarks = lms.normalized
        return frame, lms

    def release(self):
        self._vm.release()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
