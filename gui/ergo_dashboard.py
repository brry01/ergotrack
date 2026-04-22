"""ErgoDashboard — real-time postural monitoring GUI.

Layout (3-panel dark theme):
  ┌──────────────┬─────────────────┬──────────────────┐
  │  Video Feed  │   KPI Gauges    │  Alert History   │
  │  + overlay   │  Neck: 12°      │  14:23 WARN      │
  │  (320×240)   │  FHP: 0.08      │  14:30 SEVERE    │
  │              │  Asym: 2.1°     │  Current: LEVEL2  │
  └──────────────┴─────────────────┴──────────────────┘

Prefers CustomTkinter for a polished dark-mode look; falls back to plain
tkinter if CustomTkinter is not installed.

Update loop: Tkinter after() at 66 ms ≈ 15 FPS.
Frame display: downscaled to 320×240 for performance; landmark overlay
is drawn on the original resolution before downscaling.
"""
from __future__ import annotations

import collections
import datetime
import logging
import math
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CustomTkinter / tkinter compatibility shim
# ---------------------------------------------------------------------------
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    _HAS_CTK = True
    _App = ctk.CTk
    _Frame = ctk.CTkFrame
    _Label = ctk.CTkLabel
    _ScrollableFrame = ctk.CTkScrollableFrame
    _FG_COLOR = "#1c1c1e"
    _PANEL_COLOR = "#2c2c2e"
    _TEXT_COLOR = "#f2f2f7"
except ImportError:
    import tkinter as tk
    _HAS_CTK = False
    _App = tk.Tk
    _Frame = tk.Frame
    _Label = tk.Label
    _ScrollableFrame = None
    _FG_COLOR = "#1c1c1e"
    _PANEL_COLOR = "#2c2c2e"
    _TEXT_COLOR = "#f2f2f7"
    logger.info("CustomTkinter not found — using plain tkinter fallback.")

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

import tkinter as tk   # always imported for Canvas, Text, etc.

# ---------------------------------------------------------------------------
# Alert level colour mapping
# ---------------------------------------------------------------------------
_SEVERITY_COLORS = {
    0: "#34c759",   # OK    — green
    1: "#ffd60a",   # L1    — yellow
    2: "#ff9500",   # L2    — orange
    3: "#ff3b30",   # L3    — red
}

# MediaPipe skeleton connections for overlay drawing
_POSE_CONNECTIONS = [
    (7, 11), (8, 12),
    (11, 12),
    (11, 13), (12, 14),
    (13, 15), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (24, 26),
]

_DISPLAY_W, _DISPLAY_H = 320, 240   # video panel display resolution


class ErgoDashboard:
    """Tkinter-based real-time ergonomics dashboard.

    Parameters
    ----------
    vision_source:
        An object with a ``capture_with_frame()`` method returning
        ``(np.ndarray | None, PostureLandmarks)``.  Can be VisionManager
        or CameraSimulator-wrapped by the caller.
    posture_logic:
        PostureLogic instance.
    hardware_controller:
        HardwareController instance.
    config:
        ConfigProfile instance (used for future hot-reload awareness).
    """

    def __init__(self, vision_source, posture_logic, hardware_controller, config):
        self._vm = vision_source
        self._pl = posture_logic
        self._hw = hardware_controller
        self._config = config

        self._running = False
        self._last_photo: Optional[object] = None   # ImageTk.PhotoImage ref
        self._alert_history: Deque[Tuple[str, int]] = collections.deque(maxlen=50)

        # --- render-skip caches -------------------------------------------
        # Video: skip PIL/PhotoImage work when the inference thread hasn't
        #        produced a new frame yet (same object id).
        self._last_frame_id: int = -1

        # KPIs: only push to tkinter widgets when values change by ≥ 0.5°
        self._last_severity: int = -1
        self._last_neck:  float = float("nan")
        self._last_fhp:   float = float("nan")
        self._last_asym:  float = float("nan")

        # History: only rebuild the Text widget when a new alert is appended.
        self._history_version: int = 0
        self._rendered_history_version: int = -1
        # ------------------------------------------------------------------

        self._root = _App()
        self._root.title("ErgoTrack — Postural Monitor")
        self._root.configure(bg=_FG_COLOR)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Start the GUI update loop (~7 FPS) and enter the Tkinter main loop.

        7 FPS (150 ms interval) is plenty for a posture monitor: the alert
        state machine requires 10 consecutive bad frames and the camera
        itself runs at 5 FPS, so the GUI never needs to exceed that rate.
        """
        self._root.geometry("1200x680")
        self._root.minsize(900, 500)
        self._running = True
        self._root.after(150, self._update_loop)
        self._root.mainloop()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self):
        self._root.columnconfigure(0, weight=2)
        self._root.columnconfigure(1, weight=1)
        self._root.columnconfigure(2, weight=1)
        self._root.rowconfigure(0, weight=1)

        # --- Left: video panel ---
        left = tk.Frame(self._root, bg=_PANEL_COLOR, bd=0)
        left.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self._video_label = tk.Label(
            left,
            bg=_PANEL_COLOR,
            text="Initialising camera...",
            fg=_TEXT_COLOR,
            font=("Helvetica", 12),
        )
        self._video_label.grid(row=0, column=0, sticky="nsew")

        # --- Centre: KPI gauges ---
        centre = tk.Frame(self._root, bg=_PANEL_COLOR, bd=0)
        centre.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=4)
        self._build_kpi_panel(centre)

        # --- Right: alert history ---
        right = tk.Frame(self._root, bg=_PANEL_COLOR, bd=0)
        right.grid(row=0, column=2, sticky="nsew", padx=(0, 4), pady=4)
        self._build_history_panel(right)

    def _build_kpi_panel(self, parent: tk.Frame):
        parent.columnconfigure(0, weight=1)

        tk.Label(parent, text="KPI Gauges", bg=_PANEL_COLOR,
                 fg=_TEXT_COLOR, font=("Helvetica", 13, "bold")).grid(
            row=0, column=0, pady=(10, 4), padx=10, sticky="w")

        self._kpi_canvas = tk.Canvas(
            parent, width=200, height=260,
            bg=_PANEL_COLOR, highlightthickness=0,
        )
        self._kpi_canvas.grid(row=1, column=0, padx=10, pady=4, sticky="n")

        # Pre-create canvas items so _draw_gauges can use itemconfig() instead
        # of delete("all") + recreate — eliminates the most expensive per-frame
        # canvas work.
        cx, cy, r = 100, 80, 60
        x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r
        self._gauge_bg  = self._kpi_canvas.create_arc(
            x0, y0, x1, y1, start=0, extent=180, outline="#3a3a3c", width=8, style="arc")
        self._gauge_fg  = self._kpi_canvas.create_arc(
            x0, y0, x1, y1, start=180, extent=0,
            outline=_SEVERITY_COLORS[0], width=8, style="arc")
        self._gauge_val = self._kpi_canvas.create_text(
            cx, cy + 10, text="0.0\u00b0", fill=_TEXT_COLOR, font=("Helvetica", 11, "bold"))
        self._gauge_lbl = self._kpi_canvas.create_text(
            cx, cy + 28, text="Cuello", fill="#8e8e93", font=("Helvetica", 9))

        # Status badge
        self._status_label = tk.Label(
            parent, text="Status: OK",
            bg=_PANEL_COLOR, fg=_SEVERITY_COLORS[0],
            font=("Helvetica", 14, "bold"),
        )
        self._status_label.grid(row=2, column=0, pady=(4, 10), padx=10)

        # Numeric KPI labels
        self._neck_label = self._make_kpi_label(parent, row=3, title="Neck flexion")
        self._fhp_label  = self._make_kpi_label(parent, row=4, title="FHP ratio")
        self._asym_label = self._make_kpi_label(parent, row=5, title="Shoulder asym")

    def _make_kpi_label(self, parent: tk.Frame, row: int, title: str) -> tk.Label:
        tk.Label(parent, text=title, bg=_PANEL_COLOR, fg="#8e8e93",
                 font=("Helvetica", 10)).grid(row=row, column=0, sticky="w", padx=12)
        lbl = tk.Label(parent, text="—", bg=_PANEL_COLOR, fg=_TEXT_COLOR,
                       font=("Helvetica", 12, "bold"))
        lbl.grid(row=row, column=0, sticky="e", padx=12)
        return lbl

    def _build_history_panel(self, parent: tk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        tk.Label(parent, text="Alert History", bg=_PANEL_COLOR,
                 fg=_TEXT_COLOR, font=("Helvetica", 13, "bold")).grid(
            row=0, column=0, pady=(10, 4), padx=10, sticky="w")

        self._history_text = tk.Text(
            parent, state="disabled", bg="#1c1c1e", fg=_TEXT_COLOR,
            font=("Courier", 9), relief="flat", bd=0,
            insertbackground=_TEXT_COLOR,
        )
        self._history_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # Configure colour tags
        for level, color in _SEVERITY_COLORS.items():
            self._history_text.tag_configure(f"level{level}", foreground=color)

        self._current_label = tk.Label(
            parent, text="Current: OK",
            bg=_PANEL_COLOR, fg=_SEVERITY_COLORS[0],
            font=("Helvetica", 12, "bold"),
        )
        self._current_label.grid(row=2, column=0, pady=(0, 10), padx=10)

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def _update_loop(self):
        if not self._running:
            return
        try:
            frame, landmarks = self._vm.capture_with_frame()
            report = self._pl.analyze(landmarks)
            self._hw.trigger_alert(report.severity)
            self._render(frame, report)
        except Exception:
            logger.exception("Dashboard update error.")
        finally:
            self._root.after(150, self._update_loop)

    def _render(self, frame: Optional[np.ndarray], report):
        self._render_video(frame, report)
        self._render_kpis(report)
        self._render_history(report)

    # ------------------------------------------------------------------
    # Rendering sub-methods
    # ------------------------------------------------------------------

    def _render_video(self, frame: Optional[np.ndarray], report):
        if not _HAS_PIL:
            return

        # --- no-signal placeholder ------------------------------------------
        if frame is None:
            if not hasattr(self, "_no_signal_frame"):
                placeholder = np.zeros((_DISPLAY_H, _DISPLAY_W, 3), dtype=np.uint8)
                placeholder[:] = (40, 40, 40)
                cv2.putText(placeholder, "No camera signal", (40, _DISPLAY_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
                self._no_signal_frame = placeholder
            if self._last_frame_id != -2:   # draw placeholder once
                pil_img = Image.fromarray(self._no_signal_frame)
                photo = ImageTk.PhotoImage(pil_img)
                self._last_photo = photo
                self._video_label.configure(image=photo, text="")
                self._last_frame_id = -2
            return

        # --- skip render if the inference thread hasn't produced a new frame --
        fid = id(frame)
        if fid == self._last_frame_id:
            return   # same numpy array — PIL/PhotoImage conversion not needed
        self._last_frame_id = fid

        # Work on a copy so we never mutate the shared inference-thread buffer.
        frame = frame.copy()

        h_orig, w_orig = frame.shape[:2]
        color = _SEVERITY_COLORS.get(int(report.severity), "#ffffff")
        bgr_color = _hex_to_bgr(color)

        self._draw_overlay(frame, report, w_orig, h_orig, bgr_color)

        # Downscale for display then add severity border
        display = cv2.resize(frame, (_DISPLAY_W, _DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(display, (0, 0), (_DISPLAY_W - 1, _DISPLAY_H - 1), bgr_color, 4)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._last_photo = photo   # prevent GC
        self._video_label.configure(image=photo, text="")

    def _draw_overlay(self, frame: np.ndarray, report, w: int, h: int, color: tuple):
        lms = getattr(report, "_raw_landmarks", None)
        # Access landmarks from the analysis source is not directly stored on
        # PostureReport; the caller's vision_source provides them.  We draw
        # from whatever the last captured landmarks were if accessible via a
        # wrapper method on the vision source.
        if hasattr(self._vm, "last_landmarks"):
            lms = self._vm.last_landmarks
        if lms is None:
            return
        try:
            for (a, b) in _POSE_CONNECTIONS:
                if a < len(lms) and b < len(lms):
                    ax = int(lms[a].x * w); ay = int(lms[a].y * h)
                    bx = int(lms[b].x * w); by = int(lms[b].y * h)
                    cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
            for lm in lms:
                px = int(lm.x * w); py = int(lm.y * h)
                cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
        except Exception:
            pass

    def _render_kpis(self, report):
        level  = int(report.severity)
        neck   = report.neck_flexion_deg
        fhp    = report.fhp_ratio
        asym   = report.shoulder_asymmetry_deg

        # Skip all KPI widget updates if nothing has changed enough to matter.
        # 0.5° / 0.005 is below the visual resolution of the text labels.
        if (level == self._last_severity
                and abs(neck - self._last_neck) < 0.5
                and abs(fhp  - self._last_fhp)  < 0.005
                and abs(asym - self._last_asym)  < 0.5):
            return

        self._last_severity = level
        self._last_neck = neck
        self._last_fhp  = fhp
        self._last_asym = asym

        color = _SEVERITY_COLORS[level]
        val_color = color if level > 0 else _TEXT_COLOR

        self._status_label.configure(text=f"Status: {report.severity.name}", fg=color)
        self._neck_label.configure(text=f"{neck:.1f}\u00b0",  fg=val_color)
        self._fhp_label.configure( text=f"{fhp:.3f}",         fg=val_color)
        self._asym_label.configure(text=f"{asym:.1f}\u00b0",  fg=val_color)

        self._draw_gauges(report)

    def _draw_gauges(self, report):
        """Update the canvas gauge via itemconfig — no delete/recreate."""
        clr   = _SEVERITY_COLORS[int(report.severity)]
        angle = min(report.neck_flexion_deg / 60.0, 1.0) * 180
        self._kpi_canvas.itemconfig(
            self._gauge_fg,
            start=180 - angle, extent=max(angle, 0.01),   # extent=0 hides arc
            outline=clr,
        )
        self._kpi_canvas.itemconfig(
            self._gauge_val,
            text=f"{report.neck_flexion_deg:.1f}\u00b0",
        )

    def _render_history(self, report):
        level = int(report.severity)
        color = _SEVERITY_COLORS[level]

        if level > 0:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            entry = (f"{ts}  {report.severity.name:7s}  {report.dominant_issue}\n",
                     f"level{level}")
            self._alert_history.append(entry)
            self._history_version += 1

        # Only rebuild the Text widget when the history actually changed.
        # Rebuilding on every frame (even with no new alerts) was one of the
        # most expensive per-cycle tkinter operations.
        if self._history_version != self._rendered_history_version:
            self._history_text.configure(state="normal")
            self._history_text.delete("1.0", "end")
            for text, tag in list(self._alert_history)[-20:]:
                self._history_text.insert("end", text, tag)
            self._history_text.see("end")
            self._history_text.configure(state="disabled")
            self._rendered_history_version = self._history_version

        self._current_label.configure(
            text=f"Current: {report.severity.name}", fg=color)

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------

    def _on_close(self):
        self._running = False
        self._root.destroy()


# ---------------------------------------------------------------------------
# Simulation-aware wrapper for CameraSimulator
# ---------------------------------------------------------------------------

class SimVisionSource:
    """Wraps CameraSimulator to expose the VisionManager interface expected by
    ErgoDashboard (capture_with_frame + last_landmarks).

    Parameters
    ----------
    simulator:
        A CameraSimulator instance.
    posture_landmarks_cls:
        PostureLandmarks class (passed in to avoid circular import).
    """

    def __init__(self, simulator, posture_landmarks_cls):
        self._sim = simulator
        self._PostureLandmarks = posture_landmarks_cls
        self.last_landmarks = None

    def capture_with_frame(self):
        lms = self._sim.get_landmarks()
        frame = self._sim.get_frame()
        self.last_landmarks = lms
        return frame, self._PostureLandmarks(normalized=lms, world=[], is_valid=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_color: str) -> tuple:
    """Convert '#rrggbb' to (B, G, R) tuple for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)
