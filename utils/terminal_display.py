"""Scrolling ANSI terminal monitor for ErgoTrack headless mode.

Each inference cycle prints one coloured line:

    Frame   Cuello     FHP    Hombros   Alerta
    ────────────────────────────────────────────────────────────
     1156   0.0deg   0.000   11.1deg   Asimetria hombros 11.1°
     1157   0.0deg   0.000   11.2deg   Asimetria hombros 11.2°

Metric colours:
  Green  — below warning threshold
  Yellow — warning threshold reached
  Red    — severe threshold reached

An extra banner line is printed whenever the alert level changes.
"""
from __future__ import annotations

import sys
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour codes (reset to default if terminal doesn't support them)
# ---------------------------------------------------------------------------

_R     = "\033[0m"          # reset
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[32m"
_YELLW = "\033[33m"
_RED   = "\033[91m"         # bright red — most visible on dark terminals
_CYAN  = "\033[96m"
_WHITE = "\033[97m"
_GRAY  = "\033[90m"


def _supports_color() -> bool:
    """True when stdout is a TTY and the platform supports ANSI codes."""
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    # Windows cmd doesn't support ANSI unless ANSICON / WT is set
    import os
    return os.name != "nt" or "WT_SESSION" in os.environ or "ANSICON" in os.environ


def _supports_unicode() -> bool:
    """True when stdout can encode the box-drawing and symbol characters we use."""
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "\u2500\u2713\u26a0".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# ---------------------------------------------------------------------------
# TerminalMonitor
# ---------------------------------------------------------------------------

class TerminalMonitor:
    """Prints one coloured status line per inference cycle.

    Parameters
    ----------
    thresholds:
        ``AlertThresholds`` dataclass (from ConfigProfile).
    use_color:
        Force-enable or force-disable ANSI colours.  When *None* the
        decision is made automatically from the terminal type.
    header_every:
        Re-print the column header every N lines so it stays visible when
        scrolling through a long session.  Set to 0 to print once only.
    """

    # Spanish labels for each dominant-issue key
    _ISSUE_LABELS = {
        "neck_flexion":       "Flexion cuello",
        "fhp":                "Proyeccion cabeza",
        "shoulder_asymmetry": "Asimetria hombros",
        "none":               "Postura OK",
    }

    # AlertLevel.value → display name
    _LEVEL_NAMES = {0: "OK", 1: "LEVE", 2: "MODERADO", 3: "SEVERO"}

    # AlertLevel.value → ANSI colour
    _LEVEL_COLORS = {0: _GREEN, 1: _YELLW, 2: _YELLW, 3: _RED}

    def __init__(self, thresholds, use_color: Optional[bool] = None,
                 header_every: int = 40):
        self._t = thresholds
        self._color   = _supports_color()   if use_color is None else use_color
        self._unicode = _supports_unicode()
        self._header_every = header_every
        self._frame = 0
        self._prev_level = -1   # sentinel — force banner on first call
        self._print_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, report) -> None:
        """Print one line for the current PostureReport."""
        self._frame += 1

        # Re-print column header periodically
        if self._header_every and self._frame % self._header_every == 0:
            self._print_header()

        # Print a banner when the alert level changes
        level_val = int(report.severity)
        if level_val != self._prev_level:
            self._print_level_banner(level_val)
            self._prev_level = level_val

        neck = report.neck_flexion_deg
        fhp  = report.fhp_ratio
        asym = report.shoulder_asymmetry_deg
        issue_key = report.dominant_issue

        neck_c = self._metric_color(neck, self._t.neck_flexion_warn,
                                    self._t.neck_flexion_severe)
        fhp_c  = self._metric_color(fhp,  self._t.fhp_warn,
                                    self._t.fhp_severe)
        asym_c = self._metric_color(asym, self._t.shoulder_asymmetry_warn,
                                    self._t.shoulder_asymmetry_severe)

        issue_c = {"neck_flexion": neck_c, "fhp": fhp_c,
                   "shoulder_asymmetry": asym_c,
                   "none": _GREEN}.get(issue_key, _WHITE)

        issue_label = self._ISSUE_LABELS.get(issue_key, issue_key)
        if issue_key == "neck_flexion":
            issue_str = f"{issue_label} {neck:.1f}\u00b0"
        elif issue_key == "fhp":
            issue_str = f"{issue_label} {fhp:.3f}"
        elif issue_key == "shoulder_asymmetry":
            issue_str = f"{issue_label} {asym:.1f}\u00b0"
        else:
            issue_str = issue_label

        line = (
            f"{self._c(_GRAY, f'{self._frame:>6}')}"
            f"  {self._c(neck_c, f'{neck:>6.1f}deg')}"
            f"  {self._c(fhp_c,  f'{fhp:>7.3f}')}"
            f"  {self._c(asym_c, f'{asym:>6.1f}deg')}"
            f"   {self._c(issue_c, issue_str)}"
        )
        print(line, flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metric_color(self, value: float, warn: float, severe: float) -> str:
        if value >= severe:
            return _RED
        if value >= warn:
            return _YELLW
        return _GREEN

    def _c(self, code: str, text: str) -> str:
        """Wrap *text* in an ANSI code if colours are enabled."""
        if self._color:
            return f"{code}{text}{_R}"
        return text

    def _print_header(self) -> None:
        sep_char = "\u2500" if self._unicode else "-"
        sep = sep_char * 62
        if self._color:
            print(
                f"\n{_BOLD}{_WHITE}"
                f"{'Frame':>6}  {'Cuello':>9}  {'FHP':>7}  {'Hombros':>9}"
                f"   Alerta{_R}"
            )
            print(f"{_GRAY}{sep}{_R}")
        else:
            print(f"\n{'Frame':>6}  {'Cuello':>9}  {'FHP':>7}  {'Hombros':>9}   Alerta")
            print(sep)

    def _print_level_banner(self, level_val: int) -> None:
        name  = self._LEVEL_NAMES.get(level_val, str(level_val))
        color = self._LEVEL_COLORS.get(level_val, _WHITE)
        ok_sym   = "\u2713" if self._unicode else "OK"
        warn_sym = "\u26a0 " if self._unicode else "!!"
        if level_val == 0:
            msg = f"  {ok_sym}  Alerta resuelta -- postura correcta"
        else:
            msg = f"  {warn_sym}  Alerta {name}"
        print(self._c(color, self._c(_BOLD, msg)), flush=True)
