#!/usr/bin/env bash
# =============================================================================
# ErgoTrack — Setup script for Raspberry Pi OS (Bookworm, 64-bit)
# =============================================================================
# Run once after cloning the repo:
#   bash scripts/setup_rpi.sh
#
# What this script does:
#   1. Enables Camera and I2C interfaces
#   2. Installs apt packages (OpenCV, picamera2, GPIO, tkinter, NumPy, PIL)
#   3. Installs pip packages (PyYAML, customtkinter, luma.oled, mediapipe)
#   4. Downloads the MediaPipe PoseLandmarker model file (~30 MB)
#
# Requirements:
#   - Raspberry Pi OS Bookworm (64-bit)
#   - Python 3.11+
#   - Internet connection
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colours
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Guard: must run on RPi OS
# ---------------------------------------------------------------------------
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    warn "This script is designed for Raspberry Pi OS."
    warn "On a PC, use simulation mode instead: python main.py --simulate"
    read -rp "Continue anyway? [y/N] " ans
    [[ "${ans,,}" == "y" ]] || exit 0
fi

cd "$PROJECT_DIR"
info "Working directory: $PROJECT_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Enable hardware interfaces via raspi-config
# ---------------------------------------------------------------------------
info "Step 1/4 — Enabling Camera and I2C interfaces..."

if command -v raspi-config &>/dev/null; then
    sudo raspi-config nonint do_camera 0    && info "  Camera enabled."
    sudo raspi-config nonint do_i2c    0    && info "  I2C enabled."
else
    warn "  raspi-config not found. Enabling via /boot/firmware/config.txt..."
    BOOT_CFG="/boot/firmware/config.txt"
    if [ -f "$BOOT_CFG" ]; then
        grep -q "^camera_auto_detect" "$BOOT_CFG" || \
            echo "camera_auto_detect=1" | sudo tee -a "$BOOT_CFG" > /dev/null
        grep -q "^dtparam=i2c_arm=on" "$BOOT_CFG" || \
            echo "dtparam=i2c_arm=on"  | sudo tee -a "$BOOT_CFG" > /dev/null
        # Faster I2C (400 kHz) — improves OLED refresh
        grep -q "dtparam=i2c_arm_baudrate" "$BOOT_CFG" || \
            echo "dtparam=i2c_arm_baudrate=400000" | sudo tee -a "$BOOT_CFG" > /dev/null
        info "  Entries added to $BOOT_CFG"
    else
        warn "  $BOOT_CFG not found. Enable Camera and I2C manually via raspi-config."
    fi
fi

# ---------------------------------------------------------------------------
# Step 2 — apt packages
# ---------------------------------------------------------------------------
info "Step 2/4 — Installing apt packages..."

sudo apt-get update -qq

sudo apt-get install -y \
    python3-picamera2 \
    python3-rpi.gpio \
    python3-lgpio \
    python3-opencv \
    python3-numpy \
    python3-pil \
    python3-tk \
    i2c-tools

info "  apt packages installed."

# Verify I2C tools can see the bus (non-fatal — hardware may not be connected)
if command -v i2cdetect &>/dev/null; then
    info "  I2C bus scan (OLED should appear at 0x3C if connected):"
    i2cdetect -y 1 2>/dev/null || warn "  i2cdetect failed — I2C may not be active yet (reboot needed)."
fi

# ---------------------------------------------------------------------------
# Step 3 — pip packages
# ---------------------------------------------------------------------------
info "Step 3/4 — Installing pip packages..."

# Use --break-system-packages only if pip version requires it (Python 3.11+)
PIP_FLAGS="--break-system-packages"
python3 -m pip install $PIP_FLAGS --upgrade pip --quiet

# Core pip packages
python3 -m pip install $PIP_FLAGS \
    "PyYAML>=6.0" \
    "customtkinter>=5.2" \
    "pytest>=7.4" \
    "luma.oled>=3.12"

info "  Core pip packages installed."

# mediapipe — arm64 community wheel
info "  Installing mediapipe (arm64 community wheel)..."
if python3 -m pip install $PIP_FLAGS mediapipe \
        --extra-index-url https://pinto0309.github.io/simple/ \
        --quiet 2>/dev/null; then
    info "  mediapipe installed from pinto0309 index."
else
    warn "  pinto0309 index failed. Trying PyPI directly..."
    if python3 -m pip install $PIP_FLAGS mediapipe --quiet 2>/dev/null; then
        info "  mediapipe installed from PyPI."
    else
        warn "  mediapipe could not be installed automatically."
        warn "  Download a wheel manually from: https://github.com/PINTO0309/mediapipe-bin"
        warn "  Then: pip install mediapipe-*.whl"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4 — Download MediaPipe model
# ---------------------------------------------------------------------------
info "Step 4/4 — Downloading MediaPipe PoseLandmarker model..."

python3 scripts/download_models.py && info "  Model ready." || \
    warn "  Model download failed. Run manually: python3 scripts/download_models.py"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}  ErgoTrack setup complete!${NC}"
echo -e "${GREEN}=====================================================${NC}"
echo ""
echo "  Headless mode:       python3 main.py"
echo "  GUI mode:            python3 main.py --gui"
echo "  Simulation (no HW):  python3 main.py --simulate"
echo "  Tests:               python3 -m pytest tests/ -v"
echo ""
echo -e "${YELLOW}  NOTE: A reboot may be required for Camera and I2C to activate.${NC}"
echo -e "${YELLOW}  After reboot, verify camera with: rpicam-hello${NC}"
echo -e "${YELLOW}  Verify OLED with:                 i2cdetect -y 1${NC}"
echo ""
