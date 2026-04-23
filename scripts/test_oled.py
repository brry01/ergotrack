"""Standalone OLED diagnostic — run this on the RPi to pinpoint the issue.

Usage:
    python scripts/test_oled.py
    python scripts/test_oled.py --address 0x3D   # if your OLED uses 0x3D
    python scripts/test_oled.py --bus 0           # if you wired to I2C bus 0
"""
import argparse
import sys
import time


def step(msg):
    print(f"\n[STEP] {msg}")


def ok(msg=""):
    print(f"  ✓  {msg}" if msg else "  ✓")


def fail(msg):
    print(f"  ✗  {msg}")
    sys.exit(1)


def warn(msg):
    print(f"  !  {msg}")


# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--address", default="0x3C",
                    help="I2C address of the OLED (default: 0x3C)")
parser.add_argument("--bus", type=int, default=1,
                    help="I2C bus number (default: 1)")
args = parser.parse_args()

i2c_addr = int(args.address, 16)
i2c_bus  = args.bus

print("=" * 55)
print("  ErgoTrack — OLED diagnostic")
print("=" * 55)
print(f"  Target: SSD1306 at bus={i2c_bus}, address=0x{i2c_addr:02X}")

# ---------------------------------------------------------------------------
step("1 — Check I2C bus with i2cdetect")
import subprocess, shutil
if shutil.which("i2cdetect"):
    result = subprocess.run(
        ["i2cdetect", "-y", str(i2c_bus)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    addr_hex = f"{i2c_addr:02x}"
    if addr_hex in result.stdout:
        ok(f"Device found at 0x{i2c_addr:02X} on bus {i2c_bus}")
    else:
        fail(
            f"No device at 0x{i2c_addr:02X} on bus {i2c_bus}.\n"
            "  Check wiring: SDA→pin3, SCL→pin5, VCC→3.3V, GND→GND.\n"
            "  If the address is wrong, rerun with --address 0x3D\n"
            "  Enable I2C: sudo raspi-config → Interface Options → I2C"
        )
else:
    warn("i2cdetect not found — skipping bus scan "
         "(install with: sudo apt install i2c-tools)")

# ---------------------------------------------------------------------------
step("2 — Check luma.oled import")
try:
    from luma.core.interface.serial import i2c as luma_i2c
    from luma.oled.device import ssd1306
    ok("luma.oled imported successfully")
except ImportError as e:
    fail(f"luma.oled not installed: {e}\n"
         "  Fix: pip install luma.oled")

# ---------------------------------------------------------------------------
step("3 — Check Pillow import")
try:
    from PIL import Image, ImageDraw, ImageFont
    ok("Pillow imported successfully")
except ImportError as e:
    fail(f"Pillow not installed: {e}\n"
         "  Fix: pip install Pillow")

# ---------------------------------------------------------------------------
step("4 — Open I2C device")
try:
    serial = luma_i2c(port=i2c_bus, address=i2c_addr)
    ok(f"I2C handle opened (bus={i2c_bus}, addr=0x{i2c_addr:02X})")
except Exception as e:
    fail(f"Failed to open I2C: {e}\n"
         "  Check /dev/i2c-* permissions: sudo usermod -aG i2c $USER\n"
         "  Or run with sudo to verify hardware is the issue.")

# ---------------------------------------------------------------------------
step("5 — Initialise SSD1306 display")
try:
    device = ssd1306(serial, width=128, height=64)
    ok("SSD1306 initialised")
except Exception as e:
    fail(f"SSD1306 init failed: {e}")

# ---------------------------------------------------------------------------
step("6 — Render test pattern (blink 3×)")
try:
    from luma.core.render import canvas

    font = ImageFont.load_default()

    for i in range(3):
        with canvas(device) as draw:
            draw.rectangle((0, 0, 127, 63), outline="white", fill="black")
            draw.text((4, 4),  "ErgoTrack", font=font, fill="white")
            draw.text((4, 18), "OLED OK!",  font=font, fill="white")
            draw.text((4, 32), f"Test {i+1}/3", font=font, fill="white")
        time.sleep(0.6)

        with canvas(device) as draw:
            draw.rectangle((0, 0, 127, 63), fill="black")   # clear
        time.sleep(0.3)

    ok("Test pattern rendered — you should have seen 3 blinks on the OLED")
except Exception as e:
    fail(f"Render failed: {e}")

# ---------------------------------------------------------------------------
step("7 — Verify ErgoTrack config matches")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from modules.config_profile import ConfigProfile
    cfg = ConfigProfile("config/default.yaml")
    hw = cfg.hardware
    cfg_addr = hw.oled_address
    if cfg_addr != i2c_addr:
        warn(
            f"Config oled_address (0x{cfg_addr:02X}) differs from "
            f"tested address (0x{i2c_addr:02X}).\n"
            f"  Update config/default.yaml → oled_address: "
            f'"0x{i2c_addr:02X}"'
        )
    else:
        ok(f"Config address matches (0x{cfg_addr:02X})")
except Exception as e:
    warn(f"Could not load config: {e}")

print("\n" + "=" * 55)
print("  All checks passed — OLED should work in ErgoTrack.")
print("  Run:  git pull && python main.py --monitor")
print("=" * 55)
