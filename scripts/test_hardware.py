"""Diagnóstico completo de hardware — GPIO, buzzer, motor, OLED.

Corre ANTES de ErgoTrack para confirmar que todo el hardware
responde correctamente.

Uso:
    python scripts/test_hardware.py           # prueba todo
    python scripts/test_hardware.py --gpio    # solo GPIO / buzzer / motor
    python scripts/test_hardware.py --oled    # solo OLED
"""
from __future__ import annotations
import argparse, os, sys, time, shutil, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpio", action="store_true")
parser.add_argument("--oled", action="store_true")
# pines configurables (deben coincidir con config/default.yaml)
parser.add_argument("--led-pin",    type=int, default=24)
parser.add_argument("--buzzer-pin", type=int, default=18)
parser.add_argument("--oled-addr",  default="0x3C")
parser.add_argument("--i2c-bus",    type=int, default=1)
args = parser.parse_args()
run_all  = not args.gpio and not args.oled
run_gpio = run_all or args.gpio
run_oled = run_all or args.oled

PASS = "\033[32m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[33m!\033[0m"

def ok(msg):   print(f"  {PASS}  {msg}")
def fail(msg): print(f"  {FAIL}  {msg}"); return False
def warn(msg): print(f"  {WARN}  {msg}")
def hdr(msg):  print(f"\n\033[1m{msg}\033[0m")

all_ok = True

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LIBRERÍAS
# ══════════════════════════════════════════════════════════════════════════════
hdr("1 ─ Librerías Python")

# GPIO
GPIO = None
if run_gpio:
    try:
        import RPi.GPIO as GPIO
        ok(f"RPi.GPIO importado  ({GPIO.VERSION})")
    except ImportError:
        all_ok = fail(
            "RPi.GPIO no encontrado.\n"
            "     Fix: pip install rpi-lgpio"
        )

# luma.oled + PIL
luma_i2c = luma_ssd1306 = None
if run_oled:
    try:
        from luma.core.interface.serial import i2c as luma_i2c
        from luma.oled.device import ssd1306 as luma_ssd1306
        ok("luma.oled importado")
    except ImportError:
        all_ok = fail(
            "luma.oled no encontrado.\n"
            "     Fix: pip install luma.oled"
        )

    try:
        from PIL import Image, ImageDraw, ImageFont
        ok("Pillow importado")
    except ImportError:
        all_ok = fail(
            "Pillow no encontrado.\n"
            "     Fix: pip install Pillow"
        )

# ══════════════════════════════════════════════════════════════════════════════
# 2.  GPIO  (LED + Buzzer/Motor)
# ══════════════════════════════════════════════════════════════════════════════
if run_gpio:
    hdr("2 ─ GPIO")

    if GPIO is None:
        warn("Omitido — RPi.GPIO no disponible")
    else:
        # Permisos
        gpiomem = "/dev/gpiomem"
        if os.access(gpiomem, os.R_OK | os.W_OK):
            ok(f"{gpiomem} accesible")
        else:
            all_ok = fail(
                f"{gpiomem} sin permisos.\n"
                "     Fix: sudo usermod -aG gpio $USER  (luego cerrar sesión)"
            )

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # ── LED ──────────────────────────────────────────────────────────
            hdr("  2a ─ LED (BCM {})".format(args.led_pin))
            GPIO.setup(args.led_pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"     Encendiendo LED 3 veces (pin BCM {args.led_pin})…")
            for _ in range(3):
                GPIO.output(args.led_pin, GPIO.HIGH); time.sleep(0.3)
                GPIO.output(args.led_pin, GPIO.LOW);  time.sleep(0.2)
            ok("LED pulsado — ¿lo viste parpadear?")

            # ── Buzzer / Motor de vibración ───────────────────────────────────
            hdr("  2b ─ Buzzer / Motor (BCM {})".format(args.buzzer_pin))
            GPIO.setup(args.buzzer_pin, GPIO.OUT, initial=GPIO.LOW)
            pwm = GPIO.PWM(args.buzzer_pin, 2000)   # 2 kHz para buzzer pasivo
            print(f"     Patrón LEVEL2 (2 bips) en pin BCM {args.buzzer_pin}…")
            for _ in range(2):
                pwm.start(50); time.sleep(0.1)
                pwm.stop();    time.sleep(0.08)
            time.sleep(0.3)
            ok("Patrón enviado — ¿escuchaste / sentiste la vibración?")

            # ── DC puro para motor de vibración ──────────────────────────────
            print("     DC continuo 1 s para motor de vibración…")
            GPIO.output(args.buzzer_pin, GPIO.HIGH)
            time.sleep(1.0)
            GPIO.output(args.buzzer_pin, GPIO.LOW)
            ok("DC enviado — ¿vibró el motor?")

        except Exception as exc:
            all_ok = fail(f"Error GPIO: {exc}")
        finally:
            try:
                GPIO.cleanup()
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# 3.  I2C / OLED
# ══════════════════════════════════════════════════════════════════════════════
if run_oled:
    hdr("3 ─ I2C / OLED")

    # i2cdetect
    addr_int = int(args.oled_addr, 16)
    if shutil.which("i2cdetect"):
        res = subprocess.run(
            ["i2cdetect", "-y", str(args.i2c_bus)],
            capture_output=True, text=True,
        )
        print(res.stdout)
        if f"{addr_int:02x}" in res.stdout:
            ok(f"Dispositivo detectado en 0x{addr_int:02X}  (bus {args.i2c_bus})")
        else:
            all_ok = fail(
                f"Nada en 0x{addr_int:02X}. Causas comunes:\n"
                "     • VCC conectado a 5V en vez de 3.3V\n"
                "     • SDA / SCL intercambiados\n"
                "     • I2C deshabilitado:  sudo raspi-config → Interface Options → I2C\n"
                f"     • Probar otra dirección: python scripts/test_hardware.py --oled --oled-addr 0x3D"
            )
    else:
        warn("i2cdetect no disponible (instalar: sudo apt install i2c-tools)")

    if luma_i2c and luma_ssd1306:
        try:
            serial = luma_i2c(port=args.i2c_bus, address=addr_int)
            device = luma_ssd1306(serial, width=128, height=64)
            ok("SSD1306 inicializado")

            from luma.core.render import canvas
            from PIL import ImageFont
            font = ImageFont.load_default()

            for i in range(3):
                with canvas(device) as draw:
                    draw.rectangle((0, 0, 127, 63), fill="black")
                    draw.text((4,  2), "ErgoTrack",     font=font, fill="white")
                    draw.text((4, 14), "OLED OK!",       font=font, fill="white")
                    draw.text((4, 26), f"Test {i+1}/3", font=font, fill="white")
                time.sleep(0.5)
                with canvas(device) as draw:
                    draw.rectangle((0, 0, 127, 63), fill="black")
                time.sleep(0.25)

            ok("Pantalla probada — ¿viste 3 parpadeos?")
        except Exception as exc:
            all_ok = fail(f"Error OLED: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# Resumen
# ══════════════════════════════════════════════════════════════════════════════
hdr("─" * 50)
if all_ok:
    print(f"  {PASS}  Todo OK — el hardware responde correctamente.")
    print("       Corre:  python main.py --monitor")
else:
    print(f"  {FAIL}  Hay errores — revisa los mensajes de arriba.")
