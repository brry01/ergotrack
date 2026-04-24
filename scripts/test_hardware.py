"""Diagnóstico completo de hardware — GPIO buzzer/motor + OLED.

Corre ANTES de ErgoTrack para confirmar que todo el hardware
responde correctamente.

Uso:
    python scripts/test_hardware.py                    # prueba todo
    python scripts/test_hardware.py --gpio             # solo buzzer + motor
    python scripts/test_hardware.py --motor            # solo motor de vibración
    python scripts/test_hardware.py --buzzer           # solo buzzer
    python scripts/test_hardware.py --oled             # solo OLED
    python scripts/test_hardware.py --passive-buzzer   # buzzer pasivo (PWM)
    python scripts/test_hardware.py --invert-buzzer    # lógica invertida buzzer
    python scripts/test_hardware.py --invert-motor     # lógica invertida motor
"""
from __future__ import annotations
import argparse, os, sys, time, shutil, subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpio",   action="store_true", help="Probar buzzer + motor")
parser.add_argument("--motor",  action="store_true", help="Probar solo motor")
parser.add_argument("--buzzer", action="store_true", help="Probar solo buzzer")
parser.add_argument("--oled",   action="store_true", help="Probar solo OLED")
parser.add_argument("--passive-buzzer", action="store_true",
                    help="Tratar buzzer como pasivo (PWM 2kHz) en vez de activo (DC)")
parser.add_argument("--invert-buzzer", action="store_true",
                    help="Lógica invertida buzzer: LOW=ON, HIGH=OFF")
parser.add_argument("--invert-motor", action="store_true",
                    help="Lógica invertida motor: LOW=ON, HIGH=OFF")
# pines configurables (deben coincidir con config/default.yaml)
parser.add_argument("--buzzer-pin", type=int, default=18,
                    help="Pin BCM del buzzer (default: 18)")
parser.add_argument("--motor-pin",  type=int, default=17,
                    help="Pin BCM del motor de vibración (default: 17)")
parser.add_argument("--oled-addr",  default="0x3C")
parser.add_argument("--i2c-bus",    type=int, default=1)
args = parser.parse_args()

# Si no se especifica nada → probar todo
run_explicit = args.gpio or args.motor or args.buzzer or args.oled
run_motor  = not run_explicit or args.gpio or args.motor
run_buzzer = not run_explicit or args.gpio or args.buzzer
run_oled   = not run_explicit or args.oled

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
if run_motor or run_buzzer:
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
# helpers GPIO
# ══════════════════════════════════════════════════════════════════════════════

def _check_gpio_access():
    gpio_dev = None
    for dev in ["/dev/gpiochip4", "/dev/gpiochip0", "/dev/gpiomem"]:
        if os.path.exists(dev):
            gpio_dev = dev
            break
    if gpio_dev and os.access(gpio_dev, os.R_OK | os.W_OK):
        ok(f"{gpio_dev} accesible")
    else:
        warn(
            f"Sin acceso a {gpio_dev or 'dispositivo GPIO'}.\n"
            "     Fix permanente: sudo usermod -aG gpio $USER  (luego cerrar sesión)\n"
            "     O corre el script con 'newgrp gpio' primero."
        )

def _pulse_pin(GPIO, pin, on, off, passive_pwm=False, duration=0.2, count=2):
    """Emit `count` pulses on `pin`. Returns True on success."""
    try:
        if passive_pwm:
            pwm = GPIO.PWM(pin, 2000)
            for i in range(count):
                pwm.start(50); time.sleep(duration)
                pwm.stop();    time.sleep(0.1)
            pwm.stop()
        else:
            for i in range(count):
                GPIO.output(pin, on);  time.sleep(duration)
                GPIO.output(pin, off); time.sleep(0.1)
        return True
    except Exception as exc:
        fail(f"Error GPIO pin {pin}: {exc}")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MOTOR DE VIBRACIÓN  (BCM 17 por defecto)
# ══════════════════════════════════════════════════════════════════════════════
if run_motor:
    hdr(f"2 ─ Motor de vibración (BCM {args.motor_pin})")

    if GPIO is None:
        warn("Omitido — RPi.GPIO no disponible")
    else:
        _check_gpio_access()

        motor_inv = args.invert_motor
        ON  = GPIO.LOW  if motor_inv else GPIO.HIGH
        OFF = GPIO.HIGH if motor_inv else GPIO.LOW
        idle = GPIO.HIGH if motor_inv else GPIO.LOW

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(args.motor_pin, GPIO.OUT, initial=idle)

            print(f"     Simulando LEVEL1 — 1 vibración (150 ms)…")
            _pulse_pin(GPIO, args.motor_pin, ON, OFF, duration=0.15, count=1)
            time.sleep(0.3)

            print(f"     Simulando LEVEL2 — 2 vibraciones…")
            _pulse_pin(GPIO, args.motor_pin, ON, OFF, duration=0.15, count=2)
            time.sleep(0.3)

            print(f"     Simulando LEVEL3 — 3 vibraciones…")
            _pulse_pin(GPIO, args.motor_pin, ON, OFF, duration=0.30, count=3)

            ok("Motor probado — ¿vibró?")

        except Exception as exc:
            all_ok = fail(f"Error motor: {exc}")
        finally:
            try:
                GPIO.output(args.motor_pin, idle)
                GPIO.cleanup()
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# 3.  BUZZER  (BCM 18 por defecto)
# ══════════════════════════════════════════════════════════════════════════════
if run_buzzer:
    hdr(f"3 ─ Buzzer KY-012 (BCM {args.buzzer_pin})")

    if GPIO is None:
        warn("Omitido — RPi.GPIO no disponible")
    else:
        _check_gpio_access()

        buzzer_inv = args.invert_buzzer
        ON  = GPIO.LOW  if buzzer_inv else GPIO.HIGH
        OFF = GPIO.HIGH if buzzer_inv else GPIO.LOW
        idle = GPIO.HIGH if buzzer_inv else GPIO.LOW

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(args.buzzer_pin, GPIO.OUT, initial=idle)

            if args.passive_buzzer:
                hdr("  Modo: buzzer PASIVO (PWM 2 kHz)")
                print("     3 bips PWM…")
                _pulse_pin(GPIO, args.buzzer_pin, ON, OFF, passive_pwm=True,
                           duration=0.30, count=3)
                ok("Patrón PWM enviado — ¿escuchaste el tono?")
            else:
                hdr("  Modo: buzzer ACTIVO (DC HIGH/LOW)")
                print("     3 pulsos DC  (simulando LEVEL3)…")
                _pulse_pin(GPIO, args.buzzer_pin, ON, OFF, duration=0.30, count=3)
                time.sleep(0.3)

                print("     DC continuo 1 s…")
                GPIO.output(args.buzzer_pin, ON)
                time.sleep(1.0)
                GPIO.output(args.buzzer_pin, OFF)
                ok("¿Sonó el buzzer durante 1 segundo?")

        except Exception as exc:
            all_ok = fail(f"Error buzzer: {exc}")
        finally:
            try:
                GPIO.output(args.buzzer_pin, idle)
                GPIO.cleanup()
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# 4.  I2C / OLED
# ══════════════════════════════════════════════════════════════════════════════
if run_oled:
    hdr("4 ─ I2C / OLED")

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
