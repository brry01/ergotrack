# ErgoTrack

> Monitor ergonómico de postura en tiempo real para Raspberry Pi 5.

Detecta postura incorrecta mediante visión por computadora, calcula métricas de cuello, proyección de cabeza y asimetría de hombros, y emite alertas físicas progresivas a través de motor de vibración, buzzer y pantalla OLED.

---

## Características

- **Inferencia local** — MoveNet Lightning (TFLite int8) vía `ai-edge-litert`. Sin envío de datos a la nube.
- **Privacidad** — los frames se eliminan de memoria inmediatamente después de la inferencia. Nada se escribe a disco.
- **Alertas físicas progresivas** — OLED en nivel 1, motor en nivel 2, buzzer en nivel 3.
- **OLED animada** — alterna cada 5 s entre una cara expresiva y las métricas de postura.
- **Recordatorio** — si la postura no se corrige, la alerta se repite cada 15 minutos.
- **Tres modos de ejecución** — headless, monitor con colores ANSI, y dashboard GUI.
- **Simulador** — modo sin cámara para desarrollo y pruebas en cualquier PC.
- **Hot-reload de config** — edita `config/default.yaml` sin reiniciar el proceso.
- **Protección térmica** — `ThermalGuard` pausa la inferencia si la CPU supera 75 °C.

---

## Hardware requerido

| Componente | Modelo |
|-----------|--------|
| SBC | Raspberry Pi 5 |
| Cámara | RPi Camera Module 3 |
| Pantalla | OLED SSD1306 128×64 (I2C) |
| Alerta táctil | Motor de vibración (módulo) |
| Alerta sonora | Buzzer pasivo |

---

## Pinout

| Componente | Pin físico | BCM |
|-----------|-----------|-----|
| OLED VCC | Pin 1 | 3.3 V |
| OLED GND | Pin 6 | GND |
| OLED SDA | Pin 3 | GPIO 2 |
| OLED SCL | Pin 5 | GPIO 3 |
| Motor VCC | Pin 2 | 5 V |
| Motor GND | Pin 6 | GND |
| Motor S | Pin 11 | GPIO 17 |
| Buzzer VCC | Pin 2 | 5 V |
| Buzzer GND | Pin 6 | GND |
| Buzzer S | Pin 12 | GPIO 18 |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/brry01/ergotrack
cd ergotrack
```

### 2. Instalar dependencias del sistema (RPi OS)

```bash
sudo apt install python3-picamera2 python3-opencv python3-numpy \
                 python3-pil python3-tk i2c-tools
```

### 3. Instalar dependencias Python

```bash
pip install ai-edge-litert luma.oled PyYAML customtkinter
```

### 4. Habilitar I2C y GPIO

```bash
sudo raspi-config   # Interface Options → I2C → Yes
sudo usermod -aG gpio,i2c $USER
# Cerrar sesión y volver a entrar
```

### 5. Descargar el modelo MoveNet

```bash
python scripts/download_models.py
```

### 6. Verificar hardware

```bash
i2cdetect -y 1                      # OLED debe aparecer en 0x3C
python scripts/test_hardware.py     # prueba motor + buzzer + OLED
```

---

## Uso

### Modos principales

```bash
# Headless — logs en terminal
python main.py

# Monitor — métricas en tiempo real con colores ANSI
python main.py --monitor

# Dashboard GUI — video + gauges + historial
python main.py --gui
```

### Simulación (sin cámara, hardware real activo)

```bash
python main.py --simulate --monitor
python main.py --simulate --monitor --scenario severe_neck
python main.py --simulate --gui
```

Escenarios disponibles: `good_posture`, `mild_fhp`, `severe_neck`, `shoulder_tilt`, `cycling` (rota todos, default).

### Modo prueba

Reproduce el comportamiento completo pero la alerta se repite cada **10 segundos** en vez de 15 minutos:

```bash
python main.py --test --monitor
python main.py --test --simulate --monitor
```

### Todos los flags

| Flag | Descripción |
|------|-------------|
| `--gui` | Lanza el dashboard gráfico |
| `--simulate` | Usa landmarks sintéticos (sin cámara) |
| `--monitor` | Métricas ANSI en terminal (solo headless) |
| `--test` | Alerta cada 10 s en vez de 15 min |
| `--scenario` | Escenario del simulador (requiere `--simulate`) |
| `--config` | Ruta a un archivo YAML de configuración alternativo |
| `--user-config` | YAML de override que se fusiona sobre el default |

---

## Diagnóstico de hardware

```bash
# Componentes individuales
python scripts/test_hardware.py --motor
python scripts/test_hardware.py --buzzer
python scripts/test_hardware.py --oled

# Diagnóstico completo de OLED (7 pasos)
python scripts/test_oled.py
```

---

## Configuración

Edita `config/default.yaml`. El proceso detecta cambios y recarga sin reiniciar.

```yaml
vision:
  fps: 5
  resolution: [320, 240]
  model_path: "models/movenet_lightning.tflite"

thresholds:
  neck_flexion_warn: 25.0       # grados; 0 = erguido
  neck_flexion_severe: 40.0
  fhp_warn: 0.15                # normalizado al ancho de hombros
  fhp_severe: 0.30
  shoulder_asymmetry_warn: 5.0  # grados de inclinación vertical
  shoulder_asymmetry_severe: 10.0

hardware:
  buzzer_pin: 18        # BCM — solo activo en LEVEL3
  buzzer_active: false  # false = pasivo (PWM); true = activo (DC)
  motor_pin: 17         # BCM — activo en LEVEL2
  motor_active: true
  oled_address: "0x3C"

alert_state:
  bad_frame_threshold: 10   # frames malos consecutivos para disparar alerta
  good_frame_reset: 5       # frames buenos para restablecer a OK

logging:
  level: "INFO"
  csv_output: false
  csv_path: "ergotrack_log.csv"
```

---

## Sistema de alertas

### Niveles y salidas

| Nivel | Condición de postura | Salida física | Repetición |
|-------|---------------------|---------------|------------|
| **OK** | Postura correcta | OLED: cara feliz 😊 | — |
| **LEVEL 1** | Cuello leve / FHP leve / asimetría leve | OLED: cara seria 😐 | — |
| **LEVEL 2** | Cuello moderado (25–40°) / asimetría severa | Motor: 2 vibraciones | Cada 15 min |
| **LEVEL 3** | Cuello severo (>40°) / FHP severo | Buzzer: 3 bips | Cada 15 min |

### OLED

La pantalla alterna cada 5 segundos entre:
- **Página cara** — expresión facial según el nivel (😊 / 😐 / 😟)
- **Página datos** — métricas numéricas de cuello, FHP y asimetría

### Lógica de debounce

La alerta no se dispara en el primer frame malo. Requiere **10 frames consecutivos** con mala postura. Se restablece tras **5 frames buenos** consecutivos. Esto evita falsas alarmas por oclusiones momentáneas.

---

## Arquitectura

```
Camera / CameraSimulator
        │
        ▼
  VisionManager          MoveNet Lightning (TFLite)
  (hilo background)      192×192 → 17 keypoints COCO
        │
        ▼ PostureLandmarks
  PostureLogic
  ├── math_utils (cuello, FHP, asimetría)
  └── AlertStateMachine (debounce N/M frames)
        │
        ▼ PostureReport
  ┌─────┴──────┬───────────────┐
  ▼            ▼               ▼
Motor       Buzzer           OLED
(LEVEL2)   (LEVEL3)    (cara + datos)
```

### Hilos concurrentes

| Hilo | Función | Frecuencia |
|------|---------|-----------|
| Principal | Captura + display | 5 FPS |
| Inferencia BG | MoveNet | cada 2 frames |
| Motor BG | Pulso GPIO | on-demand |
| Buzzer BG | Pulso PWM | on-demand (LEVEL3) |
| ThermalGuard | Temperatura CPU | cada 10 s |

---

## Estructura del proyecto

```
ergotrack/
├── main.py                     # Punto de entrada
├── config/
│   └── default.yaml            # Configuración principal
├── models/
│   └── movenet_lightning.tflite
├── modules/
│   ├── config_profile.py       # Carga y hot-reload de YAML
│   ├── vision_manager.py       # Captura + inferencia MoveNet
│   ├── posture_logic.py        # Métricas + state machine
│   └── hardware_controller.py # GPIO motor/buzzer + OLED
├── gui/
│   └── ergo_dashboard.py       # Dashboard tkinter
├── utils/
│   ├── math_utils.py           # Cálculo de métricas de postura
│   ├── thermal_guard.py        # Monitoreo de temperatura CPU
│   ├── camera_simulator.py     # Landmarks sintéticos para pruebas
│   └── terminal_display.py     # Monitor ANSI en terminal
└── scripts/
    ├── download_models.py      # Descarga modelo MoveNet
    ├── test_hardware.py        # Diagnóstico de GPIO + OLED
    └── test_oled.py            # Diagnóstico detallado de OLED
```

---

## Requisitos

- Raspberry Pi 5 (Cortex-A76, RPi OS Bookworm)
- Python 3.11+ (pyenv recomendado)
- `ai-edge-litert` — inferencia TFLite sin MediaPipe
- `luma.oled` — driver SSD1306 por I2C
- `RPi.GPIO` / `rpi-lgpio` — control GPIO

Para desarrollo en PC (modo simulación):
- Python 3.9+
- `PyYAML`, `numpy`, `opencv-python`, `customtkinter`

---

## Licencia

MIT
