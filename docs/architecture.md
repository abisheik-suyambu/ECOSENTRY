# 🏗️ ECOSENTRY System Architecture

**Project Title:** ECOSENTRY — Resilient, AI-Powered Environmental Monitoring Network for Early Disaster Detection  
**Problem Statement:** SIH26178 — Resilient, AI-Powered Environmental Monitoring Network for Early Disaster Detection  
**Theme:** Disaster Management  
**Category:** Hardware & Edge AI  

---

## 1. High-Level Architectural Diagram

```text
┌────────────────────────────────────────────────────────┐
│               TIER 1: GROUND SENSOR NODES              │
│       ESP32 + Environmental Sensors (Temp/Gas/Water)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ Simulated LoRa Telemetry (RFM95 / SX1278)
                            ▼
┌────────────────────────────────────────────────────────┐
│               TIER 2: LoRa BASE GATEWAY                │
│    Telemetry Ingestion, Anomaly Detection & Dispatch   │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ UAV Investigation Command (MAVLink-ready)
                            ▼
┌────────────────────────────────────────────────────────┐
│           TIER 3: AUTONOMOUS UAV / DRONE               │
│   Flight Navigation to Coordinates + Dual Capture      │
└─────────────┬────────────────────────────┬─────────────┘
              │ Synchronized TIR           │ Synchronized RGB
              ▼                            ▼
┌────────────────────────────┐ ┌───────────────────────────┐
│ TIER 4A: THERMAL PIPELINES │ │ TIER 4B: VISUAL AI (YOLO) │
│ 1. Radiometric Celsius Path│ │ - Ultralytics YOLOv8 Nano │
│    - 2D Temp Matrix in °C  │ │ - Model: best.pt          │
│    - Adaptive Hotspots     │ │ - Classes: Fire & Smoke   │
│ 2. Demo Relative TIR Path  │ │ - Genuine Bounding Boxes  │
│    - Standard Intensity    │ │                           │
│    - CLAHE Contrast Boost  │ │                           │
└─────────────┬──────────────┘ └───────────┬───────────────┘
              │                            │
              └─────────────┬──────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│      TIER 5: EVIDENCE FUSION & SPATIAL VERIFICATION    │
│  - Cross-correlates Hotspot coordinates with YOLO boxes│
│  - Fuses Ground Sensors + Radiometric Temp + YOLO Confs│
│  - Assesses Potential Vegetation / Biomass Fire Risk   │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│          TIER 6: ALERTS & REAL-TIME DASHBOARD          │
│  - Structured JSON Dispatch (results/latest_alert.json)│
│  - Multi-panel Evidence (results/*_evidence.png)       │
│  - Live Flask Disaster Monitoring Command Center UI    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Component Subsystems

### 1. Ground Sensor Network (`ecosentry/sensor/`)
- **Simulated Hardware:** ESP32 Microcontroller + DHT22/DS18B20 (Temperature) + MQ-2/MQ-135 (Gas & Smoke) + Capacitive Soil Moisture / Water Level Sensor.
- **Responsibility:** Emits periodic telemetry packets over low-bandwidth LoRa. When abnormal thresholds are exceeded, triggers an immediate emergency alert packet.

### 2. LoRa Gateway (`ecosentry/gateway/`)
- **Simulated Hardware:** Gateway base station with SX1278 LoRa transceiver.
- **Responsibility:** Ingests radio packets, verifies threshold breaches, logs telemetry, and formulates an `INVESTIGATE` command containing target latitude and longitude.

### 3. Drone Controller & Dual Camera (`ecosentry/drone/`)
- **Simulated Hardware:** Quadcopter / Hexacopter UAV with ArduPilot flight controller and Raspberry Pi edge companion computer.
- **Payload:** Synchronized Thermal Infrared (TIR) camera (e.g. FLIR Lepton) + RGB visual camera.
- **Responsibility:** Simulates waypoint navigation to ground coordinates and extracts synchronized frame pairs from demonstration composite inputs (`examples/hotspot-detect.gif`).

### 4. Thermal Processing Pathways (`ecosentry/thermal/`)
The system strictly enforces two decoupled pathways:
- **A. Radiometric Temperature Path (`temperature_detector.py`):**
  - Operates on calibrated 2D matrices in degrees Celsius ($^\circ\text{C}$).
  - Performs matrix validation, configurable temperature thresholding ($T \ge 70.0^\circ\text{C}$), morphological noise reduction, contour bounding boxes, and ranks hotspots by peak temperature.
- **B. Demo Relative Anomaly Path (`hotspot_detector.py`):**
  - Converts false-color thermal video/GIFs to standardized intensity maps, applies CLAHE contrast optimization, estimates background ambient baseline statistically ($\mu + k \cdot \sigma$), isolates thermal anomalies, and calculates intensity delta ($\Delta I$).
  - **Scientific Integrity:** Uncalibrated thermal visualization intensity is strictly preserved as intensity delta ($\Delta I$) and is **never** falsely claimed as calibrated degrees Celsius without radiometric calibration metadata.

### 5. YOLOv8 Visual AI (`ecosentry/yolo/`)
- **Engine:** Ultralytics YOLOv8 Nano edge model (`ecosentry/models/best.pt`).
- **Target Classes:** `0: fire`, `1: smoke`.
- **Responsibility:** Performs real-time inference on imagery, returning genuine bounding boxes and confidence scores.

### 6. Evidence Fusion & Spatial Verification (`ecosentry/fusion/` & `ecosentry/verification/`)
- **Responsibility:** Verifies spatial intersection between thermal hotspots and YOLO bounding boxes (IoU / IoH). Fuses ground sensor telemetry with aerial thermal and visual findings to eliminate false alarms (sunlight reflections, vehicle engines, heated metal roofs).

### 7. Alert Manager & Web Dashboard (`ecosentry/alerts/` & `ecosentry/dashboard/`)
- **Responsibility:** Saves auditable, timestamped evidence frames and structured JSON alerts. Hosts a lightweight Flask dashboard accessible at `http://localhost:5000` with live metrics, system state, and manual simulation triggers.

---

## 3. Future Hardware Migration Path

The code is architected to transition directly onto physical hardware:
- Replace `SensorNode.simulate_hotspot()` with Serial / SPI LoRa read from an ESP32.
- Replace `DroneController.fly_to_target()` with MAVLink commands via `dronekit-python` or `pymavlink` talking to ArduPilot / Pixhawk.
- Replace `DroneCamera.capture_frame()` with OpenCV `cv2.VideoCapture` streams from a USB thermal camera (MLX90640 / Seek / FLIR Lepton) and Raspberry Pi Camera Module.
- Run `ecosentry/yolo/fire_detector.py` directly on the Raspberry Pi 4/5 or Jetson Nano using PyTorch / ONNX Runtime.
