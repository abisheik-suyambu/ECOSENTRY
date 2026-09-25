<div align="center">

# 🌿 ECOSENTRY

### *Resilient, AI-Powered Environmental Monitoring Network for Early Disaster Detection*

[![SIH Problem](https://img.shields.io/badge/SIH-26178-green?style=for-the-badge)](https://www.sih.gov.in/)
[![Theme](https://img.shields.io/badge/Theme-Disaster%20Management-red?style=for-the-badge)](https://www.sih.gov.in/)
[![Category](https://img.shields.io/badge/Category-Hardware%20%26%20Edge%20AI-blue?style=for-the-badge)](https://www.sih.gov.in/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-purple?style=for-the-badge)](https://ultralytics.com/)

> **Forests cannot speak. ECOSENTRY listens.**
>
> A network of intelligent sensor nodes deployed across dense forests — **10 km apart** — working silently 24/7 to detect the earliest signs of wildfire, long before flames can spread.

</div>

---

## 🔥 The Problem

Every year, **millions of hectares of forest** are destroyed by wildfires that go undetected until it is too late. Traditional monitoring methods — satellite imagery, fire watchtowers, manual patrolling — all suffer from:

| Problem | Impact |
|---|---|
| ⏱️ **Detection Delay** | Satellites pass every 1–3 hours; fires spread in minutes |
| 🌫️ **Smoke Blindness** | Visual cameras fail in heavy smoke |
| 📡 **No Infrastructure** | Remote forests have zero communication coverage |
| ❌ **False Alarms** | Sun-heated rocks, vehicles trigger costly false positives |
| 🧭 **No Location Precision** | Alerts without GPS coordinates delay response |

**ECOSENTRY solves all of this** with a resilient mesh of fixed sensor nodes, LoRa radio telemetry, and AI-powered drone verification.

---

## 💡 How ECOSENTRY Works — Three Layers

```
🌳 Ground Layer  →  Fixed Sensor Nodes (every 10 km across the forest)
📡 Radio Layer   →  LoRa Long-Range Wireless Telemetry (no internet needed)
🚁 Sky Layer     →  AI Drone with Thermal + Visual Cameras
```

### Complete System Flow

```mermaid
flowchart TD
    A["🌡️ Sensor Node\n(ESP32 + Sensors)\nFixed every 10 km in forest"] -->|"LoRa Radio\n10–15 km range"| B["📡 LoRa Gateway\n(Base Station)"]
    B -->|"Anomaly Detected!\nSend GPS Coordinates"| C["🚁 Drone Command\n(Launch Investigation)"]
    C --> D["🛸 Autonomous Drone\n(Flies to GPS Location)"]
    D --> E["🌡️ Thermal Camera\n(Infrared TIR)"]
    D --> F["📷 Visual Camera\n(RGB)"]
    E --> G["🔥 Hotspot Detector\n(Statistical Analysis)"]
    F --> H["🤖 YOLOv8 AI\n(Fire & Smoke Detection)"]
    G --> I["⚖️ Evidence Fusion\n(Cross-Verification)"]
    H --> I
    I -->|"Verified Alert"| J["📊 Dashboard\n(Real-time Map + Alerts)"]
    J --> K["🚨 Emergency Response\n(Forest Department)"]

    style A fill:#2d6a4f,color:#fff,stroke:#1b4332
    style B fill:#1d3557,color:#fff,stroke:#0d2136
    style C fill:#e63946,color:#fff,stroke:#c1121f
    style D fill:#457b9d,color:#fff,stroke:#1d3557
    style E fill:#f4a261,color:#fff,stroke:#e76f51
    style F fill:#f4a261,color:#fff,stroke:#e76f51
    style G fill:#e9c46a,color:#000,stroke:#f4a261
    style H fill:#e9c46a,color:#000,stroke:#f4a261
    style I fill:#264653,color:#fff,stroke:#2a9d8f
    style J fill:#2a9d8f,color:#fff,stroke:#264653
    style K fill:#e63946,color:#fff,stroke:#c1121f
```

---

## 🗺️ Node Network — Deployment Concept

Each sensor node covers roughly a **10 km radius**. Nodes are deployed in a grid so there are **zero gaps** in monitoring coverage.

```mermaid
graph LR
    subgraph FOREST["🌲 Forest Zone — Node Deployment Grid (10 km spacing)"]
        N1["📍 Node 01\n12.97N, 80.22E"]
        N2["📍 Node 02\n13.07N, 80.22E"]
        N3["📍 Node 03\n12.97N, 80.31E"]
        N4["📍 Node 04\n13.07N, 80.31E"]
        N5["📍 Node 05\n12.87N, 80.22E"]
        GW["📡 Gateway\nBase Station"]
    end

    N1 -->|"LoRa 10 km"| GW
    N2 -->|"LoRa 10 km"| GW
    N3 -->|"LoRa 13 km"| GW
    N4 -->|"LoRa 15 km"| GW
    N5 -->|"LoRa 10 km"| GW
    GW -->|"Internet / 4G"| DASH["☁️ Cloud Dashboard"]

    style N1 fill:#2d6a4f,color:#fff
    style N2 fill:#2d6a4f,color:#fff
    style N3 fill:#2d6a4f,color:#fff
    style N4 fill:#2d6a4f,color:#fff
    style N5 fill:#2d6a4f,color:#fff
    style GW fill:#e63946,color:#fff
    style DASH fill:#457b9d,color:#fff
```

> **Coverage Calculation:** Each node covers ~314 km² (π × 10²). A 5-node grid protects over **1,500 km²** of forest with overlapping coverage for redundancy.

---

## 🧱 System Architecture Overview

```mermaid
graph TB
    subgraph GROUND["🌳 Ground Layer — Fixed Sensor Nodes"]
        SN["ESP32 Microcontroller\nTemp | Gas | Soil Moisture | GPS\nSolar Powered | IP67 Weatherproof"]
    end

    subgraph COMM["📡 Communication Layer"]
        GW["LoRa Gateway\nReceives packets from all nodes\nFilters anomalies"]
    end

    subgraph AERIAL["🚁 Aerial Investigation Layer"]
        DR["Autonomous Drone\nGPS Waypoint Navigation"]
        RPI["Raspberry Pi (Onboard)\nEdge AI Inference"]
    end

    subgraph INTEL["🧠 Intelligence Layer"]
        AI["AI Fusion Engine\nThermal + YOLO"]
        DB["Web Dashboard\nReal-time Map + Evidence"]
    end

    SN -->|"LoRa 433MHz"| GW
    GW -->|"MAVLink / WiFi"| DR
    DR -->|"Camera Feed"| RPI
    RPI -->|"Inference Results"| AI
    AI -->|"Verified Alert"| DB
    DB -->|"SMS / Email"| RESP["🚨 Forest Department"]

    style SN fill:#2d6a4f,color:#fff
    style GW fill:#1d3557,color:#fff
    style DR fill:#457b9d,color:#fff
    style RPI fill:#457b9d,color:#fff
    style AI fill:#264653,color:#fff
    style DB fill:#2a9d8f,color:#fff
    style RESP fill:#e63946,color:#fff
```

---

## ⚙️ Hardware Components

### 🌡️ Ground Sensor Node (one per deployment point)

| Component | Model | Purpose |
|---|---|---|
| Microcontroller | ESP32 DevKit V1 | Core processor, WiFi/BT capable |
| LoRa Radio | SX1278 / RFM95W | Long-range 10–15 km data transmission |
| Temperature Sensor | DHT22 / DS18B20 | Ambient temperature monitoring |
| Gas Sensor | MQ-2 / MQ-135 | Smoke, CO, LPG detection |
| Soil Moisture | Capacitive Sensor | Dry vegetation risk assessment |
| GPS Module | NEO-6M | Precise node location tagging |
| Power | Solar Panel + LiPo | Off-grid continuous operation |
| Enclosure | IP67 Waterproof Box | Outdoor forest deployment |

### 🚁 Aerial Investigation Unit (Drone)

| Component | Model | Purpose |
|---|---|---|
| Flight Controller | Pixhawk 4 / ArduPilot | Autonomous GPS waypoint navigation |
| Companion Computer | Raspberry Pi 4 / 5 | Runs AI inference at the edge |
| Thermal Camera | MLX90640 / FLIR Lepton 3.5 | Infrared hotspot detection |
| Visual Camera | Pi Camera v3 | RGB fire/smoke YOLO input |
| Connectivity | 4G LTE Module | Real-time data uplink to dashboard |

---

## 🧠 AI Detection Pipeline

```mermaid
flowchart LR
    subgraph INPUT["📥 Drone Capture"]
        TC["🌡️ Thermal Frame\nInfrared TIR"]
        VC["📷 Visual Frame\nRGB"]
    end

    subgraph THERM["🔥 Thermal Processing"]
        T1["Intensity Normalization\nCLAHE"]
        T2["Statistical Thresholding\nu + 2.5s above background"]
        T3["Hotspot Ranking\nBBox + Peak Intensity Delta"]
        TC --> T1 --> T2 --> T3
    end

    subgraph YOLO["🤖 YOLOv8 Visual AI"]
        Y1["Load best.pt\nFire and Smoke Model"]
        Y2["Run Inference\nconf threshold 0.25"]
        Y3["Output Bounding Boxes\nFire or Smoke class"]
        VC --> Y1 --> Y2 --> Y3
    end

    subgraph FUSION["⚖️ Evidence Fusion"]
        F1["Spatial IoU Check\nDoes hotspot overlap YOLO box?"]
        T3 --> F1
        Y3 --> F1
    end

    F1 -->|"Both Confirmed"| AL1["🚨 FIRE / SMOKE ALERT\nHigh Confidence — Dispatch Response"]
    F1 -->|"Thermal Only"| AL2["⚠️ THERMAL ANOMALY\nInvestigate Further"]
    F1 -->|"Visual Only"| AL3["⚠️ POSSIBLE FIRE\nAwaiting Thermal Confirmation"]
    F1 -->|"Neither"| AL4["✅ ALL CLEAR\nNo Threat Detected"]

    style AL1 fill:#e63946,color:#fff
    style AL2 fill:#f4a261,color:#000
    style AL3 fill:#e9c46a,color:#000
    style AL4 fill:#2d6a4f,color:#fff
```

### Alert Levels

| Alert Level | Meaning | Action Required |
|---|---|---|
| ✅ **ALL CLEAR** | All sensors normal | Continue monitoring |
| ⚠️ **WARNING** | Ground sensors flagged, drone shows nothing | Watch closely |
| 🌡️ **THERMAL ANOMALY** | Heat detected, no visible flame | Could be hot rock or machinery |
| 🔥 **POSSIBLE FIRE** | Visual smoke/fire without thermal confirmation | Deploy ranger patrol |
| 🚨 **FIRE / SMOKE ALERT** | Both thermal AND visual confirmed | Immediate emergency response |

---

## 📡 Why LoRa — The Smart Radio Choice for Forests

```mermaid
mindmap
  root((LoRa Radio))
    Range
      10 to 15 km open terrain
      5 to 8 km dense forest
    Power
      Under 40 mA transmit current
      Years of battery life
    Cost
      Under 5 USD per module
      No SIM card needed
    Reliability
      No internet dependency at node
      Works in remote areas
    Network
      Star topology
      All nodes to one gateway
```

**Sensor data packet (sent every 60 seconds):**
```json
{
  "node_id": "NODE_01",
  "latitude": 12.9716,
  "longitude": 80.2209,
  "temperature_c": 68.5,
  "gas_level": 82.0,
  "soil_moisture_pct": 15.0,
  "timestamp": "2026-09-25T10:30:00Z",
  "alert": true
}
```

---

## 🚀 Quick Start — Run the Demo

> No hardware needed! Run the complete system simulation on any laptop.

### Step 1 — Install Dependencies
```bash
git clone https://github.com/your-username/ecosentry.git
cd ecosentry
python -m pip install -r requirements.txt
```

### Step 2 — Verify All Components
```bash
python tests/test_components.py
```

### Step 3 — Run Full End-to-End Simulation
```bash
python simulation/run_demo.py
```

This simulates the complete flow:
- Sensor nodes transmit LoRa packets
- Gateway detects anomaly at Node 01
- Drone dispatched to GPS coordinates
- Thermal + YOLO AI runs on captured frames
- Evidence fusion generates verified alert
- Results saved to `results/`

### Step 4 — Launch Live Dashboard
```bash
python ecosentry/dashboard/app.py
```
Open **http://localhost:5000** to view live telemetry, bounding boxes, and evidence images.

---

## 📁 Repository Structure

```
ecosentry/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
│
├── ecosentry/                   ← Core Python package
│   ├── sensor/                  ← ESP32 node simulation
│   │   └── sensor_node.py
│   ├── gateway/                 ← LoRa gateway simulation
│   │   └── gateway.py
│   ├── drone/                   ← Drone + camera simulation
│   │   ├── drone_controller.py
│   │   └── camera.py
│   ├── thermal/                 ← Infrared hotspot detection
│   │   ├── preprocessor.py
│   │   └── hotspot_detector.py
│   ├── yolo/                    ← YOLOv8 fire/smoke detector
│   │   └── fire_detector.py
│   ├── verification/            ← Spatial cross-validation
│   │   └── fire_verifier.py
│   ├── fusion/                  ← Multi-modal evidence fusion
│   │   └── evidence_fusion.py
│   ├── alerts/                  ← Alert builder and formatter
│   │   └── alert_manager.py
│   ├── dashboard/               ← Flask web dashboard
│   │   ├── app.py
│   │   └── templates/index.html
│   └── models/
│       └── best.pt              ← Pre-trained YOLO weights
│
├── simulation/                  ← Full demo runner
│   └── run_demo.py
│
├── tests/                       ← Component test suite
│   └── test_components.py
│
├── results/                     ← Generated alert evidence output
│   ├── latest_alert.json
│   ├── composite_evidence.png
│   ├── drone_thermal_evidence.png
│   └── yolo_fire_detection.png
│
├── examples/                    ← Demo input media
│   └── hotspot-detect.gif
│
└── docs/                        ← Technical documentation
    ├── architecture.md
    └── workflow.md
```

---

## 📊 Technical Specifications

| Parameter | Value |
|---|---|
| Node Spacing | 10 km apart |
| LoRa Range | 10–15 km open, 5–8 km forested |
| Coverage per Node | ~314 km² (π × 10²) |
| Sensor Read Interval | Every 60 seconds |
| Alert Latency | Less than 2 minutes (trigger to alert) |
| AI Model | YOLOv8 Nano — edge optimized |
| Detection Classes | Fire, Smoke |
| YOLO Confidence Threshold | 0.25 |
| Thermal Threshold | Mean + 2.5 × Standard Deviation |
| Node Power | ~200 mW average (solar powered) |
| Operating Temperature | -10°C to +60°C |
| Communication | LoRa 433 MHz / 915 MHz |

---

## 🌐 Development Roadmap

```mermaid
gantt
    title ECOSENTRY Development Roadmap
    dateFormat  YYYY-MM
    section Phase 1 - Software
        Core AI Pipeline              :done,    2026-07, 2026-09
        Dashboard and Simulation      :done,    2026-08, 2026-09
        Test Suite and Validation     :done,    2026-09, 2026-09
    section Phase 2 - Hardware Prototype
        ESP32 Node Firmware           :active,  2026-10, 2026-11
        LoRa Gateway Setup            :         2026-10, 2026-11
        Drone Integration             :         2026-11, 2026-12
    section Phase 3 - Field Deployment
        Pilot Forest Deployment       :         2027-01, 2027-03
        Scale to 50 plus nodes        :         2027-03, 2027-06
        Satellite Sync Integration    :         2027-06, 2027-09
```

### Hardware Integration Path

| Phase | Task | Technology |
|---|---|---|
| Phase 2 — Nodes | Flash sensor logic onto ESP32 | Arduino IDE + FreeRTOS |
| Phase 2 — Radio | Connect SX1278 LoRa transceiver via SPI | MicroPython / Arduino |
| Phase 2 — Gateway | Raspberry Pi LoRa packet receiver | Python + PySerial |
| Phase 2 — Drone | Interface with ArduPilot flight controller | MAVLink / pymavlink |
| Phase 3 — Thermal | Connect MLX90640 or FLIR Lepton camera | I2C / SPI |
| Phase 3 — Cloud | MQTT broker for multi-gateway uplink | AWS IoT / Mosquitto |

---

## 🤝 Attribution

- **Project:** Smart India Hackathon 2026 — Problem Statement SIH26178
- **Theme:** Disaster Management
- **Category:** Hardware and Edge AI
- **YOLO Weights:** [luminous0219/fire-and-smoke-detection-yolov8](https://github.com/luminous0219/fire-and-smoke-detection-yolov8)
- **AI Framework:** [Ultralytics YOLOv8](https://ultralytics.com/)

---

<div align="center">

**Built with love for India's Forests**

*"Detect early. Respond fast. Save forests."*

</div>
