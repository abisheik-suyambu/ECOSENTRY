# 🌡️ Radiometric Temperature Pipeline Documentation

## 1. Overview & Problem Definition

Thermal infrared (TIR) cameras observe radiation in the Long-Wave Infrared (LWIR, 8–14 µm) spectrum. In computer vision disaster monitoring, there is a fundamental distinction between:

1. **Uncalibrated / False-Color Thermal Images:**
   - Standard 8-bit RGB/grayscale imagery or colormapped GIFs (e.g. Ironbow, Rainbow, Jet).
   - Pixel values represent relative visual display intensity, not calibrated physical temperature.
   - **Scientific Integrity Principle:** Arbitrary RGB/BGR thermal visualizations **cannot** be converted into calibrated degrees Celsius without sensor-specific calibration equations, emissivity settings, ambient temperature compensation, and raw sensor ADC counts.

2. **Radiometric Thermal Data (Calibrated Celsius Matrix):**
   - High-precision sensor output where each pixel $(y, x)$ represents an actual temperature reading in degrees Celsius ($^\circ\text{C}$).
   - Examples of target physical sensors: **FLIR Lepton 3.5** (radiometric output in centikelvin), **MLX90640** (factory-calibrated $32 \times 24$ I2C thermopile array).

ECOSENTRY establishes two explicitly decoupled processing pipelines to ensure scientific integrity:
- **Radiometric Temperature Path** (`ecosentry/thermal/temperature_detector.py`): Operates on 2D calibrated temperature matrices in $^\circ\text{C}$.
- **Demo Thermal Anomaly Path** (`ecosentry/thermal/hotspot_detector.py`): Operates on false-color demonstration frames to extract statistical relative intensity deltas ($\Delta I$).

---

## 2. Radiometric Temperature Algorithm

The core temperature detection algorithm is implemented in [`ecosentry/thermal/temperature_detector.py`](file:///c:/Users/Kiruthika/OneDrive/Desktop/ghxg/ecosentry/thermal/temperature_detector.py) through the function `detect_temperature_hotspots()`:

```text
2D Temperature Matrix in °C [H x W]
                │
                ▼
       Matrix Validation (2D, float32, non-null)
                │
                ▼
   Threshold Segmentation (temp >= threshold_c)
                │
                ▼
    Morphological Clean (3x3 Close & Open)
                │
                ▼
     Contour Extraction & Bounding Boxes
                │
                ▼
  Region Metrics Calculation (Max °C, Mean °C, Area)
                │
                ▼
   Rank Hotspots (Descending by Peak Temperature)
```

### Steps in the Pipeline:
1. **Input Validation:** Ensures the input matrix is a valid 2D NumPy array with floating-point temperature values.
2. **Threshold Segmentation:**
   $$\text{Mask}(y, x) = \begin{cases} 255 & \text{if } T(y, x) \ge T_{\text{threshold}} \\ 0 & \text{otherwise} \end{cases}$$
   The threshold is fully configurable (default: $70.0^\circ\text{C}$ for incipient thermal anomalies, or higher for active open blazes).
3. **Morphological Noise Suppression:** Applies a $3 \times 3$ morphological closing and opening filter to eliminate isolated single-pixel sensor noise spikes.
4. **Connected Region Extraction:** Extracts external connected contours to isolate spatially distinct thermal anomalies.
5. **Quantitative Property Extraction:**
   - **Bounding Box:** $[x, y, \text{width}, \text{height}]$
   - **Area:** Number of pixels exceeding threshold in the connected component.
   - **Peak Temperature ($T_{\text{max}}$):** Highest recorded Celsius reading within the anomaly.
   - **Mean Temperature ($T_{\text{mean}}$):** Average Celsius reading across the anomaly area.
   - **Threshold Reference ($T_{\text{threshold}}$):** The threshold applied during detection.
6. **Hotspot Ranking:** All isolated hotspots are sorted in descending order of $T_{\text{max}}$, identifying the primary heat hazard immediately.

---

## 3. Structured Output Schema

The detector returns a structured Python dictionary for each detected hotspot:

```json
{
    "bbox": [391, 261, 99, 99],
    "area": 7522.0,
    "max_temperature_c": 287.4,
    "mean_temperature_c": 126.1,
    "threshold_c": 70.0
}
```

---

## 4. Temperature Visualization & Evidence Generation

The visualization module converts the physical Celsius matrix into an auditable BGR evidence image:
- **Normalization:** Scales physical temperature range to $[0, 255]$.
- **False-Color Mapping:** Applies the `cv2.COLORMAP_INFERNO` palette (black $\to$ purple $\to$ orange $\to$ yellow).
- **Bounding Boxes & Overlay:**
  - Highlights the hottest hotspot with a thick red border and `[HOTTEST]` tag.
  - Draws secondary hotspots in yellow.
  - Annotates each box with peak temperature (e.g. `287.4 °C`).
  - Prepends an auditable telemetry banner displaying hotspot count, peak temperature, mean temperature, and threshold.
- **Saved Artifact:** Saved automatically to [`results/thermal_temperature_evidence.png`](file:///c:/Users/Kiruthika/OneDrive/Desktop/ghxg/results/thermal_temperature_evidence.png).

---

## 5. Synthetic Temperature Generation for Testing

Until physical radiometric hardware is connected to the drone companion computer, unit tests and demonstration runs utilize the deterministic generator:
```python
from ecosentry.thermal.temperature_detector import generate_synthetic_temperature_map

temp_map = generate_synthetic_temperature_map(
    shape=(480, 640),
    ambient_c=34.0,
    hotspot_specs=[
        {"center": (160, 220), "radius": 40, "peak_c": 128.5},
        {"center": (310, 440), "radius": 50, "peak_c": 287.4},
        {"center": (350, 160), "radius": 30, "peak_c": 84.2}
    ],
    seed=42
)
```
Every synthetic image and report is explicitly marked with `[SYNTHETIC TEST DATA]` to prevent misrepresentation.

---

## 6. Future Radiometric Hardware Integration

Connecting a physical radiometric camera requires zero changes to the hotspot detection logic. The camera driver simply provides the 2D NumPy array:

```python
# Example future hardware integration snippet:
from flirpy.camera.lepton import Lepton
from ecosentry.thermal.temperature_detector import detect_temperature_hotspots

# 1. Acquire raw radiometric frame from Lepton 3.5 (centikelvin)
camera = Lepton()
raw_kelvin = camera.grab()  # Values in centikelvin (e.g. 30015 = 300.15 K)

# 2. Convert to Celsius
celsius_map = (raw_kelvin / 100.0) - 273.15

# 3. Direct execution through existing ECOSENTRY detector
hotspots, mask = detect_temperature_hotspots(celsius_map, threshold_c=70.0)
```
This clean modular boundary guarantees effortless transition from software prototype to physical UAV hardware.
