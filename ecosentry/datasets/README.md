# ECOSENTRY Dataset Preparation Guide

This directory manages ground-truth dataset assets for training a genuine, single-class YOLO detection model for:
```text
class 0: thermal_hotspot
```

---

## 📁 Directory Structure

```text
ecosentry/datasets/
├── sample_thermal/      # Visual prototype frames (25 uncalibrated demonstration frames)
├── real_thermal/        # Ingested authentic thermal/infrared images (TIFF / PNG / Radiometric)
├── annotations/         # Raw source ground-truth labels and conversion manifests
├── train/               # YOLO training split (images/ and labels/)
├── val/                 # YOLO validation split (images/ and labels/)
├── test/                # Unseen evaluation split (images/ and labels/)
└── README.md            # Dataset specification and requirements
```

---

## 📋 Required Information for a Genuine Dataset

To achieve dependable, scientifically valid detection and prevent false alarms, the following five criteria must be satisfied before training:

### 1. Thermal / Infrared Images (`real_thermal/`)
- **Modality**: Long-Wave Infrared (LWIR), Mid-Wave Infrared (MWIR), or Radiometric Thermal IR.
- **Bit Depth & Formats**:
  - Raw 16-bit radiometric TIFF / NumPy array (`.npy`), or
  - Clean standardized 8-bit thermal visualizations (grayscale or standardized thermal colormaps like Ironbow/White-Hot).
- **Metadata**: Sensor model, resolution (e.g. 640×512, 1920×1080), flight altitude, and ambient environmental conditions (if available).

### 2. Verified Hotspot / Fire Regions
- **Ground Truth Integrity**: Every marked region must correspond to a verified thermal anomaly (such as smoldering biomass, early root fire, or defective PV cell hotspot).
- **False-Positive Hard Negatives**: The dataset must include non-fire thermal clutter (e.g., sun-glint on metal/water, hot rocks, animal heat signatures, running vehicle engines) explicitly labelled as background (no bounding box) to prevent model hallucination.

### 3. Bounding-Box Annotations (`annotations/`)
- **Format**: Standard YOLO annotation text files (`.txt` per image):
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
  Where coordinates are float values normalized to $[0.0, 1.0]$ relative to image dimensions.
- **Class Mapping**:
  ```yaml
  names:
    0: thermal_hotspot
  ```
- **Precision**: Boxes must tightly enclose the anomaly without enclosing excessive cool surrounding background.

### 4. Dataset License and Source Attribution
- **Provenance**: Verified academic or institutional dataset origin (e.g., FLAME Thermal Dataset, Corsican Fire, or authorized UAV field survey).
- **Licensing**: Permissive research or commercial license (e.g., CC BY 4.0, MIT, or institutional data transfer agreement).
- **Ethical & Safety Compliance**: Ensure no restricted operational or proprietary infrastructure data is exposed.

### 5. Train / Validation / Test Split Strategy
- **Partition Ratios**: Standard partition recommendation:
  - **70%** Training (`train/`)
  - **15%** Validation (`val/` — used for hyperparameter tuning & early stopping)
  - **15%** Test (`test/` — strictly unseen, used for final mAP evaluation)
- **Temporal & Flight-Level Independence**:
  - Consecutive frames from the same UAV flight video must **NOT** be split across train and test sets (to avoid data leakage and falsely inflated accuracy).
  - Whole flights or distinct geographic scenes must be assigned entirely to either train, val, or test.

---

> [!IMPORTANT]
> **Data Integrity Policy:**  
> The 25 demonstration frames in `sample_thermal/` are uncalibrated visual demonstration assets. They are strictly reserved for testing inference pipeline mechanics and must **never** be automatically pseudo-labelled or treated as ground-truth training data.
