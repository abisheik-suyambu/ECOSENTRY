# ECOSENTRY — Thermal Hotspot Detection

**ECOSENTRY** is an early forest-fire monitoring and alert system utilizing thermal infrared (TIR) computer vision and deep-learning object detection.

---

## 🎯 Project Objective

Early detection of forest fires relies on capturing incipient thermal anomalies—such as subterranean root fires, smoldering biomass, or small localized fire patches—before they erupt into uncontrollable canopy blazes.

Traditional standard object detection models often struggle with:
- **Small Target Scale**: Early thermal hotspots can occupy as few as $3 \times 3$ to $16 \times 16$ pixels from UAV or tower perspectives.
- **Feature Degradation**: Aggressive downsampling (strided convolutions and max-pooling) in standard CNN backbones can destroy small hotspot spatial signatures.
- **Thermal Noise & False Positives**: Solar reflections on bare rock or water surfaces can create thermal clutter.

ECOSENTRY establishes a robust detection foundation using a modern, verifiable YOLO architecture (such as YOLOv8/v9/v11) and incorporates small-object thermal techniques (such as **SPD-Conv** and **CBAM attention**) to ensure high sensitivity to early fire hotspots.

---

## 📁 Project Structure

```text
ecosentry/
├── datasets/          # Dataset configs, annotation converters, and split manifests
├── models/            # Model definitions, custom layer modules (SPD-Conv, CBAM), and YAML configs
├── training/          # Training pipelines, hyperparameter definitions, and checkpoint managers
├── inference/         # Real-time stream detection, batch inference, and visualizers
├── evaluation/        # Validation scripts, mAP calculators, PR curves, and benchmark tools
├── thermal/           # Radiometric processing, 16-bit to 8-bit normalization, and pseudocolor mapping
├── results/           # Run outputs, generated predictions, checkpoints, and metrics logs
├── utils/             # Helper utilities (geometry, logging, alert triggers, drawing)
├── README.md          # Project documentation and operational guide
└── requirements.txt   # Verified Python runtime dependencies
```

### Detailed Directory Responsibilities

| Directory | Purpose & Contents |
| :--- | :--- |
| **`datasets/`** | Manages dataset YAML manifests (`data.yaml`), tools to organize images and YOLO `.txt` labels into `train/val/test` partitions, and download/conversion scripts for open thermal benchmarks (e.g., FLAME dataset). |
| **`models/`** | Contains neural network definitions and YAML architectures. Hosts custom layers such as Space-to-Depth (`SPD-Conv`) to preserve small hotspot features and Convolutional Block Attention Modules (`CBAM`) to suppress background noise. |
| **`training/`** | Scripts for model training (`train.py`), hyperparameter sweeps, loss configurations (CIoU / DFL / Focal Loss), and device management (CUDA / MPS / CPU). |
| **`inference/`** | Scripts for running detection on static images, video files, and real-time UAV/RTSP video streams (`detect.py`), with bounding box drawing and fire threshold alerts. |
| **`evaluation/`** | Independent scripts to validate trained weights against unseen test sets (`val.py`), generating genuine metrics including mAP@50, mAP@50:95, Precision, Recall, and Confusion Matrices. |
| **`thermal/`** | Domain-specific thermal utilities: handling 16-bit raw radiometric TIFFs/arrays, ambient-to-Celsius conversion, threshold isotherm masking, and false-color conversions (Ironbow, Rainbow, White-Hot). |
| **`results/`** | Destination directory for test inferences, heatmaps, training log runs, and model checkpoints. *(Ignored by version control for large weights).* |
| **`utils/`** | General-purpose helper functions: logging formatters, IoU calculations, geometry utilities, and automated notification triggers. |

---

## ⚙️ Environment Setup

1. **Create and activate a Python virtual environment** (recommended Python 3.10+):
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

2. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📌 Implementation Status

- [x] Initial project directory structure established.
- [x] Runtime requirements defined (`requirements.txt`).
- [ ] Thermal preprocessing module implementation (`thermal/`).
- [ ] YOLO base model configuration with thermal attention (`models/`).
- [ ] Dataset ingestion and annotation pipeline (`datasets/`).
- [ ] Model training pipeline setup (`training/`).
- [ ] Real-time inference and alert pipeline (`inference/`).
- [ ] Independent validation and metric evaluation (`evaluation/`).

> [!NOTE]
> No training has been executed yet, and no weights or metrics are claimed. The next phase involves preparing the thermal preprocessing pipeline and configuring the YOLO model baseline.
