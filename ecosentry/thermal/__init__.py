"""
ECOSENTRY Thermal Processing Module

Supports dual analysis paths:
1. Radiometric Temperature Path:
   True physical temperature matrix in Celsius (°C) from calibrated radiometric hardware
   (FLIR Lepton, MLX90640).
2. Demo Thermal Image Path:
   Relative thermal anomaly extraction and intensity standardization from false-color frames/GIFs.
"""

from .preprocessor import ThermalPreprocessor, preprocess_thermal_frame
from .hotspot_detector import (
    HotspotDetector,
    HotspotCandidate,
    detect_hotspot_candidates,
    detect_hotspots,
    draw_hotspot_candidates,
)
from .temperature_detector import (
    detect_temperature_hotspots,
    generate_synthetic_temperature_map,
    render_temperature_visualization,
)

__all__ = [
    "ThermalPreprocessor",
    "preprocess_thermal_frame",
    "HotspotDetector",
    "HotspotCandidate",
    "detect_hotspot_candidates",
    "detect_hotspots",
    "draw_hotspot_candidates",
    "detect_temperature_hotspots",
    "generate_synthetic_temperature_map",
    "render_temperature_visualization",
]
