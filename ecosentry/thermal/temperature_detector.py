"""
Radiometric Temperature-Aware Hotspot Detection Module

Designed for calibrated radiometric thermal sensors (e.g. FLIR Lepton 3.5, MLX90640).
Accepts a 2D NumPy array containing true temperature values in Celsius (°C).

Scientific & Integrity Safeguards:
- Arbitrary RGB/BGR images or false-color GIFs are NEVER directly converted to Celsius.
- For testing and simulation, synthetic temperature matrices are clearly labelled
  as SYNTHETIC TEST DATA.
- The detection and ranking algorithms operate directly on physical temperature values.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np


def detect_temperature_hotspots(
    temperature_map: np.ndarray,
    threshold_c: float = 70.0,
    minimum_area: int = 15,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Detect abnormal temperature hotspots from a 2D calibrated Celsius matrix.

    Args:
        temperature_map: 2D NumPy array of temperature values in Celsius (°C).
        threshold_c: Temperature threshold in Celsius to flag an anomaly (default: 70.0°C).
        minimum_area: Minimum connected pixel area to eliminate noise artifacts (default: 15 px).

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray]:
            - List of detected hotspot dictionaries ranked by max_temperature_c (descending).
            - Binary mask (uint8) of thresholded regions.
    """
    if temperature_map is None:
        raise ValueError("Temperature map cannot be None")

    temperature_map = np.asarray(temperature_map, dtype=np.float32)

    if temperature_map.ndim != 2:
        raise ValueError(
            f"Temperature map must be a 2D array, got shape {temperature_map.shape} with {temperature_map.ndim} dimensions"
        )

    # 1. Segment regions exceeding the Celsius threshold
    mask = (temperature_map >= threshold_c).astype(np.uint8) * 255

    # 2. Morphological filtering to eliminate single-pixel noise spikes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 3. Find connected candidate contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspots: List[Dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        region = temperature_map[y : y + height, x : x + width]

        if region.size == 0:
            continue

        max_temp = float(np.max(region))
        mean_temp = float(np.mean(region))

        hotspots.append({
            "bbox": [int(x), int(y), int(width), int(height)],
            "area": area,
            "max_temperature_c": round(max_temp, 2),
            "mean_temperature_c": round(mean_temp, 2),
            "threshold_c": float(threshold_c),
        })

    # 4. Rank hotspots by peak maximum temperature (hottest first)
    hotspots.sort(key=lambda hs: hs["max_temperature_c"], reverse=True)

    return hotspots, mask


def generate_synthetic_temperature_map(
    shape: Tuple[int, int] = (480, 640),
    ambient_c: float = 32.0,
    hotspot_specs: Optional[List[Dict[str, Any]]] = None,
    noise_std: float = 1.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate deterministic synthetic 2D temperature matrices for unit tests
    and software demonstration.

    NOTE: This is strictly for software testing until physical radiometric hardware
    (e.g., FLIR Lepton / MLX90640) is connected.
    """
    if seed is not None:
        np.random.seed(seed)

    rows, cols = shape
    temp_map = np.full((rows, cols), ambient_c, dtype=np.float32)

    # Add subtle ambient thermal noise
    noise = np.random.normal(0.0, noise_std, (rows, cols)).astype(np.float32)
    temp_map += noise

    if hotspot_specs is None:
        # Default representative scenario
        hotspot_specs = [
            {"center": (150, 200), "radius": 35, "peak_c": 125.0},
            {"center": (320, 450), "radius": 45, "peak_c": 285.0},
            {"center": (360, 180), "radius": 25, "peak_c": 82.0},
        ]

    y_indices, x_indices = np.indices((rows, cols))

    for spec in hotspot_specs:
        cy, cx = spec["center"]
        radius = spec["radius"]
        peak_c = spec["peak_c"]

        # Gaussian radial temperature profile
        dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
        sigma_sq = (radius / 2.0) ** 2
        added_temp = (peak_c - ambient_c) * np.exp(-dist_sq / (2.0 * sigma_sq))
        temp_map = np.maximum(temp_map, ambient_c + added_temp)

    return temp_map.astype(np.float32)


def render_temperature_visualization(
    temperature_map: np.ndarray,
    hotspots: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    is_synthetic: bool = True,
) -> np.ndarray:
    """
    Render a false-color representation of the temperature matrix with
    hotspot annotations and calibrated °C labels.

    Args:
        temperature_map: 2D float array in Celsius.
        hotspots: List of hotspot dicts from detect_temperature_hotspots.
        output_path: Optional path to save resulting PNG.
        is_synthetic: Whether to label as synthetic test data.

    Returns:
        np.ndarray: BGR colorized visualization image.
    """
    t_min = float(np.min(temperature_map))
    t_max = float(np.max(temperature_map))
    t_range = max(1.0, t_max - t_min)

    # Normalize to [0, 255] for colormap rendering
    normalized = np.clip((temperature_map - t_min) / t_range * 255.0, 0, 255).astype(np.uint8)
    color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

    strongest = hotspots[0] if hotspots else None

    # Draw all detected hotspots
    for idx, hs in enumerate(hotspots, start=1):
        x, y, w, h = hs["bbox"]
        max_t = hs["max_temperature_c"]
        mean_t = hs["mean_temperature_c"]
        is_strongest = (hs == strongest)

        box_color = (0, 0, 255) if is_strongest else (0, 255, 255)  # Red for hottest, Yellow for others
        thickness = 3 if is_strongest else 2

        cv2.rectangle(color_img, (x, y), (x + w, y + h), box_color, thickness)

        label = f"#{idx}: {max_t:.1f} C"
        if is_strongest:
            label += " [HOTTEST]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        (tw, th), base = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y - 6, th + 4)
        cv2.rectangle(
            color_img,
            (x, label_y - th - 3),
            (x + tw + 6, label_y + base + 2),
            (15, 15, 15),
            -1,
        )
        cv2.putText(color_img, label, (x + 3, label_y), font, font_scale, box_color, 1, cv2.LINE_AA)

    # Top overlay header banner
    h, w = color_img.shape[:2]
    banner_h = 65
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)

    title_tag = "[SYNTHETIC TEST DATA] " if is_synthetic else ""
    banner_title = f"{title_tag}RADIOMETRIC TEMPERATURE ANALYSIS (CELSIUS)"

    stats_str = f"Hotspots: {len(hotspots)}"
    if strongest:
        stats_str += f" | Peak: {strongest['max_temperature_c']:.1f} C (Mean: {strongest['mean_temperature_c']:.1f} C) | Threshold: {strongest['threshold_c']:.1f} C"
    else:
        stats_str += f" | Scene Min: {t_min:.1f} C, Max: {t_max:.1f} C (Below threshold)"

    cv2.putText(banner, banner_title, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        banner,
        stats_str,
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 165, 255) if strongest else (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    composite = np.vstack([banner, color_img])

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), composite)

    return composite
