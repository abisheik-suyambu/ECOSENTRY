"""
Evidence Saver Module for ECOSENTRY

Generates and saves auditable visual verification artifacts:
- results/thermal_temperature_evidence.png (Radiometric Celsius Path)
- results/thermal_hotspot_evidence.png (Demo Thermal Anomaly Path)
- results/yolo_fire_detection.png (Visual AI Path)
- results/composite_evidence.png (Unified Command Center Summary)
- results/latest_alert.json (Structured Alert Payload)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import cv2
import numpy as np

from ecosentry.thermal.temperature_detector import render_temperature_visualization


def save_temperature_evidence(
    temperature_map: np.ndarray,
    temperature_hotspots: List[Dict[str, Any]],
    output_path: Union[str, Path] = "results/thermal_temperature_evidence.png",
    is_synthetic: bool = True,
) -> Path:
    """
    Render and save radiometric Celsius temperature evidence image.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    render_temperature_visualization(
        temperature_map=temperature_map,
        hotspots=temperature_hotspots,
        output_path=out_p,
        is_synthetic=is_synthetic,
    )
    return out_p


def save_thermal_evidence(
    thermal_frame: np.ndarray,
    hotspots: List[Dict[str, Any]],
    output_path: Union[str, Path] = "results/thermal_hotspot_evidence.png",
) -> Path:
    """
    Annotate thermal frame with hotspot bounding boxes and intensity delta.
    """
    img = thermal_frame.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    strongest = hotspots[0] if hotspots else None

    # Draw all detected hotspots
    for idx, hotspot in enumerate(hotspots, start=1):
        x, y, w, h = hotspot["bbox"]
        delta = hotspot.get("delta_intensity", 0.0)
        is_strongest = (hotspot == strongest)

        color = (0, 0, 255) if is_strongest else (0, 255, 255)  # Red for strongest, Yellow for others
        thickness = 3 if is_strongest else 2

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        label = f"Hotspot #{idx} (Delta: +{delta:.1f})"
        if is_strongest:
            label += " [STRONGEST]"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), base = cv2.getTextSize(label, font, font_scale, 1)
        label_y = max(y - 6, th + 4)
        cv2.rectangle(img, (x, label_y - th - 3), (x + tw + 6, label_y + base + 2), (20, 20, 20), -1)
        cv2.putText(img, label, (x + 3, label_y), font, font_scale, color, 1, cv2.LINE_AA)

    # Top overlay header banner
    banner_text = f"THERMAL ANOMALIES: {len(hotspots)}"
    if strongest:
        banner_text += f" | STRONGEST DELTA: +{strongest.get('delta_intensity', 0.0):.1f}"

    cv2.rectangle(img, (10, 10), (min(img.shape[1] - 10, 600), 50), (20, 20, 20), -1)
    cv2.putText(img, banner_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), img)

    # Also keep backward compatibility if output_path was thermal_hotspot_evidence
    alt_path = out_p.parent / "drone_thermal_evidence.png"
    if out_p.name != "drone_thermal_evidence.png":
        cv2.imwrite(str(alt_path), img)

    return out_p


def save_yolo_evidence(
    frame: np.ndarray,
    yolo_detections: List[Dict[str, Any]],
    output_path: Union[str, Path] = "results/yolo_fire_detection.png",
) -> Path:
    """
    Annotate frame with YOLO fire and smoke bounding boxes and confidence scores.
    """
    img = frame.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for det in yolo_detections:
        cls_name = det.get("class_name", "object")
        conf = det.get("confidence", 0.0)
        box = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in box]

        color = (0, 0, 255) if cls_name == "fire" else (255, 180, 0)  # Red for fire, Cyan for smoke
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        label = f"YOLO: {cls_name.upper()} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (tw, th), base = cv2.getTextSize(label, font, font_scale, 2)
        label_y = max(y1 - 8, th + 6)
        cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 6, label_y + base + 2), (20, 20, 20), -1)
        cv2.putText(img, label, (x1 + 3, label_y), font, font_scale, color, 2, cv2.LINE_AA)

    # Top overlay header banner
    fire_count = sum(1 for d in yolo_detections if d.get("class_name") == "fire")
    smoke_count = sum(1 for d in yolo_detections if d.get("class_name") == "smoke")
    banner_text = f"YOLO DETECTIONS: {fire_count} Fire | {smoke_count} Smoke"

    cv2.rectangle(img, (10, 10), (min(img.shape[1] - 10, 520), 50), (20, 20, 20), -1)
    cv2.putText(
        img,
        banner_text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255) if fire_count > 0 else (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), img)
    return out_p


def save_composite_evidence(
    thermal_img_path: Union[str, Path],
    yolo_img_path: Union[str, Path],
    alert_info: Dict[str, Any],
    output_path: Union[str, Path] = "results/composite_evidence.png",
    temp_img_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create side-by-side composite proof for dashboard and disaster response teams.
    Optionally includes temperature panel if available.
    """
    th_img = cv2.imread(str(thermal_img_path))
    yo_img = cv2.imread(str(yolo_img_path))

    if th_img is None or yo_img is None:
        return Path(thermal_img_path)

    target_h = 520
    th_w = int(th_img.shape[1] * (target_h / th_img.shape[0]))
    yo_w = int(yo_img.shape[1] * (target_h / yo_img.shape[0]))

    th_resized = cv2.resize(th_img, (th_w, target_h))
    yo_resized = cv2.resize(yo_img, (yo_w, target_h))

    panels = [th_resized, yo_resized]

    # If temperature map evidence is provided and exists, add as third panel or side panel
    if temp_img_path is not None and Path(temp_img_path).exists():
        temp_img = cv2.imread(str(temp_img_path))
        if temp_img is not None:
            temp_w = int(temp_img.shape[1] * (target_h / temp_img.shape[0]))
            temp_resized = cv2.resize(temp_img, (temp_w, target_h))
            panels = [temp_resized, th_resized, yo_resized]

    composite_body = np.hstack(panels)

    # Add top status banner
    total_w = composite_body.shape[1]
    banner_h = 75
    banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
    banner[:] = (25, 25, 25)

    status_str = alert_info.get("status", "MONITORING")
    badge_color = (0, 0, 255) if "FIRE" in status_str else (0, 180, 255)

    cv2.rectangle(banner, (15, 12), (320, 62), badge_color, -1)
    cv2.putText(banner, status_str, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    temp_info = ""
    if "temperature_max_c" in alert_info and alert_info["temperature_max_c"] is not None:
        temp_info = f" | Max Temp: {alert_info['temperature_max_c']:.1f}C"

    info_str = (
        f"Node: {alert_info.get('node_id')} | "
        f"Sensor Temp: {alert_info.get('sensor_temperature')}C / Gas: {alert_info.get('gas_level')}"
        f"{temp_info} | Status: {alert_info.get('verification_status')}"
    )
    cv2.putText(banner, info_str, (340, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    composite = np.vstack([banner, composite_body])
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), composite)
    return out_p
