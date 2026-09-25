"""
ECOSENTRY Unified Detection and Fusion Pipeline.

Architecture:
Thermal Image
  → Thermal Preprocessing (standardized intensity, safe min-max, CLAHE)
  → Hotspot Candidate Detection (adaptive statistical thresholding, Delta I, area)
  → Lightweight YOLO Inference (object detection, COCO or custom weights)
  → Fusion & Decision Logic (dual-condition corroboration)
  → Evidence Image Generation (composite visual archive with telemetry)
  → Structured Alert Emission (JSON format)

States:
- NO_ANOMALY: No thermal hotspot candidate detected.
- UNVERIFIED_THERMAL_HOTSPOT: Statistical thermal anomaly isolated; awaiting custom YOLO validation.
- YOLO_CONFIRMED_HOTSPOT: Corroborated by both thermal delta and target YOLO hotspot class.

Important Scientific Safeguards:
- No Celsius conversions are performed.
- Standard COCO classes (e.g., person, car) are NEVER equated to thermal hotspots.
- Zero detections from generic models are recorded transparently.
- 'CONFIRMED_FIRE' is deliberately omitted until verified by multi-modal ground truth.
"""

import os
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2

from ecosentry.thermal.preprocessor import ThermalPreprocessor
from ecosentry.thermal.hotspot_detector import HotspotDetector, HotspotCandidate
from ecosentry.models.detector import YOLODetector, YOLODetection
from ecosentry.utils.alert import AlertManager, PipelineAlert
from ecosentry.utils.evidence_saver import EvidenceSaver


def compute_bbox_overlap(
    thermal_xywh: Tuple[int, int, int, int],
    yolo_xyxy: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    """
    Compute Intersection-over-Union (IoU) and Intersection-over-Candidate (IoC)
    between a thermal hotspot box and a YOLO detection box.

    Args:
        thermal_xywh: (x, y, w, h) of thermal candidate.
        yolo_xyxy: (x1, y1, x2, y2) of YOLO detection box.

    Returns:
        Tuple[float, float]: (IoU, IoC)
    """
    tx, ty, tw, th = thermal_xywh
    tx1, ty1, tx2, ty2 = tx, ty, tx + tw, ty + th
    yx1, yy1, yx2, yy2 = yolo_xyxy

    ix1 = max(tx1, yx1)
    iy1 = max(ty1, yy1)
    ix2 = min(tx2, yx2)
    iy2 = min(ty2, yy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = float(inter_w * inter_h)

    thermal_area = float(tw * th)
    yolo_area = float(max(0, yx2 - yx1) * max(0, yy2 - yy1))

    if inter_area == 0.0 or thermal_area == 0.0:
        return 0.0, 0.0

    union_area = thermal_area + yolo_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    ioc = inter_area / thermal_area  # Fraction of hotspot enclosed in YOLO box

    return iou, ioc


class EcoSentryPipeline:
    """
    Unified end-to-end detection and evidence pipeline for ECOSENTRY.
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        thermal_delta_threshold: float = 30.0,
        minimum_hotspot_area: float = 15.0,
        yolo_confidence_threshold: float = 0.25,
        iou_threshold: float = 0.1,
        target_yolo_class: str = "thermal_hotspot",
        evidence_dir: str = "ecosentry/results/evidence",
    ):
        """
        Initialize the ECOSENTRY pipeline with configurable parameters.

        Args:
            yolo_model_name: YOLO weights identifier or local path (default: 'yolov8n.pt').
            thermal_delta_threshold: Minimum relative intensity delta (Delta I) to qualify (default: 30.0).
            minimum_hotspot_area: Minimum candidate area in pixels (default: 15.0).
            yolo_confidence_threshold: Minimum confidence required from target YOLO model (default: 0.25).
            iou_threshold: Minimum bounding box spatial overlap (default: 0.1).
            target_yolo_class: Label for future thermal hotspot model (default: 'thermal_hotspot').
            evidence_dir: Destination path for saved visual evidence.
        """
        self.thermal_delta_threshold = thermal_delta_threshold
        self.minimum_hotspot_area = minimum_hotspot_area
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_yolo_class = target_yolo_class

        # Initialize submodules
        self.preprocessor = ThermalPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
        self.hotspot_detector = HotspotDetector(
            k_sigma=2.5,
            min_area_pixels=int(minimum_hotspot_area),
        )
        self.yolo_detector = YOLODetector(
            model_name=yolo_model_name,
            conf_threshold=yolo_confidence_threshold,
        )
        self.evidence_saver = EvidenceSaver(output_dir=evidence_dir)

    def process_frame(
        self,
        image_source: Union[str, np.ndarray],
        source_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full pipeline on a single thermal image frame.

        Args:
            image_source: Filepath or numpy BGR image array.
            source_name: Optional identifier for filename generation.

        Returns:
            dict containing:
                - 'status': One of NO_ANOMALY, UNVERIFIED_THERMAL_HOTSPOT, YOLO_CONFIRMED_HOTSPOT
                - 'thermal_candidates': List of qualifying HotspotCandidate instances
                - 'yolo_detections': List of all YOLODetection instances
                - 'confirmed_hotspot': Best corroborated candidate (or None)
                - 'alert': PipelineAlert instance
                - 'evidence_path': Filepath of saved evidence image
        """
        # Resolve source name
        if isinstance(image_source, str):
            filename = os.path.basename(image_source)
        else:
            filename = source_name if source_name else "frame_input.png"

        # 1. Thermal Preprocessing
        prep_data = self.preprocessor.process(image_source)
        raw_bgr = prep_data["raw_bgr"]

        # 2. Thermal Hotspot Candidate Detection
        raw_candidates = self.hotspot_detector.detect(prep_data)

        # Filter candidates by configurable thresholds
        qualifying_candidates = [
            c for c in raw_candidates
            if c.delta_intensity >= self.thermal_delta_threshold
            and c.area_pixels >= self.minimum_hotspot_area
        ]

        # 3. YOLO Inference (preserve all genuine detections)
        yolo_detections = self.yolo_detector.predict(raw_bgr)

        # 4. Fusion & Decision Logic
        status = "NO_ANOMALY"
        primary_thermal_delta: Optional[float] = None
        primary_hotspot_area: Optional[float] = None
        primary_bbox: Optional[Tuple[int, int, int, int]] = None
        confirmed_yolo_class: Optional[str] = None
        confirmed_yolo_conf: Optional[float] = None
        best_candidate: Optional[HotspotCandidate] = None

        if qualifying_candidates:
            # Sort candidates by intensity delta (strongest anomaly first)
            qualifying_candidates.sort(key=lambda c: c.delta_intensity, reverse=True)
            best_candidate = qualifying_candidates[0]

            primary_thermal_delta = best_candidate.delta_intensity
            primary_hotspot_area = best_candidate.area_pixels
            primary_bbox = best_candidate.bbox

            # Default to UNVERIFIED unless corroborated by custom YOLO
            status = "UNVERIFIED_THERMAL_HOTSPOT"

            # Check if any YOLO detection confirms the hotspot
            for det in yolo_detections:
                # Must match target thermal hotspot class
                if det.class_name.lower() == self.target_yolo_class.lower():
                    if det.confidence >= self.yolo_confidence_threshold:
                        iou, ioc = compute_bbox_overlap(best_candidate.bbox, det.box_xyxy)
                        if iou >= self.iou_threshold or ioc >= self.iou_threshold:
                            status = "YOLO_CONFIRMED_HOTSPOT"
                            confirmed_yolo_class = det.class_name
                            confirmed_yolo_conf = det.confidence
                            break

        # 5. Create Structured Alert
        alert = AlertManager.create_alert(
            status=status,
            image_path=filename if isinstance(image_source, str) else "in_memory",
            thermal_delta=primary_thermal_delta,
            hotspot_area=primary_hotspot_area,
            hotspot_bbox=primary_bbox,
            yolo_class=confirmed_yolo_class,
            yolo_confidence=confirmed_yolo_conf,
            details={
                "total_thermal_candidates": len(qualifying_candidates),
                "total_yolo_detections": len(yolo_detections),
                "yolo_classes_seen": [d.class_name for d in yolo_detections],
            },
        )

        # 6. Save Evidence Image
        evidence_path = self.evidence_saver.save_evidence(
            base_image=raw_bgr,
            source_filename=filename,
            status=status,
            thermal_candidates=qualifying_candidates,
            yolo_detections=yolo_detections,
            thermal_delta=primary_thermal_delta,
            hotspot_area=primary_hotspot_area,
            yolo_confidence=confirmed_yolo_conf,
            timestamp_str=alert.timestamp,
        )

        return {
            "status": status,
            "thermal_candidates": qualifying_candidates,
            "yolo_detections": yolo_detections,
            "confirmed_candidate": best_candidate,
            "alert": alert,
            "evidence_path": evidence_path,
        }
