"""
Verification Module

Cross-verifies thermal anomalies against YOLO AI detections.
Eliminates single-modality false alarms (e.g. heated roofs, car engines, sunlight glint).
"""

from typing import List, Dict, Any, Tuple, Optional


def compute_overlap(
    thermal_bbox: List[int],
    yolo_bbox: List[float],
) -> Tuple[float, float]:
    """
    Compute Intersection-over-Union (IoU) and Intersection-over-Hotspot (IoH).

    Args:
        thermal_bbox: [x, y, w, h] of thermal hotspot
        yolo_bbox: [x1, y1, x2, y2] of YOLO detection

    Returns:
        Tuple[float, float]: (IoU, IoH)
    """
    tx, ty, tw, th = thermal_bbox
    tx1, ty1, tx2, ty2 = tx, ty, tx + tw, ty + th
    yx1, yy1, yx2, yy2 = yolo_bbox

    ix1 = max(tx1, yx1)
    iy1 = max(ty1, yy1)
    ix2 = min(tx2, yx2)
    iy2 = min(ty2, yy2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    thermal_area = max(1.0, float(tw * th))
    yolo_area = max(1.0, float(max(0.0, yx2 - yx1) * max(0.0, yy2 - yy1)))

    if inter_area == 0.0:
        return 0.0, 0.0

    union_area = thermal_area + yolo_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    ioh = inter_area / thermal_area

    return round(iou, 4), round(ioh, 4)


class FireVerifier:
    """
    Correlates thermal anomaly candidates with YOLO fire and smoke bounding boxes.
    """

    def __init__(self, overlap_threshold: float = 0.05):
        self.overlap_threshold = overlap_threshold

    def verify(
        self,
        thermal_hotspots: List[Dict[str, Any]],
        yolo_detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify thermal and YOLO detections against each other.

        Args:
            thermal_hotspots: List of hotspot dicts from thermal detector.
            yolo_detections: List of detection dicts from YOLO model.

        Returns:
            Dict containing verification status and matched corroborations.
        """
        fire_dets = [d for d in yolo_detections if d.get("class_name") == "fire"]
        smoke_dets = [d for d in yolo_detections if d.get("class_name") == "smoke"]

        has_thermal = len(thermal_hotspots) > 0
        has_fire = len(fire_dets) > 0
        has_smoke = len(smoke_dets) > 0

        corroborated_pairs = []

        # Find spatial overlap between hotspots and YOLO detections
        for h_idx, hotspot in enumerate(thermal_hotspots):
            for y_idx, ydet in enumerate(yolo_detections):
                iou, ioh = compute_overlap(hotspot["bbox"], ydet["bbox"])
                if iou >= self.overlap_threshold or ioh >= self.overlap_threshold:
                    corroborated_pairs.append({
                        "hotspot_index": h_idx,
                        "hotspot_bbox": hotspot["bbox"],
                        "hotspot_delta": hotspot["delta_intensity"],
                        "yolo_class": ydet["class_name"],
                        "yolo_confidence": ydet["confidence"],
                        "yolo_bbox": ydet["bbox"],
                        "iou": iou,
                        "ioh": ioh,
                    })

        # Decision states
        if has_thermal and (has_fire or has_smoke):
            if len(corroborated_pairs) > 0:
                status = "CORROBORATED_DISASTER_EVIDENCE"
                reason = "Thermal hotspot spatially matches YOLO fire/smoke detection."
            else:
                status = "MULTIPLE_ANOMALIES_UNALIGNED"
                reason = "Thermal hotspot and YOLO detection present in scene but at different spatial coordinates."
        elif has_thermal and not (has_fire or has_smoke):
            status = "THERMAL_ANOMALY_ONLY"
            reason = "Abnormal thermal region detected; no visual fire or smoke detected by YOLO (possible heated surface, roof, or vehicle)."
        elif (has_fire or has_smoke) and not has_thermal:
            status = "VISUAL_DETECTION_WITHOUT_THERMAL"
            reason = "YOLO flagged visual fire/smoke, but no thermal hotspot exceeded anomaly threshold."
        else:
            status = "NO_DISASTER_EVIDENCE"
            reason = "No thermal hotspot or visual fire/smoke detected."

        strongest_hotspot = thermal_hotspots[0] if thermal_hotspots else None

        return {
            "verification_status": status,
            "has_thermal_anomaly": has_thermal,
            "has_yolo_fire": has_fire,
            "has_yolo_smoke": has_smoke,
            "hotspot_count": len(thermal_hotspots),
            "yolo_fire_count": len(fire_dets),
            "yolo_smoke_count": len(smoke_dets),
            "strongest_hotspot": strongest_hotspot,
            "corroborated_count": len(corroborated_pairs),
            "corroborations": corroborated_pairs,
            "reason": reason,
        }


def verify_detections(
    thermal_hotspots: List[Dict[str, Any]],
    yolo_detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convenience wrapper for verification."""
    verifier = FireVerifier()
    return verifier.verify(thermal_hotspots, yolo_detections)
