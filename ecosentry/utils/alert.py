"""
ECOSENTRY Alert System.

Responsibilities:
- Generates structured, timestamped alerts containing thermal and detection metrics.
- Serializes telemetry data to JSON format.
- Strictly adheres to honest status classification:
  * NO_ANOMALY: No thermal anomaly detected.
  * UNVERIFIED_THERMAL_HOTSPOT: Statistical thermal anomaly isolated; awaiting custom YOLO validation.
  * YOLO_CONFIRMED_HOTSPOT: Corroborated by both thermal delta and target YOLO hotspot class.
- Strictly excludes speculative 'fire confirmed' declarations.
"""

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any


VALID_STATUSES = {
    "NO_ANOMALY",
    "UNVERIFIED_THERMAL_HOTSPOT",
    "YOLO_CONFIRMED_HOTSPOT",
}


@dataclass
class PipelineAlert:
    """
    Structured alert container representing the final decision of the ECOSENTRY pipeline.
    """
    event_id: str
    timestamp: str
    status: str
    thermal_delta: Optional[float]
    hotspot_area: Optional[float]
    hotspot_bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    yolo_class: Optional[str]
    yolo_confidence: Optional[float]
    image_path: str
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{self.status}'. Must be one of: {sorted(VALID_STATUSES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Convert alert to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class AlertManager:
    """
    Manager to construct and record structured pipeline alerts.
    """

    @staticmethod
    def create_alert(
        status: str,
        image_path: str,
        thermal_delta: Optional[float] = None,
        hotspot_area: Optional[float] = None,
        hotspot_bbox: Optional[Tuple[int, int, int, int]] = None,
        yolo_class: Optional[str] = None,
        yolo_confidence: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> PipelineAlert:
        """
        Create a new timestamped PipelineAlert.
        """
        event_id = f"ECO-{uuid.uuid4().hex[:10].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        return PipelineAlert(
            event_id=event_id,
            timestamp=timestamp,
            status=status,
            thermal_delta=thermal_delta,
            hotspot_area=hotspot_area,
            hotspot_bbox=hotspot_bbox,
            yolo_class=yolo_class,
            yolo_confidence=yolo_confidence,
            image_path=image_path,
            details=details,
        )
