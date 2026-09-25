"""
Modular YOLOv8 Fire & Smoke Detector for ECOSENTRY

Loads pre-trained edge weights (best.pt) trained for:
  Class 0: fire
  Class 1: smoke

Runs genuine inference via Ultralytics and returns structured detection objects.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
from ultralytics import YOLO


class FireDetector:
    """
    Lightweight YOLOv8 object detector for Fire and Smoke signatures.
    Designed for Raspberry Pi and edge workstation execution.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the YOLO model using the pre-existing best.pt weights.
        """
        if model_path is None:
            candidates = [
                Path("ecosentry/models/best.pt"),
                Path(__file__).resolve().parents[1] / "models" / "best.pt",
                Path("RTPV-YOLO/ecosentry/models/best.pt"),
                Path(__file__).resolve().parents[2] / "ecosentry" / "models" / "best.pt",
                Path(__file__).resolve().parents[2] / "RTPV-YOLO" / "ecosentry" / "models" / "best.pt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.model_path = candidate.resolve()
                    break
            else:
                self.model_path = Path("ecosentry/models/best.pt")
        else:
            self.model_path = Path(model_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Fire detection model not found at: {self.model_path}\n"
                f"Expected weights at ecosentry/models/best.pt"
            )

        # Load weights into Ultralytics YOLO engine
        self.model = YOLO(str(self.model_path))
        self.names = self.model.names

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray],
        confidence: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on image path or numpy frame array.

        Args:
            image_input: Path to image file or numpy array (BGR).
            confidence: Minimum confidence threshold (default: 0.25).

        Returns:
            List[Dict[str, Any]]: Structured list of genuine model predictions:
                [
                    {
                        "class_id": 0,
                        "class_name": "fire",
                        "confidence": 0.59,
                        "bbox": [x1, y1, x2, y2]
                    }
                ]
        """
        if isinstance(image_input, (str, Path)):
            src = str(image_input)
        else:
            src = image_input

        results = self.model(src, conf=confidence, verbose=False)
        detections: List[Dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence_score = float(box.conf[0].item())
                coordinates = [float(c) for c in box.xyxy[0].tolist()]
                class_name = self.names.get(class_id, f"class_{class_id}")

                detections.append({
                    "class_id": class_id,
                    "class_name": str(class_name),
                    "confidence": round(confidence_score, 4),
                    "bbox": [round(c, 2) for c in coordinates],
                })

        return detections
