"""
Lightweight YOLO Detector Wrapper for ECOSENTRY.

Responsibilities:
- Initialize and load lightweight YOLO architectures (e.g. YOLOv8n) via Ultralytics.
- Automatically fetch official weights on first run if not present locally.
- Execute inference on individual frames with configurable confidence and image dimensions.
- Return structured detection objects containing bounding boxes, class IDs, class names,
  and confidence scores.
- Provide visualization annotation utilities.

Important Scientific Limitations:
- The base pretrained model is trained on standard visible-spectrum benchmark data (e.g. COCO 80 classes).
- The standard model does NOT natively identify thermal forest fires or thermal anomalies.
- Detections produced by this general model should NOT be reported as confirmed fires or hotspots
  unless the model has been explicitly fine-tuned and validated on thermal hotspot labels.
"""

import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2


@dataclass
class YOLODetection:
    """
    Structured container for a single YOLO object detection.
    """
    box_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    box_xywh: Tuple[int, int, int, int]  # (x, y, w, h)
    class_id: int                        # Numeric class ID
    class_name: str                      # Human-readable class label
    confidence: float                    # Model confidence score [0.0 - 1.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection instance to dictionary for logging and serialization."""
        return asdict(self)


class YOLODetector:
    """
    Inference wrapper for lightweight YOLO models using the Ultralytics engine.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        img_size: int = 640,
        device: Optional[str] = None,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_name: Pretrained weights file or model identifier (default: 'yolov8n.pt').
            conf_threshold: Minimum confidence score to accept detection (default: 0.25).
            img_size: Image resolution passed to network inference (default: 640).
            device: Computing device ('cpu', 'cuda', or None for auto-detect).
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.device = device
        self._load_model()

    def _load_model(self):
        """
        Load the model via Ultralytics. Official weights will be downloaded
        automatically by the framework if not locally present.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "The 'ultralytics' package is required. "
                "Install it via: pip install ultralytics"
            ) from e

        # Check if weights exist in the same directory as detector.py
        if not os.path.isabs(self.model_name) and not os.path.exists(self.model_name):
            local_weights = os.path.join(os.path.dirname(__file__), self.model_name)
            if os.path.exists(local_weights):
                self.model_name = local_weights

        print(f"[YOLODetector] Loading model weights: '{self.model_name}'...")
        self.model = YOLO(self.model_name)
        self.class_names = getattr(self.model, "names", {})
        print(f"[YOLODetector] Model loaded successfully. Classes available: {len(self.class_names)}")

    def predict(
        self,
        image_source: Union[str, np.ndarray],
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[YOLODetection]:
        """
        Execute object detection on a single image.

        Args:
            image_source: File path to image or in-memory BGR numpy array.
            conf: Optional override for confidence threshold.
            imgsz: Optional override for inference image size.

        Returns:
            List[YOLODetection]: Extracted detections meeting the confidence threshold.
        """
        effective_conf = conf if conf is not None else self.conf_threshold
        effective_imgsz = imgsz if imgsz is not None else self.img_size

        # Run inference
        results = self.model.predict(
            source=image_source,
            conf=effective_conf,
            imgsz=effective_imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[YOLODetection] = []

        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract boxes, confidences, and class IDs
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()

        for i in range(len(xyxy_arr)):
            x1, y1, x2, y2 = [int(v) for v in xyxy_arr[i]]
            score = float(conf_arr[i])
            cid = int(cls_arr[i])
            cname = self.class_names.get(cid, str(cid))

            w = x2 - x1
            h = y2 - y1

            det = YOLODetection(
                box_xyxy=(x1, y1, x2, y2),
                box_xywh=(x1, y1, w, h),
                class_id=cid,
                class_name=cname,
                confidence=score,
            )
            detections.append(det)

        return detections

    def annotate(
        self,
        image: np.ndarray,
        detections: List[YOLODetection],
        box_color: Tuple[int, int, int] = (255, 100, 0),  # Blue/Cyan
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw YOLO bounding boxes and detection labels on an image.

        Args:
            image: BGR image array.
            detections: List of YOLODetection instances.
            box_color: Bounding box line color in BGR.
            thickness: Line thickness.

        Returns:
            np.ndarray: Annotated BGR image.
        """
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            label = f"{det.class_name} {det.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 2),
                (x1 + tw + 4, label_y + baseline),
                (30, 30, 30),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )
        return annotated
