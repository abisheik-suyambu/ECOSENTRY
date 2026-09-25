"""
ECOSENTRY Models Module.

Provides wrappers for lightweight object detection architectures (e.g., Ultralytics YOLO).
"""

from .detector import YOLODetector, YOLODetection

__all__ = ["YOLODetector", "YOLODetection"]
