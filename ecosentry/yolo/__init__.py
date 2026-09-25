"""
ECOSENTRY YOLO Module
Loads trained YOLOv8 fire and smoke detector weights.
"""

from .fire_detector import FireDetector

__all__ = ["FireDetector"]
