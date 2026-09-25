"""
ECOSENTRY Verification Package
Cross-correlates thermal anomalies with YOLO visual detections.
"""

from .fire_verifier import FireVerifier, verify_detections

__all__ = ["FireVerifier", "verify_detections"]
