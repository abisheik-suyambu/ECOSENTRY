"""
ECOSENTRY Evidence Fusion Package
Combines multi-modal sensor, thermal, and computer vision evidence.
"""

from .evidence_fusion import EvidenceFusion, fuse_evidence

__all__ = ["EvidenceFusion", "fuse_evidence"]
