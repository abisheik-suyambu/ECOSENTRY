"""
Drone Camera Module

Simulates onboard dual-payload camera (Thermal Infrared + Visual RGB).
In future deployment on Raspberry Pi, this module can interface with:
- FLIR Lepton / Seek Thermal / MLX90640 (I2C/SPI/USB)
- Raspberry Pi Camera Module V2/V3 (CSI/libcamera)
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import cv2
import numpy as np


class DroneCamera:
    """
    Simulated dual-sensor camera interface.
    Extracts thermal and visual frames from the demonstration composite input.
    """

    def __init__(self, source_path: Optional[Union[str, Path]] = None):
        if source_path is None:
            # Check standard relative paths
            candidates = [
                Path("examples/hotspot-detect.gif"),
                Path("../examples/hotspot-detect.gif"),
                Path("RTPV-YOLO/examples/hotspot-detect.gif"),
                Path(__file__).resolve().parents[2] / "examples" / "hotspot-detect.gif",
                Path(__file__).resolve().parents[3] / "examples" / "hotspot-detect.gif",
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.source_path = candidate.resolve()
                    break
            else:
                self.source_path = Path("examples/hotspot-detect.gif")
        else:
            self.source_path = Path(source_path).resolve()

    def capture_frame(self, frame_number: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Capture/load a frame and split the composite into thermal and visual portions.

        Args:
            frame_number: Frame index for animated GIF or video (default: 20).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (thermal_bgr, visual_bgr)
        """
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Camera source file not found at: {self.source_path}\n"
                f"Please ensure examples/hotspot-detect.gif exists."
            )

        capture = cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source_path}")

        # Seek to frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        capture.release()

        if not success or frame is None:
            # Fallback to reading first frame
            capture = cv2.VideoCapture(str(self.source_path))
            success, frame = capture.read()
            capture.release()
            if not success or frame is None:
                raise RuntimeError(f"Could not capture frame from {self.source_path}")

        height, width = frame.shape[:2]

        # The demo composite is side-by-side: left half is thermal, right half is visual
        if width > height and width >= 640:
            half_width = width // 2
            thermal_frame = frame[:, :half_width].copy()
            visual_frame = frame[:, half_width:].copy()
        else:
            # Fallback if image is square or standalone
            thermal_frame = frame.copy()
            visual_frame = frame.copy()

        return thermal_frame, visual_frame
