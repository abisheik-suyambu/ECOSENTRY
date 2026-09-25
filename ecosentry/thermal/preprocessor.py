"""
Thermal Preprocessing Module for ECOSENTRY.

Responsibilities:
- Load thermal image frames.
- Standardize into single-channel intensity representation.
- Perform safe min-max normalization without division-by-zero risk.
- Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement.
- Expose reusable functions and classes for pipeline integration.

Note:
The processed values represent normalized visual thermal intensity, not absolute
radiometric temperature in degrees Celsius.
"""

import os
from typing import Dict, Any, Union
import numpy as np
import cv2


class ThermalPreprocessor:
    """
    Thermal image preprocessor providing standardized normalization and contrast enhancement.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """
        Initialize the preprocessor with CLAHE parameters.

        Args:
            clip_limit: Threshold for contrast limiting in CLAHE (default: 2.0).
            tile_grid_size: Grid size for histogram equalization (default: (8, 8)).
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def load_image(self, image_source: Union[str, np.ndarray]) -> np.ndarray:
        """
        Load an image from a filepath or validate an existing numpy array.

        Args:
            image_source: Path to image file or an in-memory BGR numpy array.

        Returns:
            np.ndarray: BGR image array.
        """
        if isinstance(image_source, str):
            if not os.path.isfile(image_source):
                raise FileNotFoundError(f"Thermal image file not found: {image_source}")
            img = cv2.imread(image_source, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from path: {image_source}")
            return img
        elif isinstance(image_source, np.ndarray):
            return image_source
        else:
            raise TypeError(f"Unsupported image source type: {type(image_source)}")

    def to_standard_intensity(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a multi-channel image into a single-channel grayscale intensity map.

        Args:
            image_bgr: BGR or Grayscale input image.

        Returns:
            np.ndarray: 2D uint8 grayscale intensity array.
        """
        if len(image_bgr.shape) == 2:
            return image_bgr.copy()
        elif len(image_bgr.shape) == 3:
            if image_bgr.shape[2] == 3:
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            elif image_bgr.shape[2] == 4:
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Unsupported image dimensions: {image_bgr.shape}")

    def safe_min_max_normalize(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply safe min-max scaling to stretch dynamic range to [0.0, 1.0].
        Guards against zero-division for uniform / flat images.

        Args:
            gray: 2D uint8 or float grayscale intensity array.

        Returns:
            np.ndarray: 2D float32 array normalized to [0.0, 1.0].
        """
        gray_f = gray.astype(np.float32)
        min_val = float(np.min(gray_f))
        max_val = float(np.max(gray_f))

        diff = max_val - min_val
        if diff <= 1e-6:
            # Flat image: avoid division by zero, return mid-scale
            return np.zeros_like(gray_f)
        return (gray_f - min_val) / diff

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to amplify subtle local thermal contrast against background.

        Args:
            gray: 2D uint8 grayscale intensity array.

        Returns:
            np.ndarray: 2D uint8 contrast-enhanced image.
        """
        if gray.dtype != np.uint8:
            gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        else:
            gray_uint8 = gray
        return self.clahe.apply(gray_uint8)

    def process(self, image_source: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run the complete thermal preprocessing pipeline on an image.

        Args:
            image_source: Image file path or numpy array.

        Returns:
            dict containing:
                - 'raw_bgr': original color frame (np.ndarray)
                - 'gray': standardized single-channel grayscale (np.ndarray, uint8)
                - 'normalized_float': float array scaled to [0.0, 1.0] (np.ndarray, float32)
                - 'normalized_uint8': stretched 8-bit image [0, 255] (np.ndarray, uint8)
                - 'enhanced': CLAHE contrast-enhanced 8-bit image (np.ndarray, uint8)
                - 'min_intensity': minimum pixel intensity in raw gray
                - 'max_intensity': maximum pixel intensity in raw gray
                - 'mean_intensity': mean pixel intensity in raw gray
                - 'std_intensity': standard deviation of raw gray
        """
        raw_bgr = self.load_image(image_source)
        gray = self.to_standard_intensity(raw_bgr)

        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        norm_float = self.safe_min_max_normalize(gray)
        norm_uint8 = np.clip(norm_float * 255.0, 0, 255).astype(np.uint8)
        enhanced = self.apply_clahe(norm_uint8)

        return {
            "raw_bgr": raw_bgr,
            "gray": gray,
            "normalized_float": norm_float,
            "normalized_uint8": norm_uint8,
            "enhanced": enhanced,
            "min_intensity": min_val,
            "max_intensity": max_val,
            "mean_intensity": mean_val,
            "std_intensity": std_val,
        }


def preprocess_thermal_frame(
    image_source: Union[str, np.ndarray],
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> Dict[str, Any]:
    """
    Functional convenience wrapper for single frame preprocessing.

    Args:
        image_source: Path to image file or numpy array.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE grid size.

    Returns:
        dict: Preprocessing results and intermediate representations.
    """
    preprocessor = ThermalPreprocessor(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    return preprocessor.process(image_source)
