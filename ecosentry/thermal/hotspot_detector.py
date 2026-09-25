"""
Thermal Hotspot Candidate Detector for ECOSENTRY.

Responsibilities:
- Estimate scene ambient background thermal intensity statistically.
- Apply adaptive statistical thresholding to isolate regions significantly brighter/hotter
  than surrounding baseline.
- Extract connected candidate components, bounding boxes, and centroids.
- Compute quantitative intensity metrics (peak intensity, mean intensity, Delta I).
- Filter out sub-resolution noise artifacts.

Important Scientific & Safety Limitations:
- The input frames are uncalibrated thermal visualizations, not raw radiometric Celsius data.
- NO pixel intensity is converted to degrees Celsius.
- NO particular color is assumed to inherently mean fire.
- NO detection is classified as a confirmed fire at this stage.
- Detections are labeled strictly as 'HOTSPOT_CANDIDATE'.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


@dataclass
class HotspotCandidate:
    """
    Structured representation of an isolated thermal hotspot candidate.
    """
    candidate_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid: Tuple[int, int]         # (cx, cy)
    area_pixels: float               # Area of contour in pixels
    max_intensity: float             # Peak pixel intensity inside candidate
    mean_intensity: float            # Average pixel intensity inside candidate
    bg_intensity: float              # Estimated ambient background intensity
    delta_intensity: float           # Relative difference: (mean_intensity - bg_intensity)
    delta_max: float                 # Peak relative difference: (max_intensity - bg_intensity)
    status: str = "HOTSPOT_CANDIDATE"

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to dictionary for serialization and logging."""
        return asdict(self)


class HotspotDetector:
    """
    Adaptive statistical detector for isolating thermal anomaly candidates.
    """

    def __init__(
        self,
        k_sigma: float = 2.5,
        min_area_pixels: int = 15,
        max_area_ratio: float = 0.25,
        use_enhanced: bool = True,
    ):
        """
        Initialize HotspotDetector.

        Args:
            k_sigma: Number of standard deviations above ambient background mean
                     required for anomaly thresholding (default: 2.5).
            min_area_pixels: Minimum contour area in pixels to eliminate noise (default: 15).
            max_area_ratio: Maximum area fraction of whole image to reject global false alarms (e.g. 0.25).
            use_enhanced: Whether to run segmentation on the CLAHE-enhanced image (default: True).
        """
        self.k_sigma = k_sigma
        self.min_area_pixels = min_area_pixels
        self.max_area_ratio = max_area_ratio
        self.use_enhanced = use_enhanced

    def estimate_background(self, gray: np.ndarray) -> Tuple[float, float]:
        """
        Estimate ambient background thermal intensity using robust statistics.

        In thermal imagery, the ambient background constitutes the majority of the scene.
        We use median and standard deviation to form an anomaly baseline.

        Args:
            gray: 2D uint8 grayscale array.

        Returns:
            Tuple[float, float]: (background_median, background_std)
        """
        bg_median = float(np.median(gray))
        bg_std = float(np.std(gray))
        return bg_median, bg_std

    def compute_adaptive_threshold(
        self,
        intensity_map: np.ndarray,
        bg_median: float,
        bg_std: float,
    ) -> Tuple[int, np.ndarray]:
        """
        Compute an adaptive statistical threshold and generate a binary anomaly mask.

        Threshold condition:
            T_stat = bg_median + k_sigma * bg_std
        Bounded to ensure it isolates genuine upper-tail outliers.

        Args:
            intensity_map: 2D uint8 intensity image (gray or enhanced).
            bg_median: Ambient background median.
            bg_std: Ambient background standard deviation.

        Returns:
            Tuple[int, np.ndarray]: (threshold_value, binary_mask)
        """
        # Calculate statistical anomaly threshold
        stat_threshold = bg_median + (self.k_sigma * bg_std)

        # Ensure threshold is meaningfully above background and within valid 8-bit bounds
        threshold_val = int(np.clip(stat_threshold, bg_median + 10.0, 254.0))

        # Generate binary mask
        _, binary_mask = cv2.threshold(
            intensity_map, threshold_val, 255, cv2.THRESH_BINARY
        )

        # Apply morphological opening to eliminate single-pixel noise spikes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        return threshold_val, clean_mask

    def detect(self, preprocessed_data: Dict[str, Any]) -> List[HotspotCandidate]:
        """
        Detect hotspot candidates from preprocessed thermal frame data.

        Args:
            preprocessed_data: Output dictionary from ThermalPreprocessor.process().

        Returns:
            List[HotspotCandidate]: Extracted candidate anomalies.
        """
        gray = preprocessed_data["gray"]
        analysis_image = (
            preprocessed_data["enhanced"] if self.use_enhanced else gray
        )
        img_h, img_w = gray.shape[:2]
        total_pixels = img_h * img_w

        # 1. Estimate ambient background
        bg_median, bg_std = self.estimate_background(gray)

        # 2. Compute adaptive threshold and binary mask
        threshold_val, binary_mask = self.compute_adaptive_threshold(
            analysis_image, bg_median, bg_std
        )

        # 3. Find connected anomaly contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: List[HotspotCandidate] = []
        candidate_idx = 1

        for cnt in contours:
            area = float(cv2.contourArea(cnt))

            # Filter out sub-resolution noise
            if area < self.min_area_pixels:
                continue

            # Filter out whole-scene saturation artifacts
            if area > (total_pixels * self.max_area_ratio):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Create a mask for this specific contour to compute accurate interior stats
            contour_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1)

            # Extract pixel intensities belonging strictly to the hotspot region
            hotspot_pixels = gray[contour_mask == 255]

            if len(hotspot_pixels) == 0:
                continue

            max_int = float(np.max(hotspot_pixels))
            mean_int = float(np.mean(hotspot_pixels))
            delta_int = float(mean_int - bg_median)
            delta_max = float(max_int - bg_median)

            # Compute centroid
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            candidate = HotspotCandidate(
                candidate_id=candidate_idx,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area_pixels=area,
                max_intensity=max_int,
                mean_intensity=mean_int,
                bg_intensity=bg_median,
                delta_intensity=delta_int,
                delta_max=delta_max,
                status="HOTSPOT_CANDIDATE",
            )
            candidates.append(candidate)
            candidate_idx += 1

        return candidates


def detect_hotspot_candidates(
    preprocessed_data: Dict[str, Any],
    k_sigma: float = 2.5,
    min_area_pixels: int = 15,
) -> List[HotspotCandidate]:
    """
    Functional convenience wrapper for hotspot candidate detection.

    Args:
        preprocessed_data: Output dictionary from ThermalPreprocessor.
        k_sigma: Statistical threshold multiplier.
        min_area_pixels: Minimum contour area threshold.

    Returns:
        List[HotspotCandidate]: List of detected hotspot candidates.
    """
    detector = HotspotDetector(k_sigma=k_sigma, min_area_pixels=min_area_pixels)
    return detector.detect(preprocessed_data)


def draw_hotspot_candidates(
    image: np.ndarray,
    candidates: List[HotspotCandidate],
    color: Tuple[int, int, int] = (0, 165, 255),  # Amber/Orange for candidate
    thickness: int = 2,
) -> np.ndarray:
    """
    Annotate an image with bounding boxes and metadata for candidate hotspots.

    Args:
        image: BGR image array.
        candidates: List of HotspotCandidate instances.
        color: Box color in BGR (default: amber/orange).
        thickness: Line thickness.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = image.copy()
    for cand in candidates:
        x, y, w, h = cand.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        # Label: Candidate ID and relative Delta I
        label = f"Cand #{cand.candidate_id} (dI: +{cand.delta_intensity:.1f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = max(y - 6, th + 4)
        cv2.rectangle(
            annotated,
            (x, label_y - th - 2),
            (x + tw + 4, label_y + baseline),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 2, label_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

        # Mark centroid
        cv2.circle(annotated, cand.centroid, 3, (0, 255, 255), -1)

    return annotated


def detect_hotspots(
    image: np.ndarray,
    k: float = 2.5,
    minimum_area: int = 200,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Beginner-friendly direct hotspot detection function.

    Takes a thermal frame (BGR or Grayscale), performs background estimation,
    adaptive statistical thresholding, contour extraction, and metric calculation.

    Args:
        image: 2D or 3D numpy array.
        k: Multiplier for standard deviation above background (default: 2.5).
        minimum_area: Minimum area in pixels to eliminate noise (default: 200).

    Returns:
        Tuple[List[Dict], np.ndarray]: (detections, binary_mask)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    background = float(np.median(blurred))
    sigma = float(np.std(blurred))
    threshold_val = min(255.0, background + k * sigma)

    mask = (blurred >= threshold_val).astype(np.uint8) * 255

    close_kernel = np.ones((21, 21), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        region = gray[y : y + h, x : x + w]
        if region.size == 0:
            continue

        peak = float(np.max(region))
        mean = float(np.mean(region))
        delta = float(peak - background)

        detections.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(area),
            "peak_intensity": peak,
            "mean_intensity": mean,
            "background_intensity": background,
            "delta_intensity": delta,
            "label": "HOTSPOT_CANDIDATE",
        })

    # Sort hotspots by delta intensity (strongest first)
    detections.sort(key=lambda d: d["delta_intensity"], reverse=True)

    return detections, mask

