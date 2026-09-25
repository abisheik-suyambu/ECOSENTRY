"""
Test script for ECOSENTRY Thermal Preprocessing and Hotspot Candidate Extractor.

This script:
1. Loads sample thermal frames from ecosentry/datasets/sample_thermal/
2. Runs the ThermalPreprocessor to normalize and enhance local contrast.
3. Runs the HotspotDetector to statistically locate thermal anomaly candidates.
4. Annotates frames with candidate bounding boxes and relative intensity deltas (Delta I).
5. Saves visual outputs to ecosentry/results/hotspot_test/
6. Prints detailed candidate statistics for each processed frame.

Note:
All detected regions are candidate anomalies only. No fire claims or Celsius conversions
are made.
"""

import os
import sys
import glob
import cv2

# Ensure workspace root is in sys.path when running script directly
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecosentry.thermal.preprocessor import ThermalPreprocessor
from ecosentry.thermal.hotspot_detector import HotspotDetector, draw_hotspot_candidates


def run_thermal_hotspot_test(num_frames: int = 5):
    """
    Execute the test pipeline across sample thermal frames.

    Args:
        num_frames: Number of sample frames to process (default: 5).
    """
    # 1. Resolve paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sample_dir = os.path.join(base_dir, "ecosentry", "datasets", "sample_thermal")
    output_dir = os.path.join(base_dir, "ecosentry", "results", "hotspot_test")

    os.makedirs(output_dir, exist_ok=True)

    # 2. Gather image paths
    image_pattern = os.path.join(sample_dir, "*.png")
    all_images = sorted(glob.glob(image_pattern))

    if not all_images:
        print(f"[ERROR] No sample images found in: {sample_dir}")
        return False

    selected_images = all_images[:num_frames]
    print("=" * 70)
    print("ECOSENTRY THERMAL HOTSPOT CANDIDATE DETECTION TEST")
    print("=" * 70)
    print(f"Sample source directory : {sample_dir}")
    print(f"Output directory        : {output_dir}")
    print(f"Found total frames      : {len(all_images)}")
    print(f"Processing frames       : {len(selected_images)}")
    print("=" * 70)

    # 3. Initialize components
    preprocessor = ThermalPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
    detector = HotspotDetector(k_sigma=2.5, min_area_pixels=15)

    total_candidates_all_images = 0

    for idx, img_path in enumerate(selected_images, start=1):
        filename = os.path.basename(img_path)
        print(f"\n--- [Frame {idx}/{len(selected_images)}: {filename}] ---")

        # Step A: Preprocess
        prep_data = preprocessor.process(img_path)
        print(f"  Image shape        : {prep_data['raw_bgr'].shape}")
        print(f"  Intensity range    : min={prep_data['min_intensity']:.1f}, max={prep_data['max_intensity']:.1f}")
        print(f"  Ambient background : mean={prep_data['mean_intensity']:.1f}, std={prep_data['std_intensity']:.1f}")

        # Step B: Detect hotspot candidates
        candidates = detector.detect(prep_data)
        num_candidates = len(candidates)
        total_candidates_all_images += num_candidates
        print(f"  Hotspot candidates detected : {num_candidates}")

        if candidates:
            for cand in candidates:
                print(
                    f"    * Cand #{cand.candidate_id}: "
                    f"bbox={cand.bbox} | "
                    f"area={cand.area_pixels:.1f} px | "
                    f"max_I={cand.max_intensity:.1f} | "
                    f"mean_I={cand.mean_intensity:.1f} | "
                    f"Delta_I=+{cand.delta_intensity:.1f} | "
                    f"status={cand.status}"
                )
        else:
            print("    * No significant thermal anomaly candidates above statistical threshold.")

        # Step C: Draw annotations
        annotated_image = draw_hotspot_candidates(prep_data["raw_bgr"], candidates)

        # Overlay global frame summary
        summary_text = f"Frame: {filename} | Candidates: {num_candidates}"
        cv2.putText(
            annotated_image,
            summary_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Step D: Save output
        out_filename = f"detected_{filename}"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, annotated_image)
        print(f"  Saved visualization -> {out_path}")

    print("\n" + "=" * 70)
    print("TEST EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total frames processed         : {len(selected_images)}")
    print(f"Total hotspot candidates found : {total_candidates_all_images}")
    print(f"Visual results directory       : {output_dir}")
    print("Note: Candidates are anomaly regions only (not verified fires).")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = run_thermal_hotspot_test(num_frames=5)
    if success:
        print("\n[SUCCESS] Hotspot candidate test completed successfully.")
    else:
        print("\n[FAILED] Test execution encountered errors.")
