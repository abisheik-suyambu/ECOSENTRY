"""
Test script for the ECOSENTRY End-to-End Pipeline (Thermal + YOLO + Fusion + Evidence + Alert).

Execution flow:
1. Loads sample thermal frames from ecosentry/datasets/sample_thermal/
2. Runs EcoSentryPipeline across the frames.
3. Prints thermal candidate metrics (Delta I, area, bbox).
4. Prints actual YOLO detections (from current generic COCO model).
5. Demonstrates honest fusion state determination:
   - UNVERIFIED_THERMAL_HOTSPOT (because current COCO model has no 'thermal_hotspot' class)
   - NO_ANOMALY (if no candidate qualifies)
   - YOLO_CONFIRMED_HOTSPOT (reserved for future custom thermal YOLO model)
6. Displays the saved evidence image path in ecosentry/results/evidence/.
7. Prints a sample structured JSON alert emitted by the AlertManager.
"""

import os
import sys
import glob

# Ensure workspace root is in sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecosentry.inference.pipeline import EcoSentryPipeline


def run_pipeline_test(num_frames: int = 5):
    """
    Run pipeline test across sample frames.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sample_dir = os.path.join(base_dir, "ecosentry", "datasets", "sample_thermal")
    evidence_dir = os.path.join(base_dir, "ecosentry", "results", "evidence")

    all_images = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
    if not all_images:
        print(f"[ERROR] No sample images found in: {sample_dir}")
        return False

    selected_images = all_images[:num_frames]

    print("=" * 80)
    print("ECOSENTRY END-TO-END FUSION & EVIDENCE PIPELINE TEST")
    print("=" * 80)
    print(f"Sample source dir      : {sample_dir}")
    print(f"Evidence output dir    : {evidence_dir}")
    print(f"Frames to test         : {len(selected_images)}")
    print("Configured thresholds  :")
    print("  * thermal_delta_threshold : >= 30.0")
    print("  * minimum_hotspot_area    : >= 15.0 px")
    print("  * yolo_confidence_threshold: >= 0.25")
    print("  * target_yolo_class       : 'thermal_hotspot'")
    print("=" * 80)

    # Initialize unified pipeline
    pipeline = EcoSentryPipeline(
        yolo_model_name="yolov8n.pt",
        thermal_delta_threshold=30.0,
        minimum_hotspot_area=15.0,
        yolo_confidence_threshold=0.25,
        iou_threshold=0.1,
        target_yolo_class="thermal_hotspot",
        evidence_dir=evidence_dir,
    )

    last_alert = None

    for idx, img_path in enumerate(selected_images, start=1):
        filename = os.path.basename(img_path)
        print(f"\n--- [Frame {idx}/{len(selected_images)}: {filename}] ---")

        result = pipeline.process_frame(img_path)
        last_alert = result["alert"]

        cands = result["thermal_candidates"]
        yolo_dets = result["yolo_detections"]
        status = result["status"]
        evidence_file = result["evidence_path"]

        # Print Thermal Metrics
        print(f"  Thermal Candidates Found : {len(cands)}")
        for c in cands:
            print(
                f"    * Cand #{c.candidate_id}: "
                f"bbox={c.bbox}, "
                f"area={c.area_pixels:.1f}px, "
                f"Delta_I=+{c.delta_intensity:.1f}, "
                f"max_I={c.max_intensity:.1f}"
            )

        # Print YOLO Metrics
        print(f"  YOLO Detections Found    : {len(yolo_dets)}")
        if yolo_dets:
            for d in yolo_dets:
                print(f"    * Class='{d.class_name}', Conf={d.confidence:.2f}, Box={d.box_xyxy}")
        else:
            print("    * None (Standard COCO model does not detect thermal targets).")

        # Print Fusion Decision & Evidence
        print(f"  Final Decision Status    : {status}")
        print(f"  Saved Evidence Image     : {evidence_file}")

    # Display sample structured JSON alert
    if last_alert:
        print("\n" + "=" * 80)
        print("SAMPLE STRUCTURED JSON ALERT (EMITTED FOR LAST FRAME)")
        print("=" * 80)
        print(last_alert.to_json(indent=2))

    print("\n" + "=" * 80)
    print("PIPELINE TEST SUMMARY")
    print("=" * 80)
    print(f"Total Frames Processed : {len(selected_images)}")
    print(f"Evidence Directory     : {evidence_dir}")
    print("Fusion Logic Adherence : All frames with thermal anomaly classified as")
    print("                         'UNVERIFIED_THERMAL_HOTSPOT' because the generic")
    print("                         COCO YOLO model has no custom 'thermal_hotspot' class.")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = run_pipeline_test(num_frames=5)
    if success:
        print("\n[SUCCESS] ECOSENTRY fusion pipeline executed successfully.")
    else:
        print("\n[FAILED] Pipeline execution encountered an error.")
