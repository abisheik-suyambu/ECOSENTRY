"""
Test script for ECOSENTRY Lightweight YOLO Detector (Inference-Only Stage).

Responsibilities:
1. Loads sample thermal frames from ecosentry/datasets/sample_thermal/
2. Initializes the lightweight Ultralytics YOLO model (e.g., yolov8n.pt).
3. Executes inference on the sample thermal frames.
4. Saves annotated visual detection results to ecosentry/results/yolo_test/
5. Reports model configuration, detections, bounding boxes, and confidence scores.

Important Scientific Context:
- The sample frames are thermal visualization images from UAV solar patrol, not labelled
  forest-fire data.
- The standard pretrained YOLO model is trained on the general COCO 80-class dataset
  (e.g., person, vehicle, etc.), NOT on thermal hotspots or wildfires.
- Therefore, no detection is classified as a fire or thermal hotspot.
- If the model finds 0 objects, this is reported honestly as expected behavior for uncalibrated
  thermal imagery tested against generic COCO weights.
"""

import os
import sys
import glob
import cv2

# Ensure workspace root is in sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecosentry.models.detector import YOLODetector


def run_yolo_test(
    model_name: str = "yolov8n.pt",
    num_frames: int = 5,
    conf_threshold: float = 0.25,
    img_size: int = 640,
):
    """
    Execute YOLO inference test on sample thermal frames.

    Args:
        model_name: Model identifier (default: 'yolov8n.pt').
        num_frames: Number of sample frames to evaluate (default: 5).
        conf_threshold: Confidence threshold for filtering detections (default: 0.25).
        img_size: Inference input dimension (default: 640).
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sample_dir = os.path.join(base_dir, "ecosentry", "datasets", "sample_thermal")
    output_dir = os.path.join(base_dir, "ecosentry", "results", "yolo_test")

    os.makedirs(output_dir, exist_ok=True)

    image_pattern = os.path.join(sample_dir, "*.png")
    all_images = sorted(glob.glob(image_pattern))

    if not all_images:
        print(f"[ERROR] No sample thermal images found in: {sample_dir}")
        return False

    selected_images = all_images[:num_frames]

    print("=" * 75)
    print("ECOSENTRY LIGHTWEIGHT YOLO INFERENCE TEST")
    print("=" * 75)
    print(f"Model Identifier        : {model_name}")
    print(f"Confidence Threshold    : {conf_threshold}")
    print(f"Inference Image Size    : {img_size}")
    print(f"Sample Directory        : {sample_dir}")
    print(f"Output Directory        : {output_dir}")
    print(f"Total Frames Available  : {len(all_images)}")
    print(f"Frames to Test          : {len(selected_images)}")
    print("=" * 75)

    # 1. Initialize YOLO Detector
    try:
        detector = YOLODetector(
            model_name=model_name,
            conf_threshold=conf_threshold,
            img_size=img_size,
        )
        print(f"[SUCCESS] Model '{model_name}' initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize YOLO model: {e}")
        return False

    total_detections_all_images = 0

    # 2. Iterate through sample frames
    for idx, img_path in enumerate(selected_images, start=1):
        filename = os.path.basename(img_path)
        print(f"\n--- [Frame {idx}/{len(selected_images)}: {filename}] ---")

        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [ERROR] Could not read image: {img_path}")
            continue

        h, w = img_bgr.shape[:2]
        print(f"  Frame dimensions : {w}x{h}")

        # Run inference
        detections = detector.predict(img_bgr)
        num_dets = len(detections)
        total_detections_all_images += num_dets

        print(f"  Detections found : {num_dets}")

        if num_dets > 0:
            for d_idx, det in enumerate(detections, start=1):
                print(
                    f"    * Det #{d_idx}: "
                    f"Class='{det.class_name}' (ID: {det.class_id}) | "
                    f"Confidence={det.confidence:.4f} | "
                    f"Box(xyxy)={det.box_xyxy} | "
                    f"Box(xywh)={det.box_xywh}"
                )
                print(
                    f"      [NOTE] Class '{det.class_name}' is a generic COCO category; "
                    "not a calibrated thermal hotspot label."
                )
        else:
            print(
                "    * No COCO objects detected at confidence >= "
                f"{conf_threshold} (Expected for raw thermal inspection imagery)."
            )

        # Draw annotations and save
        annotated = detector.annotate(img_bgr, detections)

        # Add top banner
        banner_text = (
            f"YOLO: {model_name} | Frame: {filename} | "
            f"Detections: {num_dets} (COCO Classes)"
        )
        cv2.putText(
            annotated,
            banner_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

        out_path = os.path.join(output_dir, f"yolo_{filename}")
        cv2.imwrite(out_path, annotated)
        print(f"  Saved visual result -> {out_path}")

    print("\n" + "=" * 75)
    print("YOLO TEST SUMMARY")
    print("=" * 75)
    print(f"Model Evaluated             : {model_name}")
    print(f"Frames Processed            : {len(selected_images)}")
    print(f"Total Detections Found      : {total_detections_all_images}")
    print(f"Output Directory            : {output_dir}")
    print("Status                      : Inference completed successfully.")
    print("=" * 75)

    return True


if __name__ == "__main__":
    success = run_yolo_test(
        model_name="yolov8n.pt",
        num_frames=5,
        conf_threshold=0.25,
        img_size=640,
    )
    if success:
        print("\n[SUCCESS] YOLO test pipeline completed.")
    else:
        print("\n[FAILED] YOLO test pipeline encountered an error.")
