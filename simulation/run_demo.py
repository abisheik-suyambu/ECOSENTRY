"""
ECOSENTRY — Complete End-to-End Simulation Pipeline

Executes the entire 13-step disaster monitoring workflow:
1. Create simulated sensor alert (ESP32 ground node).
2. Transmit alert to LoRa Gateway.
3. LoRa Gateway generates drone investigation command.
4. Simulate drone arrival at GPS waypoint.
5. Load/process demo thermal & visual data (Demo Path: relative anomaly).
6. Generate synthetic temperature matrix for temperature-path testing (Radiometric Path: Celsius °C).
7. Detect calibrated temperature hotspots.
8. Run YOLOv8 fire/smoke detection (best.pt) on available evidence.
9. Multi-source evidence fusion (Ground + Temperature + Thermal + YOLO).
10. Generate final structured disaster alert.
11. Save evidence images:
    - results/thermal_temperature_evidence.png
    - results/thermal_hotspot_evidence.png
    - results/yolo_fire_detection.png
    - results/composite_evidence.png
12. Save results/latest_alert.json.
13. Print a clean, transparent summary.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecosentry.sensor.sensor_node import SensorNode
from ecosentry.gateway.gateway import Gateway
from ecosentry.drone.drone_controller import DroneController
from ecosentry.thermal.preprocessor import ThermalPreprocessor
from ecosentry.thermal.hotspot_detector import detect_hotspots
from ecosentry.thermal.temperature_detector import (
    detect_temperature_hotspots,
    generate_synthetic_temperature_map,
)
from ecosentry.yolo.fire_detector import FireDetector
from ecosentry.verification.fire_verifier import verify_detections
from ecosentry.fusion.evidence_fusion import fuse_evidence
from ecosentry.alerts.alert_manager import AlertManager
from ecosentry.utils.evidence_saver import (
    save_temperature_evidence,
    save_thermal_evidence,
    save_yolo_evidence,
    save_composite_evidence,
)


def run_pipeline():
    print("\n" + "=" * 70)
    print("      ECOSENTRY: RESILIENT AI-POWERED DISASTER MONITORING     ")
    print("             SIH26178 — End-to-End System Demonstration        ")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 1: SENSOR NODE ALERT GENERATION (SIMULATED ESP32)
    # -------------------------------------------------------------
    print("\n[STEP 1] SENSOR NODE ALERT GENERATION (SIMULATED ESP32)")
    sensor = SensorNode(
        node_id="NODE_01",
        latitude=12.9716,
        longitude=80.2209,
        temp_threshold=50.0,
        gas_threshold=60.0,
        water_min_threshold=20.0,
    )
    sensor_alert = sensor.simulate_hotspot()
    print(f"  Station ID       : {sensor_alert['node_id']}")
    print(f"  Coordinates      : {sensor_alert['latitude']}, {sensor_alert['longitude']}")
    print(f"  Temperature      : {sensor_alert['temperature']} (Threshold: 50.0)")
    print(f"  Gas Level        : {sensor_alert['gas_level']} (Threshold: 60.0)")
    print(f"  Water Level      : {sensor_alert['water_level']}% (Min Safe: 20.0%)")
    print(f"  Sensor Status    : {sensor_alert['status']}")

    # -------------------------------------------------------------
    # STEP 2 & 3: LoRa GATEWAY RECEPTION & INVESTIGATION COMMAND
    # -------------------------------------------------------------
    print("\n[STEP 2 & 3] LoRa BASE STATION GATEWAY & DISPATCH")
    gateway = Gateway(gateway_id="GW_CENTRAL_01")
    drone_command = gateway.receive_packet(sensor_alert)

    if not drone_command:
        print("  System status normal. No UAV intervention required.")
        return

    # -------------------------------------------------------------
    # STEP 4: DRONE SIMULATION ARRIVAL
    # -------------------------------------------------------------
    print("\n[STEP 4] DRONE FLIGHT SIMULATION & WAYPOINT ARRIVAL")
    drone = DroneController()
    thermal_frame, visual_frame, mission_log = drone.receive_command(
        drone_command, simulation_delay=1.0
    )

    # -------------------------------------------------------------
    # STEP 5: LOAD & PROCESS DEMO THERMAL DATA (RELATIVE PATH)
    # -------------------------------------------------------------
    print("\n[STEP 5] DEMO THERMAL ANOMALY PATH (RELATIVE INTENSITY & CLAHE)")
    preprocessor = ThermalPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
    prep_data = preprocessor.process(thermal_frame)
    relative_hotspots, _ = detect_hotspots(prep_data["enhanced"], k=2.5, minimum_area=150)
    print(f"  Relative Thermal Anomalies Isolated : {len(relative_hotspots)}")
    if relative_hotspots:
        strongest_rel = relative_hotspots[0]
        print(f"  >>> Strongest Anomaly: BBox={strongest_rel['bbox']} | Delta I=+{strongest_rel['delta_intensity']:.1f}")

    # -------------------------------------------------------------
    # STEP 6 & 7: SYNTHETIC RADIOMETRIC TEMPERATURE MATRIX & HOTSPOTS
    # -------------------------------------------------------------
    print("\n[STEP 6 & 7] RADIOMETRIC TEMPERATURE PATH (CELSIUS °C)")
    print("  [NOTE] Generating deterministic synthetic radiometric 2D temperature array")
    print("         to test future FLIR Lepton / MLX90640 radiometric hardware integration.")

    temp_map = generate_synthetic_temperature_map(
        shape=(480, 640),
        ambient_c=34.0,
        hotspot_specs=[
            {"center": (160, 220), "radius": 40, "peak_c": 128.5},
            {"center": (310, 440), "radius": 50, "peak_c": 287.4},
            {"center": (350, 160), "radius": 30, "peak_c": 84.2},
        ],
        seed=42,
    )

    temp_hotspots, temp_mask = detect_temperature_hotspots(
        temperature_map=temp_map,
        threshold_c=70.0,
        minimum_area=15,
    )

    print(f"  Temperature Hotspots Exceeding Threshold (70.0°C): {len(temp_hotspots)}")
    for idx, hs in enumerate(temp_hotspots, start=1):
        print(
            f"    Hotspot #{idx}: BBox={hs['bbox']}, "
            f"Peak={hs['max_temperature_c']:.1f}°C, "
            f"Mean={hs['mean_temperature_c']:.1f}°C, "
            f"Area={hs['area']:.0f}px"
        )

    if temp_hotspots:
        hottest = temp_hotspots[0]
        print(f"  >>> HOTTEST HOTSPOT: {hottest['max_temperature_c']:.1f}°C at {hottest['bbox']}")

    # -------------------------------------------------------------
    # STEP 8: YOLOv8 FIRE & SMOKE INFERENCE (best.pt)
    # -------------------------------------------------------------
    print("\n[STEP 8] YOLOv8 VISUAL AI INFERENCE (MODEL: best.pt)")
    fire_detector = FireDetector()
    yolo_detections_vis = fire_detector.detect(visual_frame, confidence=0.25)
    yolo_detections_th = fire_detector.detect(thermal_frame, confidence=0.25)
    yolo_detections = yolo_detections_vis + yolo_detections_th

    print(f"  Total YOLO Predictions Detected: {len(yolo_detections)}")
    for det in yolo_detections:
        print(
            f"    Prediction: Class '{det['class_name']}' | "
            f"Confidence: {det['confidence']:.2f} | Bounding Box: {det['bbox']}"
        )

    # -------------------------------------------------------------
    # STEP 9: MULTI-SOURCE EVIDENCE FUSION & VERIFICATION
    # -------------------------------------------------------------
    print("\n[STEP 9] MULTI-SOURCE EVIDENCE FUSION & SPATIAL VERIFICATION")
    verification_data = verify_detections(relative_hotspots, yolo_detections)
    fusion_data = fuse_evidence(
        sensor_packet=sensor_alert,
        thermal_hotspots=relative_hotspots,
        yolo_detections=yolo_detections,
        verification_data=verification_data,
        temperature_hotspots=temp_hotspots,
    )

    print(f"  Alert Level        : {fusion_data['alert_level']}")
    print(f"  Assessment         : {fusion_data['assessment']}")
    print(f"  Vegetation Risk    : {fusion_data['vegetation_fire_risk']}")
    print(f"  Corroborated Pairs : {verification_data['corroborated_count']}")

    # -------------------------------------------------------------
    # STEP 10, 11 & 12: SAVE EVIDENCE IMAGES & LATEST ALERT JSON
    # -------------------------------------------------------------
    print("\n[STEP 10, 11 & 12] SAVING AUDITABLE EVIDENCE & STRUCTURED JSON")
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    temp_evidence_path = results_dir / "thermal_temperature_evidence.png"
    thermal_evidence_path = results_dir / "thermal_hotspot_evidence.png"
    yolo_evidence_path = results_dir / "yolo_fire_detection.png"
    composite_evidence_path = results_dir / "composite_evidence.png"

    # Save individual evidence images
    save_temperature_evidence(temp_map, temp_hotspots, temp_evidence_path, is_synthetic=True)
    save_thermal_evidence(thermal_frame, relative_hotspots, thermal_evidence_path)
    save_yolo_evidence(
        thermal_frame if yolo_detections_th else visual_frame,
        yolo_detections,
        yolo_evidence_path,
    )

    # Compile structured alert
    alert_mgr = AlertManager(results_dir=results_dir)
    alert_payload = alert_mgr.create_alert(
        sensor_packet=sensor_alert,
        thermal_hotspots=relative_hotspots,
        yolo_detections=yolo_detections,
        verification_data=verification_data,
        fusion_data=fusion_data,
        evidence_path=str(composite_evidence_path.name),
        temperature_hotspots=temp_hotspots,
    )

    # Save composite evidence including temperature panel
    save_composite_evidence(
        thermal_img_path=thermal_evidence_path,
        yolo_img_path=yolo_evidence_path,
        alert_info=alert_payload,
        output_path=composite_evidence_path,
        temp_img_path=temp_evidence_path,
    )

    alert_file = alert_mgr.save_alert(alert_payload)

    print(f"  Temperature Evidence Saved : {temp_evidence_path}")
    print(f"  Thermal Hotspot Saved      : {thermal_evidence_path}")
    print(f"  YOLO Fire Evidence Saved   : {yolo_evidence_path}")
    print(f"  Composite Evidence Saved   : {composite_evidence_path}")
    print(f"  Structured Alert JSON      : {alert_file}")

    # -------------------------------------------------------------
    # STEP 13: PRINT CLEAN SUMMARY
    # -------------------------------------------------------------
    alert_mgr.print_alert_summary(alert_payload)

    print("=" * 70)
    print("To launch the live command center dashboard, run:")
    print("  python ecosentry/dashboard/app.py")
    print("Then open your browser at: http://localhost:5000\n")


if __name__ == "__main__":
    run_pipeline()
