"""
ECOSENTRY Subsystem Verification & Comprehensive Test Suite

Tests all 10 core subsystems:
1. Sensor Alert Generation (ESP32 Ground Station)
2. Gateway Investigation Command (LoRa Dispatch)
3. Temperature Hotspot Detection (Calibrated Celsius Matrix with 80°C, 120°C, and 280°C tests)
4. Relative Thermal Hotspot Detection (Standardized Intensity Delta)
5. YOLO Model Loading (ecosentry/models/best.pt)
6. YOLO Fire/Smoke Inference
7. Evidence Fusion & Spatial Verification
8. Alert JSON Generation & Schema Conformance
9. Evidence Image Generation (Temperature, Hotspot, YOLO, and Composite)
10. Full End-to-End Simulation Pipeline
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ecosentry.sensor.sensor_node import SensorNode
from ecosentry.gateway.gateway import Gateway
from ecosentry.drone.camera import DroneCamera
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


def test_1_sensor_simulation():
    print("[TEST 1] Sensor Alert Generation...", end=" ")
    sensor = SensorNode(node_id="NODE_TEST_01")
    alert = sensor.simulate_hotspot()
    assert alert["node_id"] == "NODE_TEST_01"
    assert alert["temperature"] == 68.5
    assert alert["gas_level"] == 82.0
    assert alert["water_level"] == 15.0
    assert alert["is_abnormal"] is True
    print("PASSED")
    return alert


def test_2_gateway_alert(sensor_alert):
    print("[TEST 2] Gateway Investigation Command Generation...", end=" ")
    gateway = Gateway(gateway_id="GW_TEST_01")
    command = gateway.receive_packet(sensor_alert)
    assert command is not None
    assert command["command"] == "INVESTIGATE"
    assert command["node_id"] == "NODE_TEST_01"
    print("PASSED")
    return command


def test_3_temperature_hotspot_detection():
    print("[TEST 3] Temperature Hotspot Detection (Deterministic Celsius Matrix)...", end=" ")

    # Scenario A: Moderate Hotspots (Background ~35°C, Hotspots at 80°C and 120°C)
    temp_map_a = np.full((100, 100), 35.0, dtype=np.float32)
    temp_map_a[20:35, 20:35] = 80.0
    temp_map_a[60:75, 65:80] = 120.0

    hotspots_a, mask_a = detect_temperature_hotspots(temp_map_a, threshold_c=70.0, minimum_area=15)
    assert len(hotspots_a) == 2, f"Expected 2 hotspots, got {len(hotspots_a)}"
    # Verified ranked descending by peak temp
    assert hotspots_a[0]["max_temperature_c"] == 120.0
    assert hotspots_a[1]["max_temperature_c"] == 80.0
    assert hotspots_a[0]["threshold_c"] == 70.0

    # Scenario B: Extreme Incipient Blaze (Hotter scenario: 250°C - 300°C)
    temp_map_b = np.full((100, 100), 38.0, dtype=np.float32)
    temp_map_b[40:60, 40:60] = 285.5

    hotspots_b, _ = detect_temperature_hotspots(temp_map_b, threshold_c=100.0, minimum_area=15)
    assert len(hotspots_b) == 1
    assert hotspots_b[0]["max_temperature_c"] == 285.5

    print("PASSED (Tested 80°C, 120°C, and 285.5°C synthetic scenarios)")
    return hotspots_a, temp_map_a


def test_4_relative_thermal_hotspots():
    print("[TEST 4] Relative Thermal Hotspot Detection (Demo Image Path)...", end=" ")
    camera = DroneCamera()
    th, vis = camera.capture_frame(frame_number=20)
    assert th is not None and vis is not None

    hotspots, mask = detect_hotspots(th, k=2.5, minimum_area=150)
    assert len(hotspots) > 0
    strongest = hotspots[0]
    assert "delta_intensity" in strongest
    assert strongest["delta_intensity"] > 0
    print(f"PASSED ({len(hotspots)} relative anomalies, Peak delta: +{strongest['delta_intensity']:.1f})")
    return th, vis, hotspots


def test_5_yolo_model_loading():
    print("[TEST 5] YOLO Model Loading (ecosentry/models/best.pt)...", end=" ")
    detector = FireDetector()
    assert detector.names[0] == "fire"
    assert detector.names[1] == "smoke"
    print(f"PASSED (Classes: {detector.names})")
    return detector


def test_6_yolo_detection(detector, th_frame, vis_frame):
    print("[TEST 6] YOLO Fire/Smoke Inference...", end=" ")
    dets_vis = detector.detect(vis_frame, confidence=0.25)
    dets_th = detector.detect(th_frame, confidence=0.25)
    all_dets = dets_vis + dets_th
    assert isinstance(all_dets, list)
    print(f"PASSED ({len(all_dets)} genuine detections found)")
    return all_dets


def test_7_evidence_fusion(sensor_alert, rel_hotspots, temp_hotspots, yolo_dets):
    print("[TEST 7] Multi-Source Evidence Fusion & Verification...", end=" ")
    verification = verify_detections(rel_hotspots, yolo_dets)
    fusion = fuse_evidence(
        sensor_packet=sensor_alert,
        thermal_hotspots=rel_hotspots,
        yolo_detections=yolo_dets,
        verification_data=verification,
        temperature_hotspots=temp_hotspots,
    )
    assert "alert_level" in fusion
    assert fusion["alert_level"] in [
        "NORMAL",
        "WARNING",
        "THERMAL ANOMALY",
        "POSSIBLE FIRE",
        "FIRE / SMOKE ALERT",
    ]
    print(f"PASSED (Result: {fusion['alert_level']})")
    return verification, fusion


def test_8_alert_json_generation(sensor_alert, rel_hotspots, temp_hotspots, yolo_dets, verification, fusion):
    print("[TEST 8] Alert JSON Generation & Schema Conformance...", end=" ")
    alert_mgr = AlertManager()
    alert_payload = alert_mgr.create_alert(
        sensor_packet=sensor_alert,
        thermal_hotspots=rel_hotspots,
        yolo_detections=yolo_dets,
        verification_data=verification,
        fusion_data=fusion,
        evidence_path="composite_evidence.png",
        temperature_hotspots=temp_hotspots,
    )
    assert alert_payload["node_id"] == "NODE_TEST_01"
    assert alert_payload["sensor_temperature"] == 68.5
    assert "temperature_max_c" in alert_payload
    assert "timestamp" in alert_payload

    saved_file = alert_mgr.save_alert(alert_payload, filename="test_alert_validation.json")
    assert saved_file.exists()
    saved_file.unlink()  # Clean up temporary test file
    print("PASSED")


def test_9_evidence_image_generation(temp_map, temp_hotspots, th_frame, rel_hotspots, yolo_dets, sensor_alert, verification, fusion):
    print("[TEST 9] Evidence Image Generation (All 4 Artifacts)...", end=" ")
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    p1 = save_temperature_evidence(temp_map, temp_hotspots, results_dir / "thermal_temperature_evidence.png", is_synthetic=True)
    p2 = save_thermal_evidence(th_frame, rel_hotspots, results_dir / "thermal_hotspot_evidence.png")
    p3 = save_yolo_evidence(th_frame, yolo_dets, results_dir / "yolo_fire_detection.png")

    alert_mgr = AlertManager(results_dir=results_dir)
    alert_payload = alert_mgr.create_alert(
        sensor_packet=sensor_alert,
        thermal_hotspots=rel_hotspots,
        yolo_detections=yolo_dets,
        verification_data=verification,
        fusion_data=fusion,
        evidence_path="composite_evidence.png",
        temperature_hotspots=temp_hotspots,
    )

    p4 = save_composite_evidence(p2, p3, alert_payload, results_dir / "composite_evidence.png", temp_img_path=p1)

    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert p4.exists()
    print("PASSED")


def test_10_full_pipeline():
    print("[TEST 10] Full End-to-End Demo Script Execution...", end=" ")
    from simulation.run_demo import run_pipeline
    run_pipeline()
    assert (ROOT_DIR / "results" / "latest_alert.json").exists()
    assert (ROOT_DIR / "results" / "thermal_temperature_evidence.png").exists()
    assert (ROOT_DIR / "results" / "thermal_hotspot_evidence.png").exists()
    assert (ROOT_DIR / "results" / "yolo_fire_detection.png").exists()
    assert (ROOT_DIR / "results" / "composite_evidence.png").exists()
    print("PASSED")


def main():
    print("\n" + "=" * 68)
    print("      ECOSENTRY SUBSYSTEM VERIFICATION & COMPREHENSIVE TESTS      ")
    print("=" * 68 + "\n")

    alert = test_1_sensor_simulation()
    cmd = test_2_gateway_alert(alert)
    temp_hotspots, temp_map = test_3_temperature_hotspot_detection()
    th, vis, rel_hotspots = test_4_relative_thermal_hotspots()
    detector = test_5_yolo_model_loading()
    yolo_dets = test_6_yolo_detection(detector, th, vis)
    veri, fusion = test_7_evidence_fusion(alert, rel_hotspots, temp_hotspots, yolo_dets)
    test_8_alert_json_generation(alert, rel_hotspots, temp_hotspots, yolo_dets, veri, fusion)
    test_9_evidence_image_generation(temp_map, temp_hotspots, th, rel_hotspots, yolo_dets, alert, veri, fusion)
    test_10_full_pipeline()

    print("\n" + "=" * 68)
    print("      ALL 10 CORE SUBSYSTEM TESTS PASSED WITH ZERO ERRORS!       ")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
