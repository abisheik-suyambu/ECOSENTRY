import time
from pathlib import Path

import cv2

from drone.camera import DroneCamera
from thermal.hotspot_detector import detect_hotspots
from verification.fire_verifier import verify_fire
from utils.evidence_saver import save_detection_evidence
from utils.alert import create_alert, send_alert


class DroneController:
    def __init__(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "hotspot-detect.gif"
        )
        self.camera = DroneCamera(source)

    def receive_command(self, command, sensor_alert):
        print("\n[DRONE] Command received")
        print(f"Command: {command['command']}")
        print(f"Target latitude: {command['latitude']}")
        print(f"Target longitude: {command['longitude']}")
        print(f"Source node: {command['node_id']}")

        if command["command"] == "INVESTIGATE":
            return self.fly_to_target(
                command["latitude"],
                command["longitude"],
                sensor_alert
            )

    def fly_to_target(self, latitude, longitude, sensor_alert):
        print("\n[DRONE] Flying to target...")
        time.sleep(2)

        print(
            f"[DRONE] Target reached: "
            f"{latitude}, {longitude}"
        )

        thermal_frame, visual_frame = self.camera.capture_frame(
            frame_number=20
        )

        print("[DRONE] Thermal image captured")
        print(
            f"[DRONE] Thermal frame shape: "
            f"{thermal_frame.shape}"
        )

        thermal_gray = cv2.cvtColor(
            thermal_frame,
            cv2.COLOR_BGR2GRAY
        )

        detections, mask = detect_hotspots(
            thermal_gray
        )

        print(
            f"[THERMAL] Hotspot candidates detected: "
            f"{len(detections)}"
        )

        for detection in detections:
            print(
                f"[THERMAL] Hotspot: "
                f"bbox={detection['bbox']}, "
                f"delta={detection['delta_intensity']:.2f}"
            )

        verification = verify_fire(detections)

        print("\n[VERIFICATION]")
        print(
            f"Status: "
            f"{verification['verification_status']}"
        )
        print(
            f"Reason: "
            f"{verification['reason']}"
        )
        print(
            f"Fire detected: "
            f"{verification['fire_detected']}"
        )

        output_path = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "drone_thermal_evidence.png"
        )

        saved_path = save_detection_evidence(
            thermal_frame,
            detections,
            output_path
        )

        print(f"[EVIDENCE] Saved: {saved_path}")

        alert = create_alert(
            sensor_alert,
            detections,
            saved_path
        )

        alert["verification"] = verification

        send_alert(alert)

        return alert