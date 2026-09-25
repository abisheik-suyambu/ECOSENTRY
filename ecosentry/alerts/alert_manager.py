"""
Alert Manager Module

Generates standardized disaster alerts in accordance with ECOSENTRY specification:
{
    "status": "...",
    "node_id": "...",
    "latitude": ...,
    "longitude": ...,
    "sensor_temperature": ...,
    "gas_level": ...,
    "water_level": ...,
    "hotspot_count": ...,
    "temperature_max_c": ...,
    "fire_detections": ...,
    "smoke_detections": ...,
    "verification_status": "...",
    "evidence_path": "...",
    "timestamp": "..."
}
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class AlertManager:
    """
    Central alert builder and dispatcher for ECOSENTRY.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        if results_dir is None:
            self.results_dir = Path("results")
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_alert(
        self,
        sensor_packet: Dict[str, Any],
        thermal_hotspots: List[Dict[str, Any]],
        yolo_detections: List[Dict[str, Any]],
        verification_data: Dict[str, Any],
        fusion_data: Dict[str, Any],
        evidence_path: str,
        temperature_hotspots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble the final structured alert JSON with genuine runtime metrics.
        """
        if temperature_hotspots is None:
            temperature_hotspots = []

        fire_dets = [d for d in yolo_detections if d.get("class_name") == "fire"]
        smoke_dets = [d for d in yolo_detections if d.get("class_name") == "smoke"]

        hottest_temp = fusion_data.get("temperature_max_c")
        if hottest_temp is None and temperature_hotspots:
            hottest_temp = temperature_hotspots[0].get("max_temperature_c")

        alert = {
            "status": fusion_data.get("alert_level", "NORMAL"),
            "assessment": fusion_data.get("assessment", ""),
            "node_id": sensor_packet.get("node_id", "UNKNOWN"),
            "latitude": float(sensor_packet.get("latitude", 0.0)),
            "longitude": float(sensor_packet.get("longitude", 0.0)),
            "sensor_temperature": float(sensor_packet.get("temperature", 0.0)),
            "gas_level": float(sensor_packet.get("gas_level", 0.0)),
            "water_level": float(sensor_packet.get("water_level", 0.0)),
            "hotspot_count": len(thermal_hotspots),
            "temperature_hotspot_count": len(temperature_hotspots),
            "temperature_max_c": round(float(hottest_temp), 2) if hottest_temp is not None else None,
            "fire_detections": len(fire_dets),
            "smoke_detections": len(smoke_dets),
            "verification_status": verification_data.get("verification_status", "UNKNOWN"),
            "evidence_path": str(evidence_path),
            "timestamp": datetime.now().isoformat(),
            "details": {
                "corroborations": verification_data.get("corroborated_count", 0),
                "vegetation_fire_risk": fusion_data.get("vegetation_fire_risk", False),
                "strongest_relative_delta": (
                    thermal_hotspots[0].get("delta_intensity") if thermal_hotspots else 0.0
                ),
                "temperature_hotspots": temperature_hotspots,
                "yolo_predictions": yolo_detections,
            },
        }

        return alert

    def save_alert(
        self,
        alert: Dict[str, Any],
        filename: str = "latest_alert.json",
    ) -> Path:
        """
        Save the alert to results/latest_alert.json.
        """
        output_path = self.results_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alert, f, indent=4)
        return output_path

    def print_alert_summary(self, alert: Dict[str, Any]) -> None:
        """
        Print a clean, beginner-friendly terminal alert summary.
        """
        print("\n" + "=" * 65)
        print("          ECOSENTRY DISASTER MONITORING ALERT          ")
        print("=" * 65)
        print(f"  ALERT LEVEL         : {alert['status']}")
        print(f"  VERIFICATION STATUS : {alert['verification_status']}")
        print(f"  ASSESSMENT          : {alert['assessment']}")
        print("-" * 65)
        print(f"  Node ID             : {alert['node_id']}")
        print(f"  Location (GPS)      : {alert['latitude']}, {alert['longitude']}")
        print(f"  Sensor Temp / Gas   : {alert['sensor_temperature']} / {alert['gas_level']}")
        print(f"  Sensor Water Level  : {alert['water_level']}%")
        print("-" * 65)
        if alert.get("temperature_max_c") is not None:
            print(f"  Radiometric Peak    : {alert['temperature_max_c']} °C ({alert['temperature_hotspot_count']} hotspots)")
        print(f"  Thermal Hotspots    : {alert['hotspot_count']} relative anomaly candidates")
        print(f"  YOLO Fire / Smoke   : {alert['fire_detections']} Fire / {alert['smoke_detections']} Smoke")
        print(f"  Evidence Image      : {alert['evidence_path']}")
        print(f"  Timestamp           : {alert['timestamp']}")
        print("=" * 65 + "\n")
