"""
Evidence Fusion Module for ECOSENTRY

Synthesizes multiple multi-modal observation streams:
1. Ground Sensor Telemetry (Temperature, Gas Level, Water Level)
2. Radiometric Temperature Hotspots (Physical Celsius values in °C)
3. Relative Thermal Anomalies (Normalized delta intensity from demo/TIR frames)
4. YOLOv8 Visual AI Predictions (Fire & Smoke detections with confidence)
5. Environmental Risk Evaluation (Dry biomass & vegetation fire hazard)

Transparent Decision Rules:
- High Temperature / Thermal Hotspot + YOLO Fire -> FIRE / SMOKE ALERT (High Confidence)
- High Temperature + YOLO Smoke -> FIRE / SMOKE ALERT (Incipient Fire / Smoldering)
- Thermal Hotspot without YOLO Fire/Smoke -> THERMAL ANOMALY (Requires investigation; could be solar-heated roof/engine)
- YOLO Fire without Thermal Anomaly -> POSSIBLE FIRE (Visual fire detected, thermal confirmation pending)
- Sensor Threshold Violation alone -> WARNING (Sensor anomaly detected; drone dispatched to verify)
- Normal Baseline -> NORMAL
"""

from typing import Dict, Any, List, Optional


class EvidenceFusion:
    """
    Transparent multi-stream decision engine for disaster management.
    """

    def fuse(
        self,
        sensor_packet: Dict[str, Any],
        thermal_hotspots: List[Dict[str, Any]],
        yolo_detections: List[Dict[str, Any]],
        verification_data: Dict[str, Any],
        temperature_hotspots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize ground sensors, radiometric temperature, relative thermal anomalies,
        and YOLO predictions into an auditable disaster alert level.
        """
        if temperature_hotspots is None:
            temperature_hotspots = []

        # 1. Ground Sensor Evaluation
        temp = float(sensor_packet.get("temperature", 0.0))
        gas = float(sensor_packet.get("gas_level", 0.0))
        water = float(sensor_packet.get("water_level", 100.0))

        sensor_abnormal = (
            sensor_packet.get("is_abnormal", False)
            or temp >= 50.0
            or gas >= 60.0
            or water <= 20.0
        )

        # 2. Radiometric Temperature Evaluation (Celsius °C)
        has_temp_hotspots = len(temperature_hotspots) > 0
        hottest_temp_c = (
            temperature_hotspots[0].get("max_temperature_c", 0.0) if has_temp_hotspots else None
        )
        is_high_temp = hottest_temp_c is not None and hottest_temp_c >= 70.0

        # 3. Relative Thermal Anomaly Evaluation
        has_relative_hotspots = len(thermal_hotspots) > 0
        strongest_relative_delta = (
            float(thermal_hotspots[0].get("delta_intensity", 0.0)) if has_relative_hotspots else 0.0
        )

        has_thermal_anomaly = has_temp_hotspots or has_relative_hotspots

        # 4. YOLO AI Evaluation
        fire_predictions = [d for d in yolo_detections if d.get("class_name") == "fire"]
        smoke_predictions = [d for d in yolo_detections if d.get("class_name") == "smoke"]
        has_fire = len(fire_predictions) > 0
        has_smoke = len(smoke_predictions) > 0

        max_yolo_conf = 0.0
        if yolo_detections:
            max_yolo_conf = max(d.get("confidence", 0.0) for d in yolo_detections)

        # 5. Environmental Risk Context
        dry_biomass_flag = water <= 20.0 and temp >= 45.0
        vegetation_risk_flag = False

        # 6. Transparent Alert Rule Determination
        if has_thermal_anomaly and (has_fire or has_smoke):
            alert_level = "FIRE / SMOKE ALERT"
            assessment = "HIGH CONFIDENCE: Thermal anomaly corroborated by YOLO visual detection."
            if is_high_temp:
                assessment += f" Radiometric peak temperature confirmed at {hottest_temp_c:.1f}°C."
            if sensor_abnormal:
                assessment += " Ground sensors also confirm abnormal temperature/gas levels."

        elif has_fire or has_smoke:
            alert_level = "POSSIBLE FIRE"
            assessment = (
                "YOLO detected visual fire/smoke signature, but thermal anomaly "
                "is weak or unconfirmed. Requires closer inspection."
            )

        elif has_thermal_anomaly:
            if dry_biomass_flag:
                alert_level = "THERMAL ANOMALY"
                vegetation_risk_flag = True
                assessment = (
                    "POTENTIAL VEGETATION FIRE RISK: Thermal anomaly isolated under "
                    "critically dry and hot environmental conditions. Smoldering risk."
                )
            else:
                alert_level = "THERMAL ANOMALY"
                assessment = (
                    "Thermal anomaly detected; awaiting visual fire/smoke confirmation. "
                    "Could be a non-fire heat source (e.g. heated roof, vehicle, machinery, sunlight glint)."
                )

        elif sensor_abnormal:
            alert_level = "WARNING"
            assessment = (
                "Ground sensors report abnormal readings, but drone aerial imagery "
                "shows no visible thermal hotspots or fire/smoke."
            )
        else:
            alert_level = "NORMAL"
            assessment = "All environmental, thermal, and visual indicators are within baseline."

        return {
            "alert_level": alert_level,
            "assessment": assessment,
            "vegetation_fire_risk": vegetation_risk_flag,
            "temperature_max_c": hottest_temp_c,
            "evidence_summary": {
                "sensor_evidence": {
                    "temperature": temp,
                    "gas_level": gas,
                    "water_level": water,
                    "sensor_abnormal": sensor_abnormal,
                },
                "temperature_evidence_celsius": {
                    "count": len(temperature_hotspots),
                    "hottest_temp_c": hottest_temp_c,
                    "hotspots": temperature_hotspots,
                },
                "relative_thermal_evidence": {
                    "count": len(thermal_hotspots),
                    "strongest_delta": round(strongest_relative_delta, 2),
                    "hotspots": thermal_hotspots,
                },
                "yolo_evidence": {
                    "total_detections": len(yolo_detections),
                    "fire_count": len(fire_predictions),
                    "smoke_count": len(smoke_predictions),
                    "max_confidence": round(max_yolo_conf, 4),
                    "detections": yolo_detections,
                },
                "corroborated_matches": verification_data.get("corroborated_count", 0),
            },
        }


def fuse_evidence(
    sensor_packet: Dict[str, Any],
    thermal_hotspots: List[Dict[str, Any]],
    yolo_detections: List[Dict[str, Any]],
    verification_data: Dict[str, Any],
    temperature_hotspots: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for evidence fusion."""
    fusion = EvidenceFusion()
    return fusion.fuse(
        sensor_packet=sensor_packet,
        thermal_hotspots=thermal_hotspots,
        yolo_detections=yolo_detections,
        verification_data=verification_data,
        temperature_hotspots=temperature_hotspots,
    )
