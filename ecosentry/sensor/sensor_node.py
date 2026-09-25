"""
Sensor Node Module (Simulating ESP32 + Environmental Sensors)

Monitors:
- Temperature (in arbitrary environmental units / Celsius simulation)
- Gas Level (PPM or air quality index proxy)
- Water Level (%)

When values cross thresholds, generates an alert packet to be transmitted over LoRa.
"""

from datetime import datetime
from typing import Dict, Any


class SensorNode:
    """
    Simulates a ground-deployed ESP32 sensor station.
    """

    def __init__(
        self,
        node_id: str = "NODE_01",
        latitude: float = 12.9716,
        longitude: float = 80.2209,
        temp_threshold: float = 50.0,
        gas_threshold: float = 60.0,
        water_min_threshold: float = 20.0,
    ):
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        self.temp_threshold = temp_threshold
        self.gas_threshold = gas_threshold
        self.water_min_threshold = water_min_threshold

    def evaluate_reading(
        self,
        temperature: float,
        gas_level: float,
        water_level: float
    ) -> Dict[str, Any]:
        """
        Check sensor values against safety thresholds.
        Returns an alert payload if abnormal, or a normal status payload.
        """
        is_abnormal = (
            temperature >= self.temp_threshold
            or gas_level >= self.gas_threshold
            or water_level <= self.water_min_threshold
        )

        status = "ABNORMAL_ENVIRONMENTAL_CONDITION" if is_abnormal else "NORMAL"

        return {
            "node_id": self.node_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "temperature": float(temperature),
            "gas_level": float(gas_level),
            "water_level": float(water_level),
            "status": status,
            "is_abnormal": is_abnormal,
            "timestamp": datetime.now().isoformat()
        }

    def simulate_hotspot(self) -> Dict[str, Any]:
        """
        Simulate an abnormal reading (high temp, elevated gas, dry water).
        Demo values: Temperature = 68.5, Gas = 82.0, Water = 15.0.
        """
        return self.evaluate_reading(
            temperature=68.5,
            gas_level=82.0,
            water_level=15.0
        )

    def simulate_normal(self) -> Dict[str, Any]:
        """
        Simulate a normal baseline environmental reading.
        Demo values: Temperature = 28.0, Gas = 22.0, Water = 75.0.
        """
        return self.evaluate_reading(
            temperature=28.0,
            gas_level=22.0,
            water_level=75.0
        )
