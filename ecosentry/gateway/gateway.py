"""
LoRa Gateway Module

Simulates a physical LoRa gateway:
1. Receives low-bandwidth radio telemetry from ESP32 sensor nodes.
2. Checks whether readings require UAV investigation.
3. Issues a structured drone dispatch command.
"""

from typing import Dict, Any, Optional


class Gateway:
    """
    Central gateway receiving LoRa telemetry packets from field sensors.
    """

    def __init__(self, gateway_id: str = "GW_CENTRAL_01"):
        self.gateway_id = gateway_id

    def receive_packet(self, packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process an incoming simulated LoRa packet.

        Args:
            packet: Dictionary containing sensor telemetry.

        Returns:
            Drone command dict if investigation needed, otherwise None.
        """
        print(f"\n[GATEWAY: {self.gateway_id}] LoRa packet received via simulated RFM95/SX1278")
        print(f"  Node ID     : {packet.get('node_id')}")
        print(f"  Coordinates : {packet.get('latitude')}, {packet.get('longitude')}")
        print(f"  Temperature : {packet.get('temperature')} (Sensor Reading)")
        print(f"  Gas Level   : {packet.get('gas_level')}")
        print(f"  Water Level : {packet.get('water_level')}%")
        print(f"  Status      : {packet.get('status')}")

        if packet.get("is_abnormal", False) or packet.get("status") == "ABNORMAL_ENVIRONMENTAL_CONDITION":
            print(f"  [ACTION] Abnormal conditions detected! Generating Drone Investigation Command...")
            return self.create_drone_command(packet)
        else:
            print(f"  [ACTION] Normal baseline conditions. Drone remains in STANDBY.")
            return None

    def create_drone_command(self, alert_packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the flight dispatch command for the drone controller.
        """
        return {
            "command": "INVESTIGATE",
            "latitude": alert_packet["latitude"],
            "longitude": alert_packet["longitude"],
            "node_id": alert_packet["node_id"],
            "target_reason": alert_packet.get("status", "ABNORMAL_SENSOR_READING"),
            "initial_telemetry": {
                "temperature": alert_packet.get("temperature"),
                "gas_level": alert_packet.get("gas_level"),
                "water_level": alert_packet.get("water_level"),
            }
        }
