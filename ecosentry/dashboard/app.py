"""
ECOSENTRY — Disaster Monitoring Dashboard Server
Lightweight Flask backend serving real-time sensor, UAV, radiometric temperature, and AI analytics.
"""

import json
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_alert():
    """Load latest_alert.json safely with fallbacks."""
    alert_file = RESULTS_DIR / "latest_alert.json"
    if alert_file.exists():
        try:
            with open(alert_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to parse alert: {e}"}

    # Default baseline if simulation hasn't run yet
    return {
        "status": "STANDBY",
        "assessment": "System initialized. Waiting for sensor trigger or simulation run.",
        "node_id": "NODE_01",
        "latitude": 12.9716,
        "longitude": 80.2209,
        "sensor_temperature": 28.0,
        "gas_level": 22.0,
        "water_level": 75.0,
        "hotspot_count": 0,
        "temperature_hotspot_count": 0,
        "temperature_max_c": None,
        "fire_detections": 0,
        "smoke_detections": 0,
        "verification_status": "MONITORING_ACTIVE",
        "evidence_path": "composite_evidence.png",
        "timestamp": None,
        "details": {
            "corroborations": 0,
            "vegetation_fire_risk": False,
            "strongest_relative_delta": 0.0,
            "temperature_hotspots": [],
            "yolo_predictions": [],
        },
    }


@app.route("/")
def index():
    """Render the dashboard UI."""
    return render_template("index.html")


@app.route("/api/alert")
def get_alert():
    """API endpoint providing latest structured alert."""
    return jsonify(load_latest_alert())


@app.route("/api/trigger-simulation", methods=["POST"])
def trigger_simulation():
    """Trigger the complete drone simulation from the dashboard UI."""
    try:
        from simulation.run_demo import run_pipeline
        run_pipeline()
        return jsonify({"success": True, "message": "Simulation completed successfully!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serve generated evidence images."""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("      ECOSENTRY DISASTER MONITORING COMMAND CENTER      ")
    print("      Server listening at: http://127.0.0.1:5000         ")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
