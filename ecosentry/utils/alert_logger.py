"""
Alert Logger Utility

Saves structured alert payloads to JSON files for historical logging and dashboard polling.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def save_alert(
    alert: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save the given alert dictionary to a JSON file.
    Default destination is results/latest_alert.json.
    """
    if output_path is None:
        output_path = Path("results/latest_alert.json")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=4)

    return output_path
