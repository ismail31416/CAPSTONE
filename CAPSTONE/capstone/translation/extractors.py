"""Helpers that pull individual modalities back out of the serialized JSON.

After perception writes a JSON blob per image into the CSV, these functions
re-hydrate specific fields (formatted text, OCR, depth, pose, object count) as
extra columns for inspection and downstream reasoning.
"""

import json

from .formatter import display_detections


def _load(json_str):
    if isinstance(json_str, str):
        return json.loads(json_str)
    return None


def safe_display(json_str):
    """Rebuild the formatted description string from a serialized blob."""
    try:
        data = _load(json_str)
        if data is None:
            return ""
        objects = [item for item in data if "name" in item]
        depth_info = next((i["depth_analysis"] for i in data if "depth_analysis" in i), None)
        ocr_results = next((i["ocr_results"] for i in data if "ocr_results" in i), None)
        pose_results = next((i["pose_results"] for i in data if "pose_results" in i), None)
        return display_detections(objects, depth_info, ocr_results, pose_results)
    except Exception as exc:  # noqa: BLE001
        print(f"Error formatting results: {exc}")
        return ""


def extract_ocr_text(json_str):
    """Return a ``"; "``-joined string of all detected text."""
    try:
        data = _load(json_str)
        if data is None:
            return "No text detected"
        ocr_item = next((i for i in data if "ocr_results" in i), None)
        if ocr_item:
            texts = [t["text"] for t in ocr_item["ocr_results"]]
            return "; ".join(texts) if texts else "No text detected"
        return "No text detected"
    except Exception:  # noqa: BLE001
        return "Error parsing OCR results"


def extract_depth_info(json_str):
    """Return a short ``Min/Max/Mean`` depth summary string."""
    try:
        data = _load(json_str)
        if data is None:
            return "No depth info"
        item = next((i for i in data if "depth_analysis" in i), None)
        if item:
            d = item["depth_analysis"]
            return f"Min: {d['min_depth']:.2f}, Max: {d['max_depth']:.2f}, Mean: {d['mean_depth']:.2f}"
        return "No depth info"
    except Exception:  # noqa: BLE001
        return "Error parsing depth results"


def extract_pose_info(json_str):
    """Return a ``"; "``-joined string describing detected poses."""
    try:
        data = _load(json_str)
        if data is None:
            return "No pose detected"
        pose_types = []
        for item in data:
            if "pose" in item and item["pose"] and "analysis" in item["pose"]:
                pose_types.append(f"{item['name']}: {item['pose']['analysis']['pose_type']}")
        pose_item = next((i for i in data if "pose_results" in i), None)
        if pose_item:
            for pose in pose_item["pose_results"]:
                if "pose_type" in pose:
                    pose_types.append(f"Person {pose['person_id']}: {pose['pose_type']}")
        return "; ".join(pose_types) if pose_types else "No pose detected"
    except Exception:  # noqa: BLE001
        return "Error parsing pose results"


def count_objects(json_str):
    """Return the number of detected objects in a serialized blob."""
    try:
        data = _load(json_str)
        if data is None:
            return 0
        return len([i for i in data if "name" in i])
    except Exception:  # noqa: BLE001
        return 0
