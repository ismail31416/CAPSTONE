"""Symbolic translation stage: T : F_v -> D (paper Section 3)."""

from .extractors import (
    count_objects,
    extract_depth_info,
    extract_ocr_text,
    extract_pose_info,
    safe_display,
)
from .formatter import display_detections, prepare_for_json

__all__ = [
    "display_detections",
    "prepare_for_json",
    "safe_display",
    "extract_ocr_text",
    "extract_depth_info",
    "extract_pose_info",
    "count_objects",
]
