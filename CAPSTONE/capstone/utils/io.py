"""Small IO / serialization helpers shared across stages."""

import os

import numpy as np


def json_serializable(obj):
    """Default function for ``json.dumps`` to handle NumPy scalar/array types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_pil_image(pil_img, save_path):
    """Save a PIL image, creating parent directories as needed."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_img.save(save_path)
    return save_path
