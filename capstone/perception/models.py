"""Load the frozen, off-the-shelf perception stack.

CAPSTONE never fine-tunes any of these modules — they are used purely as
feature extractors (paper Section 4.1). Each can be hot-swapped for a newer
checkpoint without touching the rest of the pipeline.

Modules
-------
* detector  : YOLO object detector (default ``yolov8l.pt``; the paper also
              reports YOLOv11 variants — pass ``detector_weights="yolo11l.pt"``).
* depth     : monocular depth estimation (``Intel/dpt-large``).
* ocr       : EasyOCR text detection / recognition.
* pose      : YOLO pose estimator (``yolov8n-pose.pt``).
"""

import logging

logger = logging.getLogger(__name__)


def load_models(
    detector_weights="yolov8l.pt",
    pose_weights="yolov8n-pose.pt",
    depth_model="Intel/dpt-large",
    ocr_langs=("ch_sim", "en"),
):
    """Instantiate and return ``(detector, depth, ocr, pose)``.

    Heavy imports happen inside the function so that lightweight utilities in
    this package can be imported without pulling in torch / ultralytics.
    """
    from ultralytics import YOLO
    from transformers import pipeline
    import easyocr

    logger.info("Loading object detector: %s", detector_weights)
    detector = YOLO(detector_weights)

    logger.info("Loading depth estimator: %s", depth_model)
    depth_estimator = pipeline("depth-estimation", model=depth_model)

    logger.info("Loading OCR reader (langs=%s)", list(ocr_langs))
    ocr_reader = easyocr.Reader(list(ocr_langs))

    logger.info("Loading pose estimator: %s", pose_weights)
    pose_model = YOLO(pose_weights)

    return detector, depth_estimator, ocr_reader, pose_model
