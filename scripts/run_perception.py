#!/usr/bin/env python
"""Stage 1 — Perception + Symbolic Translation.

Runs the frozen vision stack over a dataset and writes an *enhanced* CSV in
which every image-question row carries a structured textual scene description
(plus broken-out OCR / depth / pose columns). That CSV is the input to
``run_reasoning.py``.

Example
-------
    python scripts/run_perception.py \
        --dataset cambridgeltl/vsr_zeroshot --split test \
        --output vsr_detection_results_enhanced.csv \
        --detector yolov8l.pt

The default dataset (VSR zero-shot) exposes the columns
``image_link, image, caption, label, relation``. Use ``--from-images`` to run
on a local folder of images instead (questions then come from ``--question``).
"""

import argparse
import csv
import json
import logging
import os
import sys
from urllib.request import urlopen

import pandas as pd
from PIL import Image

# Allow running directly from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capstone.perception import detect_objects, load_models
from capstone.translation import (
    count_objects,
    extract_depth_info,
    extract_ocr_text,
    extract_pose_info,
    prepare_for_json,
    safe_display,
)
from capstone.utils import json_serializable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("perception.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _open_image(item):
    """Open an image from a VSR-style record (URL link or PIL image)."""
    if isinstance(item.get("image_link"), str):
        return Image.open(urlopen(item["image_link"]))
    return item["image"]


def run_dataset(args):
    from datasets import load_dataset

    logger.info("Loading dataset %s [%s]", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    detector, depth_estimator, ocr_reader, pose_model = load_models(
        detector_weights=args.detector, pose_weights=args.pose
    )

    # Group questions by unique image so each image is processed once.
    image_to_questions = {}
    for item in dataset:
        key = item["image"]
        if key not in image_to_questions:
            image_to_questions[key] = {
                "image": _open_image(item),
                "questions": [], "answer": [], "relation": [],
            }
        image_to_questions[key]["questions"].append(item.get("caption", args.question))
        image_to_questions[key]["answer"].append(item.get("label"))
        image_to_questions[key]["relation"].append(item.get("relation"))

    raw_csv = args.output.replace(".csv", "_raw.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["image_source", "question", "answer", "relation", "detection_results"]
        )

    for image_source, data in image_to_questions.items():
        try:
            logger.info("Processing %s (%d questions)", image_source, len(data["questions"]))
            objects, depth_info, ocr_results, pose_results = detect_objects(
                data["image"], detector, depth_estimator, ocr_reader, pose_model
            )
            json_ready = prepare_for_json(objects, depth_info, ocr_results, pose_results)
            json_str = json.dumps(json_ready, default=json_serializable)

            with open(raw_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for q, a, r in zip(data["questions"], data["answer"], data["relation"]):
                    writer.writerow([image_source, q, a, r, json_str])
        except Exception:
            logger.exception("Error processing image %s", image_source)

    _enhance_and_save(raw_csv, args.output)


def run_images(args):
    detector, depth_estimator, ocr_reader, pose_model = load_models(
        detector_weights=args.detector, pose_weights=args.pose
    )
    raw_csv = args.output.replace(".csv", "_raw.csv")
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["image_source", "question", "answer", "relation", "detection_results"]
        )

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in sorted(os.listdir(args.from_images)) if f.lower().endswith(exts)]
    if args.max_samples:
        files = files[: args.max_samples]

    for fname in files:
        path = os.path.join(args.from_images, fname)
        try:
            logger.info("Processing %s", fname)
            image = Image.open(path).convert("RGB")
            objects, depth_info, ocr_results, pose_results = detect_objects(
                image, detector, depth_estimator, ocr_reader, pose_model
            )
            json_ready = prepare_for_json(objects, depth_info, ocr_results, pose_results)
            json_str = json.dumps(json_ready, default=json_serializable)
            with open(raw_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([fname, args.question, "", "", json_str])
        except Exception:
            logger.exception("Error processing image %s", fname)

    _enhance_and_save(raw_csv, args.output)


def _enhance_and_save(raw_csv, output):
    """Add the formatted-text + per-modality columns and save the final CSV."""
    df = pd.read_csv(raw_csv)
    df["formatted_results"] = df["detection_results"].apply(safe_display)
    df["ocr_text"] = df["detection_results"].apply(extract_ocr_text)
    df["depth_info"] = df["detection_results"].apply(extract_depth_info)
    df["pose_info"] = df["detection_results"].apply(extract_pose_info)
    df["object_count"] = df["detection_results"].apply(count_objects)
    df.to_csv(output, index=False)
    logger.info("Saved enhanced results to %s (%d rows)", output, len(df))


def main():
    parser = argparse.ArgumentParser(description="CAPSTONE perception stage")
    parser.add_argument("--dataset", default="cambridgeltl/vsr_zeroshot",
                        help="HuggingFace dataset id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--from-images", default=None,
                        help="Run on a local image folder instead of a HF dataset")
    parser.add_argument("--question", default="Is there a person in the image?",
                        help="Question to use for plain image folders")
    parser.add_argument("--output", default="detection_results_enhanced.csv")
    parser.add_argument("--detector", default="yolov8l.pt",
                        help="YOLO detector weights (e.g. yolov8l.pt or yolo11l.pt)")
    parser.add_argument("--pose", default="yolov8n-pose.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.from_images:
        run_images(args)
    else:
        run_dataset(args)


if __name__ == "__main__":
    main()
