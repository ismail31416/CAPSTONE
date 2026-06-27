"""The unified perception pass: detection + color + pose + depth + OCR.

``detect_objects`` runs every frozen module on a single image and returns the
raw structured outputs. These outputs are later turned into natural-language
descriptions by :mod:`capstone.translation` — implementing the transformation
``T : F_v -> D`` from the paper.
"""

import cv2
import numpy as np

from .color import get_dominant_colors
from .geometry import confidence_label, get_quadrant
from .pose import analyze_pose

HUMAN_CATEGORIES = {"person", "man", "woman", "boy", "girl", "child", "people", "human"}
POSE_MATCH_THRESHOLD = 50  # px distance to associate a detection with a pose


def detect_objects(image, yolo_model, depth_estimator, ocr_reader, pose_model):
    """Run the full perception stack on a single PIL image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    yolo_model, depth_estimator, ocr_reader, pose_model
        Frozen modules returned by :func:`capstone.perception.models.load_models`.

    Returns
    -------
    tuple
        ``(objects, depth_info, ocr_results, pose_results)``.
    """
    img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # --- Object detection -------------------------------------------------
    results = yolo_model(img_rgb)
    objects = []
    person_detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = result.names[cls]

            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            color_info = get_dominant_colors(crop)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            obj_info = {
                "name": name,
                "confidence": conf,
                "confidence_label": confidence_label(conf),
                "color": color_info,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": {"x": center_x, "y": center_y},
                "quadrant": get_quadrant(center_x, center_y, w, h),
            }

            if name.lower() in HUMAN_CATEGORIES:
                person_detections.append(obj_info)
            objects.append(obj_info)

    # --- Pose estimation for people --------------------------------------
    pose_results = []
    if person_detections:
        pose_results_raw = pose_model(img_rgb)
        if pose_results_raw:
            for person in person_detections:
                person_center = person["center"]
                closest_pose = None
                closest_distance = float("inf")

                for pose_result in pose_results_raw:
                    keypoints = pose_result.keypoints.data
                    for i, pose_box in enumerate(pose_result.boxes):
                        px1, py1, px2, py2 = map(int, pose_box.xyxy[0])
                        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                        distance = (
                            (person_center["x"] - pcx) ** 2
                            + (person_center["y"] - pcy) ** 2
                        ) ** 0.5
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_pose = {
                                "keypoints": keypoints[i].cpu().numpy(),
                                "bbox": {"x1": px1, "y1": py1, "x2": px2, "y2": py2},
                            }

                if closest_pose and closest_distance < POSE_MATCH_THRESHOLD:
                    pose_analysis = analyze_pose(closest_pose["keypoints"])
                    person["pose"] = {
                        "analysis": pose_analysis,
                        "keypoints": closest_pose["keypoints"].tolist(),
                    }
                    pose_results.append({
                        "person_id": objects.index(person),
                        "bbox": person["bbox"],
                        "pose_type": pose_analysis["pose_type"],
                        "confidence": pose_analysis["confidence"],
                        "keypoints": str(closest_pose["keypoints"].tolist()),
                    })

    # --- Depth estimation -------------------------------------------------
    pil_img = image.convert("RGB")
    depth_result = depth_estimator(pil_img)
    depth_map = np.array(depth_result["depth"])
    overall_color = get_dominant_colors(img_rgb)

    depth_info = {
        "min_depth": float(np.min(depth_map)),
        "max_depth": float(np.max(depth_map)),
        "mean_depth": float(np.mean(depth_map)),
        "depth_map": depth_map,  # kept for optional visualization
        "overall_color": str(overall_color),
    }

    for obj in objects:
        bbox = obj["bbox"]
        region = depth_map[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
        if region.size > 0:
            obj["depth"] = {
                "min": float(np.min(region)),
                "max": float(np.max(region)),
                "mean": float(np.mean(region)),
            }

    # --- OCR --------------------------------------------------------------
    ocr_results = ocr_reader.readtext(img_rgb)
    processed_ocr = []
    for box_points, text, conf in ocr_results:
        x_coords = [p[0] for p in box_points]
        y_coords = [p[1] for p in box_points]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        processed_ocr.append({
            "text": text,
            "confidence": float(conf),
            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "center": {"x": int(center_x), "y": int(center_y)},
            "quadrant": get_quadrant(center_x, center_y, w, h),
        })

    return objects, depth_info, processed_ocr, pose_results
