"""Symbolic translation T : F_v -> D.

Converts the raw perception outputs into the structured natural-language scene
description that a frozen LLM consumes. This is the heart of CAPSTONE's
"pixel-free" design (paper Section 3, "Symbolic Representation").
"""


def prepare_for_json(objects, depth_info=None, ocr_results=None, pose_results=None):
    """Convert perception outputs into a JSON-serializable list of dicts.

    The serialized blob is what gets stored in the intermediate CSV so the
    perception and reasoning stages can be decoupled / cached.
    """
    result = []
    for obj in objects:
        result.append({
            "name": obj["name"],
            "confidence": float(obj["confidence"]),
            "confidence_label": obj["confidence_label"],
            "color": str(obj["color"]),
            "bbox": {
                "x1": int(obj["bbox"]["x1"]),
                "y1": int(obj["bbox"]["y1"]),
                "x2": int(obj["bbox"]["x2"]),
                "y2": int(obj["bbox"]["y2"]),
            },
            "center": {"x": int(obj["center"]["x"]), "y": int(obj["center"]["y"])},
            "quadrant": obj["quadrant"],
            "depth": obj.get("depth"),
            "pose": obj.get("pose"),
        })

    if depth_info:
        result.append({
            "depth_analysis": {
                "min_depth": float(depth_info["min_depth"]),
                "max_depth": float(depth_info["max_depth"]),
                "mean_depth": float(depth_info["mean_depth"]),
                "overall_color": depth_info["overall_color"],
            }
        })

    if ocr_results:
        result.append({"ocr_results": ocr_results})

    if pose_results:
        result.append({"pose_results": pose_results})

    return result


def display_detections(detections, depth_info=None, ocr_results=None, pose_results=None):
    """Render perception outputs into a single formatted description string.

    The output enumerates objects with their attributes (confidence, color,
    bounding box, quadrant, depth, pose), then appends depth, OCR and pose
    sections plus a summary — i.e. ``D = (d_objects, d_relations, d_scene)``.
    """
    output = ["=" * 80, "OBJECT DETECTION RESULTS", "=" * 80]

    for i, detection in enumerate(detections, 1):
        output.append(f"\nDETECTION #{i}: {detection['name'].upper()}")
        output.append("-" * 50)
        output.append(
            f"Confidence: {detection['confidence'] * 100:.2f}% "
            f"({detection['confidence_label']})"
        )
        output.append(f"Dominant Colors: {detection['color']}")
        bbox = detection["bbox"]
        output.append(
            f"Bounding Box: x1={bbox['x1']}, y1={bbox['y1']}, "
            f"x2={bbox['x2']}, y2={bbox['y2']}"
        )
        output.append(
            f"Center Point: ({detection['center']['x']}, {detection['center']['y']})"
        )
        output.append(f"Quadrant: {detection['quadrant']}")

        if detection.get("depth"):
            d = detection["depth"]
            output.append(
                f"Depth: min={d['min']:.2f}, max={d['max']:.2f}, mean={d['mean']:.2f}"
            )

        if detection.get("pose"):
            pose = detection["pose"]["analysis"]
            output.append(f"Pose Type: {pose['pose_type']}")
            output.append(f"Pose Confidence: {pose['confidence']}")
            output.append(f"Keypoints: {pose['keypoints_detected']}")
            if "avg_keypoint_confidence" in pose:
                output.append(
                    f"Average Keypoint Confidence: {pose['avg_keypoint_confidence']:.2f}"
                )

    if depth_info:
        output += ["\n" + "=" * 80, "DEPTH ESTIMATION RESULTS", "=" * 80]
        output.append(f"Min Depth: {depth_info['min_depth']:.2f}")
        output.append(f"Max Depth: {depth_info['max_depth']:.2f}")
        output.append(f"Mean Depth: {depth_info['mean_depth']:.2f}")
        output.append(
            f"Overall color dominant of the image : {depth_info['overall_color']}"
        )

    if ocr_results:
        output += ["\n" + "=" * 80, "OCR TEXT DETECTION RESULTS", "=" * 80]
        for i, text_obj in enumerate(ocr_results, 1):
            output.append(f"\nTEXT #{i}: '{text_obj['text']}'")
            output.append("-" * 50)
            output.append(f"Confidence: {text_obj['confidence'] * 100:.2f}%")
            bbox = text_obj["bbox"]
            output.append(
                f"Bounding Box: x1={bbox['x1']}, y1={bbox['y1']}, "
                f"x2={bbox['x2']}, y2={bbox['y2']}"
            )
            output.append(
                f"Center Point: ({text_obj['center']['x']}, {text_obj['center']['y']})"
            )
            output.append(f"Quadrant: {text_obj['quadrant']}")

    if pose_results:
        output += ["\n" + "=" * 80, "POSE ESTIMATION RESULTS", "=" * 80]
        for i, pose in enumerate(pose_results, 1):
            output.append(f"\nPOSE #{i}: Person ID {pose['person_id']}")
            output.append("-" * 50)
            output.append(f"Pose Type: {pose['pose_type']}")
            output.append(f"Pose Confidence: {pose['confidence']}")

    output.append("\n" + "=" * 80)

    # Summary statistics
    object_counts = {}
    for detection in detections:
        object_counts[detection["name"]] = object_counts.get(detection["name"], 0) + 1

    output += ["\nSUMMARY:", "-" * 50, f"Total object detections: {len(detections)}"]
    for obj_name, count in object_counts.items():
        output.append(f"- {obj_name}: {count}")
    if ocr_results:
        output.append(f"Total text detections: {len(ocr_results)}")
    if pose_results:
        output.append(f"Total pose estimations: {len(pose_results)}")
    output.append("=" * 80)

    return "\n".join(output)
