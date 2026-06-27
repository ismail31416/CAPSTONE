"""Optional visualization of the perception outputs.

Renders object boxes, pose skeletons, OCR boxes and the depth map side by
side. Useful for qualitative inspection / figures, not required for the
reasoning pipeline.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Pose skeleton connectivity (pairs of keypoint indices).
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),                 # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 11), (6, 12), (11, 12),                      # torso
    (11, 13), (13, 15), (12, 14), (14, 16),          # legs
]


def visualize(image_path, objects, depth_info, ocr_results, pose_results, save_path=None):
    """Plot detections + depth map; save to ``save_path`` or show inline."""
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(img)
    ax1.set_title("Object Detection, OCR & Pose Estimation")

    for obj in objects:
        bbox = obj["bbox"]
        ax1.add_patch(plt.Rectangle(
            (bbox["x1"], bbox["y1"]),
            bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"],
            edgecolor="blue", linewidth=2, fill=False,
        ))
        ax1.text(bbox["x1"], bbox["y1"] - 5,
                 f"{obj['name']} ({obj['confidence']:.2f})",
                 color="white", bbox=dict(facecolor="black", alpha=0.5))

        if "pose" in obj:
            keypoints = np.array(obj["pose"]["keypoints"])
            for x, y, conf in keypoints:
                if conf > 0.5:
                    ax1.scatter(x, y, color="red", s=20)
            for p1, p2 in _SKELETON:
                if keypoints[p1, 2] > 0.5 and keypoints[p2, 2] > 0.5:
                    ax1.plot(
                        [keypoints[p1, 0], keypoints[p2, 0]],
                        [keypoints[p1, 1], keypoints[p2, 1]],
                        color="lime", linewidth=2,
                    )
            pose_type = obj["pose"].get("analysis", {}).get("pose_type")
            if pose_type:
                ax1.text(bbox["x1"], bbox["y2"] + 15, f"Pose: {pose_type}",
                         color="white", bbox=dict(facecolor="blue", alpha=0.7))

    for text_obj in ocr_results:
        bbox = text_obj["bbox"]
        ax1.add_patch(plt.Rectangle(
            (bbox["x1"], bbox["y1"]),
            bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"],
            edgecolor="yellow", linewidth=2, fill=False,
        ))
        ax1.text(bbox["x1"], bbox["y1"] - 5, f"Text: {text_obj['text'][:10]}...",
                 color="yellow", bbox=dict(facecolor="black", alpha=0.5))

    ax1.axis("off")

    im = ax2.imshow(depth_info["depth_map"], cmap="plasma")
    ax2.set_title("Depth Map")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2, label="Depth")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
