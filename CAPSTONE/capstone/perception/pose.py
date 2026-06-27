"""Human pose interpretation from keypoints.

For objects labelled as people we run pose estimation and derive a coarse
activity label (standing / sitting / profile / raised-arms) from the keypoint
geometry. These labels become part of the symbolic scene description used for
relational reasoning (paper Section 3 & Figure 2).
"""

import numpy as np

# YOLOv8 pose keypoint ordering:
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle


def analyze_pose(keypoints):
    """Classify a pose from a ``(17, 3)`` array of ``[x, y, conf]`` keypoints.

    Returns
    -------
    dict
        ``pose_type``, qualitative ``confidence``, the raw keypoints (as a
        string for serialization), and the average keypoint confidence.
    """
    valid_keypoints = keypoints[keypoints[:, 2] > 0]

    if len(valid_keypoints) < 5:
        return {
            "pose_type": "unknown",
            "confidence": "low",
            "keypoints_detected": len(valid_keypoints),
        }

    nose = keypoints[0] if keypoints[0, 2] > 0 else None
    left_shoulder = keypoints[5] if keypoints[5, 2] > 0 else None
    right_shoulder = keypoints[6] if keypoints[6, 2] > 0 else None
    left_hip = keypoints[11] if keypoints[11, 2] > 0 else None
    right_hip = keypoints[12] if keypoints[12, 2] > 0 else None
    left_knee = keypoints[13] if keypoints[13, 2] > 0 else None
    right_knee = keypoints[14] if keypoints[14, 2] > 0 else None
    left_ankle = keypoints[15] if keypoints[15, 2] > 0 else None
    right_ankle = keypoints[16] if keypoints[16, 2] > 0 else None

    pose_type = "unknown"
    pose_confidence = "medium"

    # --- Standing: torso above hips, hips above ankles -------------------
    if (left_hip is not None and right_hip is not None
            and left_shoulder is not None and right_shoulder is not None
            and left_ankle is not None and right_ankle is not None):

        torso_vertical = (left_shoulder[1] + right_shoulder[1]) / 2 < (left_hip[1] + right_hip[1]) / 2
        legs_vertical = (left_hip[1] + right_hip[1]) / 2 < (left_ankle[1] + right_ankle[1]) / 2

        if torso_vertical and legs_vertical:
            pose_type = "standing"
            pose_confidence = "high"

            # Raised arms: both wrists above shoulder line.
            if keypoints[9, 2] > 0 and keypoints[10, 2] > 0:
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                if keypoints[9, 1] < shoulder_y and keypoints[10, 1] < shoulder_y:
                    pose_type = "standing with raised arms"

    # --- Sitting: hips and knees at similar height ----------------------
    if (left_hip is not None and right_hip is not None
            and left_knee is not None and right_knee is not None):
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        if abs(hip_y - knee_y) < 50:
            pose_type = "sitting"
            pose_confidence = "medium"

    # --- Profile view: shoulders horizontally close ---------------------
    if left_shoulder is not None and right_shoulder is not None:
        if abs(left_shoulder[0] - right_shoulder[0]) < 30:
            pose_type = "profile view"
            pose_confidence = "medium"

    avg_confidence = np.mean(valid_keypoints[:, 2])

    return {
        "pose_type": pose_type,
        "confidence": pose_confidence,
        "keypoints_detected": str(keypoints),
        "avg_keypoint_confidence": float(avg_confidence),
    }
