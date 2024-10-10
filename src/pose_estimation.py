import cv2
import numpy as np
from ultralytics import YOLO
from alert_system import send_alert

# Load YOLOv8 Pose model
model = YOLO("../data/yolo11m-pose.pt")  # Replace with the correct path to your YOLO model


def estimate_pose(frame):
    # Perform inference with YOLO
    results = model.predict(frame, verbose=False)

    keypoints = []
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # Get keypoints as numpy array
    # Use .size to check if the array has elements
    return keypoints if keypoints.size > 0 else None  # Return None if no keypoints detected


# Classify posture based on detected keypoints
# Classify posture based on detected keypoints
def classify_posture(keypoints):
    if keypoints is None:
        return "unknown"

    # Initialize variables to store keypoints
    head, left_shoulder, right_shoulder, left_hip, right_hip = None, None, None, None, None

    # Assuming keypoints[0] contains all keypoints for the body parts
    for i, keypoint in enumerate(keypoints[0]):
        if i == 1:  # Head (Nose)
            head = keypoint
        elif i == 2:  # Left Shoulder
            left_shoulder = keypoint
        elif i == 3:  # Right Shoulder
            right_shoulder = keypoint
        elif i == 4:  # Left Hip
            left_hip = keypoint
        elif i == 5:  # Right Hip
            right_hip = keypoint

    # Calculate average shoulder and hip positions
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2

    # Check for lying down (head, shoulders, hips roughly aligned horizontally)
    if np.abs(head[1] - avg_shoulder_y) < 0.1 and np.abs(avg_shoulder_y - avg_hip_y) < 0.1:
        send_alert("lying_down")
        return "lying_down"

    # Check for sitting (hips below shoulders, significant difference in vertical height)
    if avg_hip_y > avg_shoulder_y and head[1] > avg_shoulder_y:
        send_alert("sitting")
        return "sitting"

    # Check for standing (shoulders above hips, and head above shoulders)
    if head[1] < avg_shoulder_y and avg_shoulder_y > avg_hip_y:
        send_alert("standing")
        return "standing"

    # Check for abnormal sleeping positions (tilted or twisted body)
    head_distance_shoulder = np.abs(head[1] - avg_shoulder_y)
    shoulder_distance_hip = np.abs(avg_shoulder_y - avg_hip_y)

    # Define thresholds for abnormal positions
    if head_distance_shoulder > 0.2 or shoulder_distance_hip > 0.3:
        send_alert("abnormal_sleeping_position")
        return "abnormal_sleeping_position"
    send_alert("unknown")
    return "unknown"

