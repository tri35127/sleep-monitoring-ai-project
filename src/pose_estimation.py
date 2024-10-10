import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Pose model
model = YOLO("../data/yolo11m-pose.pt")  # Replace with the correct path to your YOLO model


def estimate_pose(frame):
    # Perform inference with YOLO
    results = model.predict(frame, verbose=False)

    keypoints = []
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # Get keypoints as numpy array
            print(keypoints)
    # Use .size to check if the array has elements
    return keypoints if keypoints.size > 0 else None  # Return None if no keypoints detected


# Classify posture based on detected keypoints
def classify_posture(keypoints):
    if keypoints is None or len(keypoints) < 13:
        return "unknown"

    head = keypoints[0]  # Nose keypoint
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    # Check for lying down (head, shoulders, hips aligned horizontally)
    if head[1] > left_shoulder[1] and head[1] > left_hip[1] and np.abs(left_hip[1] - right_hip[1]) < 0.1:
        return "lying_down"

    # Check for unusual sleeping positions (head tilted excessively or body twisted)
    head_distance_shoulder = np.abs(head[1] - ((left_shoulder[1] + right_shoulder[1]) / 2))
    shoulder_distance_hip = np.abs((left_shoulder[1] + right_shoulder[1]) / 2 - (left_hip[1] + right_hip[1]) / 2)

    # Define thresholds for unusual positions
    if head_distance_shoulder > 0.2 or shoulder_distance_hip > 0.3:
        send_alert("abnormal_sleeping_position")
        return "abnormal_sleeping_position"

    # Check for sitting (hips lower than shoulders and head)
    elif left_hip[1] > left_shoulder[1]:
        return "sitting"

    # Check for standing
    elif np.abs(left_hip[1] - right_hip[1]) < 0.1 and head[1] < left_shoulder[1]:
        return "standing"

    return "unknown"


# Function to send an alert for detected posture
def send_alert(alert_type):
    print(f"Alert: Detected {alert_type} posture")
