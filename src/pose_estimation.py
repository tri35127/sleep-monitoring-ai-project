import cv2
import numpy as np
from alert_system import send_alert
import torch
from ultralytics import YOLO

# Check if MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")  # Fallback to CPU if MPS is not available
    print("MPS not available. Using CPU.")
# Load YOLOv8 Pose model
model = YOLO("../data/yolo11m-pose.pt")  # Replace with the correct path to your YOLO model


def estimate_pose(frame):
    # Perform inference with YOLO
    results = model.predict(frame, verbose=False, device=device)

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

    # Check for sitting (hips below shoulders, significant difference in vertical height)
    if avg_hip_y > avg_shoulder_y and head[1] > avg_shoulder_y:
        send_alert("sitting")
        return "sitting"

    # Check for standing (shoulders above hips, and head above shoulders)
    if head[1] < avg_shoulder_y and avg_shoulder_y > avg_hip_y:
        send_alert("standing")
        return "standing"
    
    send_alert("unknown")
    return "unknown"

def draw_pose(frame, keypoints, offset_x, offset_y):
    for i, keypoint in enumerate(keypoints[0]):
        if (keypoint[0], keypoint[1]) != (0, 0):  # Check if keypoint is valid
            x, y = int(keypoint[0]) + offset_x, int(keypoint[1]) + offset_y
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw keypoint (red)

            # Display keypoint index
            cv2.putText(frame, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
