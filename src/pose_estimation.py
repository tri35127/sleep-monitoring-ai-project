import cv2
import numpy as np
from alert_system import send_alert
import torch
from ultralytics import YOLO

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA if available
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"cuDNN is enabled: {torch.backends.cudnn.enabled}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")  # Fallback to CPU if CUDA is not available
    print("CUDA not available. Using CPU.")

# Load YOLOv Pose model
model = YOLO("../data/yolo11m-pose.pt")  # Replace with the correct path to your YOLO model

def estimate_pose(frame):
    # Perform inference with YOLO on the specified device (GPU or CPU)
    results = model.predict(frame, verbose=False, device=device)

    keypoints = []
    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # Get keypoints as numpy array
    # Use .size to check if the array has elements
    return keypoints if keypoints.size > 0 else None  # Return None if no keypoints detected

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
        elif i == 5:  # Left Shoulder
            left_shoulder = keypoint
        elif i == 6:  # Right Shoulder
            right_shoulder = keypoint
        elif i == 11:  # Left Hip
            left_hip = keypoint
        elif i == 12:  # Right Hip
            right_hip = keypoint

    # Calculate average shoulder and hip positions
    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2

def draw_pose(frame, keypoints, offset_x, offset_y):
    for i, keypoint in enumerate(keypoints[0]):
        if (keypoint[0], keypoint[1]) != (0, 0):  # Check if keypoint is valid
            x, y = int(keypoint[0]) + offset_x, int(keypoint[1]) + offset_y
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw keypoint (red)

            # Display keypoint index
            cv2.putText(frame, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
