import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter1d
import os
import configparser
# Construct the relative path to config.ini
config_path = os.path.realpath("../sleep-monitoring-ai-project/app/config/config.ini")
print(config_path)
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)

# Load YOLO Pose models
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu") # Set the device to GPU
model = YOLO(config.get('keypoint', 'yolo_model_pose_path')).to(device)

keypoint_history = []  # Store historical keypoint positions for convulsion detection
HISTORY_SIZE = config.getint('keypoint', 'frame_to_analyze_sleep_movement')  # Number of frames to analyze for convulsive detection

def estimate_pose(frame):
    # Perform inference with YOLO on the specified device (GPU or CPU)
    results = model.predict(frame, verbose=False, device=device)

    keypoints = []
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
           keypoints = result.keypoints.xy.cpu().numpy()  # Get keypoints as numpy array
    return keypoints if len(keypoints) > 0 else None  # Return None if no keypoints are detected
    

# Detect poor sleep movements based on keypoint motion patterns

def detect_poor_sleep_movement(keypoints):
    global keypoint_history

    if keypoints is None or len(keypoints) == 0:
        return False  # No keypoints detected, no convulsion

    # Ensure the keypoints array is not empty and has sufficient data
    if keypoints[0].shape[0] < 15:  # Ensure enough keypoints are available
        return False  # Not enough keypoints, skip processing

    # Define indices for relevant keypoints
    keypoint_indices = [7, 8, 13, 14]  # Left wrist, Right wrist, Left ankle, Right ankle

    # Extract relevant keypoints, using [0, 0] for missing keypoints
    relevant_keypoints = [
        keypoints[0][idx] if not np.array_equal(keypoints[0][idx], [0, 0]) else [0, 0]
        for idx in keypoint_indices
    ]

    # Append the current keypoints to the history
    keypoint_history.append(relevant_keypoints)
    if len(keypoint_history) > HISTORY_SIZE:
        keypoint_history.pop(0)  # Maintain a fixed history size

    # Apply smoothing to reduce jitter if we have enough data
    if len(keypoint_history) >= HISTORY_SIZE:
        smoothed_history = gaussian_filter1d(keypoint_history, sigma=1, axis=0)

        # Calculate velocities using smoothed history
        velocities = []
        for i in range(len(keypoint_indices)):
            motion = []
            for t in range(1, HISTORY_SIZE):
                # Compute velocity between consecutive frames
                v = np.linalg.norm(
                    np.array(smoothed_history[t][i]) - np.array(smoothed_history[t-1][i])
                )
                motion.append(v)
            velocities.append(motion)

        # Analyze motion patterns
        sustained_spikes = 0
        movement_count = 0
        still_duration = 0
        movement_clusters = 0
        for motion in velocities:
            # Movement Intensity & Frequency: High variance in velocity could indicate convulsion
            if np.std(motion) > config.getfloat('keypoint', 'max_standard_deviation_velocity') and np.max(motion) > config.getfloat('keypoint', 'max_velocity_of_one_keypoint'):  # Adjust thresholds as needed
                sustained_spikes += 1
            # Count how many movements are above a certain intensity (e.g., larger than 5 units of velocity)
            movement_count += sum(1 for v in motion if v > config.getfloat('keypoint', 'max_velocity'))
            # Detect clusters of rapid movement
            movement_clusters += sum(1 for t in range(1, len(motion)) if motion[t] > 5.0 and motion[t-1] > 5.0)

        # Track Duration of Stillness
        if sustained_spikes == 0 and movement_count == 0:
            still_duration += 1

        # Analyze the results based on thresholds
        if sustained_spikes >= config.getint('keypoint', 'max_sustained_spike'):  # At least two keypoints show convulsive patterns
            return True  # Convulsion detected

        # Check for restlessness: frequent movements within a short time (movement clusters)
        if movement_clusters > config.getint('keypoint', 'max_movement_cluster'):  # Adjust the threshold based on desired sensitivity
            return True  # Restlessness detected

        # Check if there is too much stillness, which may indicate deeper sleep
        if still_duration > 5:  # Number of frames without significant movement
            return False  # Assuming child is in deeper sleep (no restlessness)

        # If movement count is too high (indicating constant movement), return True (restlessness)
        if movement_count > config.getint('keypoint', 'max_movement_count'):  # Adjust based on real-world tests
            return True  # Restlessness detected

    return False  # No poor sleep movement detected



# Check if specific facial keypoints are covered (e.g., keypoints 0, 1, and 2)
def is_face_covered(keypoints):
    # Check if keypoints 0, 1, and 2 are missing or in unusual positions
    critical_keypoints = [0, 1, 2]  # Keypoints representing important facial areas
    for index in critical_keypoints:
        if len(keypoints[0]) <= index or keypoints[0][index][0] == 0 and keypoints[0][index][1] == 0:
            return True  # Face is considered covered if any critical keypoint is missing
    return False

# Draw keypoints and alert if face is covered
def draw_pose(frame, keypoints, offset_x=0, offset_y=0):
    if keypoints is not None:
        for i, keypoint in enumerate(keypoints[0]):
            if (keypoint[0], keypoint[1]) != (0, 0):  # Ensure keypoint is valid
                x, y = int(keypoint[0]) + offset_x, int(keypoint[1]) + offset_y
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw keypoint (red)
                cv2.putText(frame, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame