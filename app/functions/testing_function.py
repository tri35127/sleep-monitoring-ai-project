import cv2
import time
import psutil
import subprocess
import matplotlib.pyplot as plt
from person_detection import (
    detect_person, draw_bed_area, load_bed_area, create_bed_area_from_person_bbox, 
    save_bed_area, draw_bounding_boxes, is_person_outside_bed, is_sitting
)
from keypoint import estimate_pose, draw_pose, detect_poor_sleep_movement, is_face_covered
from alert_system import send_alert, display_alert_statistics
import numpy as np
import configparser
import os
import torch

if torch.backends.mps.is_available():
    config_path = os.path.realpath("../config/config.ini")
else:
    config_path = os.path.realpath("../sleep-monitoring-ai-project/app/config/config.ini")
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)


# Performance metrics
performance_metrics = {
    "fps": [],
    "response_times": [],
    "cpu_usages": [],
    "memory_usages": [],
    "gpu_usages": [],
    "gpu_memory_usages": []
}

def get_gpu_usage():
    """Retrieve GPU usage metrics using `nvidia-smi`."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        gpu_util, gpu_mem = map(int, result.strip().split(", "))
        return gpu_util, gpu_mem
    except Exception as e:
        print("Unable to retrieve GPU information:", e)
        return None, None

def update_performance_metrics(start_time):
    """Update and store system performance metrics."""
    # FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - start_time)
    performance_metrics["fps"].append(fps)
    
    # System resource usage
    performance_metrics["cpu_usages"].append(psutil.cpu_percent())
    performance_metrics["memory_usages"].append(psutil.virtual_memory().percent)
    if torch.cuda.is_available():
        gpu_util, gpu_mem = get_gpu_usage()
    else:
        gpu_util, gpu_mem = [0,0]
    if gpu_util is not None:
        performance_metrics["gpu_usages"].append(gpu_util)
        performance_metrics["gpu_memory_usages"].append(gpu_mem)

    return new_frame_time, fps

def draw_metrics(frame, fps):
    """Display FPS and other information on the frame."""
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

def process_person(frame, person, bed_areas):
    """Process each detected person, including pose estimation and alert checks."""
    x1, y1, x2, y2 = map(int, person)
    person_frame = frame[y1:y2, x1:x2]
    draw_bounding_boxes(frame, [person])

    if bed_areas:
        for bed_area in bed_areas:
            if is_sitting(person, bed_area):
                send_alert(config.get("alert_system", "is_sitting_alert"))
            elif is_person_outside_bed(person, bed_area):
                send_alert(config.get("alert_system", "is_person_outside_bed_alert"))
            else:
                keypoints = estimate_pose(person_frame)
                if keypoints is not None:
                    if is_face_covered(keypoints):
                        send_alert(
                            config.get("alert_system", "is_face_covered_alert"))  # Replace with your alert mechanism
                    if detect_poor_sleep_movement(keypoints):
                        send_alert(config.get("alert_system", "poor_sleep_movement_alert"))
                draw_pose(frame, keypoints, x1, y1)

def process_video_feed():
    cap = cv2.VideoCapture(config.getint("camera", "camera_id"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.getint("camera", "width"))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.getint("camera", "height"))

    bed_areas = load_bed_area() or []
    prev_frame_time = 0

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Handle bed area marking
        key = cv2.waitKey(10) & 0xFF
        if key == ord('b'):
            bed_areas.clear()
            persons = detect_person(frame, None)
            for person in persons:
                bed_areas.append(create_bed_area_from_person_bbox(person))
            save_bed_area(bed_areas)

        # Draw bed areas
        for bed_area in bed_areas:
            draw_bed_area(frame, bed_area)

        # Detect and process persons
        persons = detect_person(frame, bed_areas)
        for person in persons:
            process_person(frame, person, bed_areas)

        # Update and display performance metrics
        prev_frame_time, fps = update_performance_metrics(prev_frame_time)
        draw_metrics(frame, fps)

        # Calculate response time
        response_time = time.time() - start_time
        performance_metrics["response_times"].append(response_time)

        cv2.imshow("Monitoring System", frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plot_performance_metrics(performance_metrics)

def plot_performance_metrics(metrics):
    """Plot performance metrics over time and display their statistics."""
    plt.figure(figsize=(12, 10))
    titles = [
        ("FPS", "fps"), 
        ("Response Time (s)", "response_times"),
        ("CPU Usage (%)", "cpu_usages"), 
        ("Memory Usage (%)", "memory_usages"),
        ("GPU Usage (%)", "gpu_usages"), 
        ("GPU Memory Usage (MiB)", "gpu_memory_usages")
    ]

    for i, (title, key) in enumerate(titles, 1):
        plt.subplot(3, 2, i)
        plt.plot(metrics[key], label=title)
        plt.title(title)
        plt.legend()

        # Calculate and display statistics
        if metrics[key]:
            data = np.array(metrics[key])
            max_val = np.max(data)
            min_val = np.min(data)
            mean_val = np.mean(data)
            print(f"{title} Statistics:")
            print(f"  Max: {max_val}")
            print(f"  Min: {min_val}")
            print(f"  Mean/Average: {mean_val}")
            print("-" * 30)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_video_feed()
    display_alert_statistics()