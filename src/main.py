import cv2
from person_detection import detect_person, draw_bed_area, save_bed_area, load_bed_area, select_bed_area
from pose_estimation import estimate_pose, classify_posture

# Draw bounding boxes around detected persons
def draw_bounding_boxes(frame, persons):
    for person in persons:
        x1, y1, x2, y2 = map(int, person)  # Bounding box coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

# Draw pose skeleton based on keypoints
def draw_pose(frame, keypoints):
    # Define pairs of keypoints to connect for pose skeleton
    pose_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Head and arms
        (1, 5), (5, 6), (6, 7),                # Left side
        (1, 8), (8, 9), (9, 10),               # Right side
        (11, 12), (12, 13), (13, 14),          # Left leg
        (11, 15), (15, 16), (16, 17)           # Right leg
    ]
    
    for part_a, part_b in pose_pairs:
        # Ensure that the indices are within the length of the keypoints list
        if part_a < len(keypoints) and part_b < len(keypoints):
            x1, y1 = int(keypoints[part_a][0]), int(keypoints[part_a][1])
            x2, y2 = int(keypoints[part_b][0]), int(keypoints[part_b][1])
            
            # Only draw if both keypoints are valid
            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):  # Check for non-zero coordinates
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow lines
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)  # Red keypoints
                cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)  # Red keypoints

# Initialize or load bed area
def initialize_bed_area(frame, bed_area):
    if bed_area is None:
        print("No bed area found. Please select the bed area.")
        bed_area = select_bed_area(frame)
        if bed_area is not None:
            save_bed_area(bed_area)
    return bed_area

# Main video processing loop
def process_video_feed():
    cap = cv2.VideoCapture(0)
    bed_area = load_bed_area()  # Load bed area from config

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Initialize bed area if not set
        bed_area = initialize_bed_area(frame, bed_area)
        if bed_area is None:
            print("Unable to proceed without bed area. Exiting...")
            break

        # Draw the bed area on the frame
        draw_bed_area(frame, bed_area)

        # Detect persons and draw bounding boxes
        persons = detect_person(frame, bed_area)
        draw_bounding_boxes(frame, persons)

        # Detect pose and classify posture for each person
        for person in persons:
            x1, y1, x2, y2 = map(int, person)
            person_frame = frame[y1:y2, x1:x2]  # Crop the frame to the person's bounding box
            keypoints = estimate_pose(person_frame)
            posture = classify_posture(keypoints)
            cv2.putText(frame, f"Posture: {posture}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            draw_pose(frame, keypoints)

        # Display the video feed with overlays
        cv2.imshow("Monitoring System", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_feed()
