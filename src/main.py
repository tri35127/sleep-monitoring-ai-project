import cv2
import time
from person_detection import detect_person, draw_bed_area, load_bed_area, draw_bounding_boxes, initialize_bed_area
from pose_estimation import estimate_pose, classify_posture, draw_pose


# Main video processing loop
def process_video_feed():
    cap = cv2.VideoCapture(0)

    # Set resolution (for example, 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width to 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height to 480

    bed_area = load_bed_area()  # Load bed area from config
    prev_frame_time = 0  # Variable to store time of the previous frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Calculate FPS
        new_frame_time = time.time()  # Time at which current frame is processed
        fps = 1 / (new_frame_time - prev_frame_time)  # Calculate FPS
        prev_frame_time = new_frame_time  # Update previous frame time

        # Convert FPS to an integer for display
        fps = int(fps)
        fps_text = f"FPS: {fps}"

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

            if keypoints is not None:
                posture = classify_posture(keypoints)
                cv2.putText(frame, f"Posture: {posture}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                draw_pose(frame, keypoints, x1, y1)  # Offset by the bounding box coordinates

        # Display the FPS on the frame
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the video feed with overlays
        cv2.imshow("Monitoring System", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_feed()
