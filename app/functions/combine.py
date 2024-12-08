import cv2
import time
from person_detection import detect_person, draw_bed_area, load_bed_area, create_bed_area_from_person_bbox, save_bed_area, draw_bounding_boxes, is_person_outside_bed, is_sitting
from keypoint import estimate_pose, detect_poor_sleep_movement, draw_pose, is_face_covered
from alert_system import send_alert
import configparser
import os

# Construct the relative path to config.ini
config_path = os.path.realpath("../sleep-monitoring-ai-project/app/config/config.ini")
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)


def process_video_feed(cap):
    bed_areas = load_bed_area()
    prev_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Tính FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"

        key = cv2.waitKey(10) & 0xFF
        if key == ord('b'):
            bed_areas = []
            persons = detect_person(frame, None)
            for person in persons:
                bed_area = create_bed_area_from_person_bbox(person)
                bed_areas.append(bed_area)
            save_bed_area(bed_areas)

        if bed_areas:
            for bed_area in bed_areas:
                draw_bed_area(frame, bed_area)

        persons = detect_person(frame, bed_areas)
        for person in persons:
            x1, y1, x2, y2 = map(int, person)
            person_frame = frame[y1:y2, x1:x2]
            draw_bounding_boxes(frame, persons)
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
                                send_alert(config.get("alert_system", "is_face_covered_alert"))  # Replace with your alert mechanism
                            if detect_poor_sleep_movement(keypoints):
                                send_alert(config.get("alert_system", "poor_sleep_movement_alert"))
                        draw_pose(frame, keypoints, x1, y1)


        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if key == ord('q'):
            break
        return ret, frame
    cap.release()
    cv2.destroyAllWindows()
