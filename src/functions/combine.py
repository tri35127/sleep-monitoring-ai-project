import cv2
import time
from person_detection import detect_person, draw_bed_area, load_bed_area, create_bed_area_from_person_bbox, save_bed_area, draw_bounding_boxes, person_alert
from keypoint import estimate_pose, classify_posture, draw_pose
from alert_system import send_alert


# Main video processing loop
def process_video_feed():
    cap = cv2.VideoCapture(0)  # Sử dụng camera
    bed_areas = load_bed_area()  # Load danh sách vùng giường từ file config
    prev_frame_time = 0  # Biến lưu thời gian của frame trước

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Tính FPS
        new_frame_time = time.time()  # Lấy thời gian hiện tại
        fps = 1 / (new_frame_time - prev_frame_time)  # Tính FPS
        prev_frame_time = new_frame_time  # Cập nhật thời gian frame trước
        fps_text = f"FPS: {int(fps)}"  # Chuyển FPS thành số nguyên để hiển thị

        # Nhấn "b" để chọn vùng giường cho mỗi người
        key = cv2.waitKey(10) & 0xFF
        if key == ord('b'):
            bed_areas = []  # Tạo danh sách vùng giường mới
            persons = detect_person(frame, None)  # Phát hiện người mà không kiểm tra vùng giường
            for person in persons:
                bed_area = create_bed_area_from_person_bbox(person)  # Tạo vùng giường từ bounding box của người
                bed_areas.append(bed_area)
            save_bed_area(bed_areas)  # Lưu danh sách vùng giường vào file config

    # Vẽ vùng giường nếu đã
        if bed_areas:
            for bed_area in bed_areas:
                draw_bed_area(frame, bed_area)

        # Phát hiện người và kiểm tra bất thường
        persons = detect_person(frame)  # Phát hiện người
        draw_bounding_boxes(frame, persons)
        for i, person in enumerate(persons):
            p_alert = person_alert(persons, bed_areas)
            # Kiểm tra nếu người ngoài giường
            if p_alert == "Cảnh báo: Trẻ đã rời khỏi giường!":
                send_alert("Cảnh báo: Trẻ đã rời khỏi giường!")
            else:
                if p_alert == "Cảnh báo: Trẻ đang ngồi!":
                    send_alert("Cảnh báo: Trẻ đang ngồi!")
                # Phát hiện khuôn mặt trong bounding box của người
                x1, y1, x2, y2 = map(int, person)
                person_frame = frame[y1:y2, x1:x2]  # Cắt khung hình theo bounding box của người
                # Nếu có khuôn mặt, tiến hành phát hiện khung xương
                keypoints = estimate_pose(person_frame)
                if keypoints is not None:
                    posture = classify_posture(keypoints)
                    draw_pose(frame, keypoints, x1, y1)
                    # Nếu phát hiện tư thế nằm sấp, gửi thông báo
                    if posture == "prone":
                        send_alert("Cảnh báo: Tư thế nằm sấp phát hiện!")

        # Hiển thị FPS trên khung hình
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Hiển thị khung hình với các thông tin vẽ
        cv2.imshow("Monitoring System", frame)

        # Nhấn 'q' để thoát
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video_feed()