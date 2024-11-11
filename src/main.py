import cv2
import time
from person_detection import detect_person, draw_bed_area, load_bed_area, create_bed_area_from_person_bbox, save_bed_area, draw_bounding_boxes, is_person_outside_bed
from face_detection import detect_faces, draw_faces
from pose_estimation import estimate_pose, classify_posture, draw_pose
from alert_system import send_alert


# Main video processing loop
def process_video_feed():
    cap = cv2.VideoCapture(1)  # Sử dụng camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

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

        # Vẽ vùng giường nếu đã có
        if bed_areas:
            for bed_area in bed_areas:
                draw_bed_area(frame, bed_area)

        # Phát hiện người và kiểm tra bất thường
        persons = detect_person(frame, bed_areas)  # Phát hiện người
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = map(int, person)
            person_frame = frame[y1:y2, x1:x2]  # Cắt khung hình theo bounding box của người

            # Kiểm tra nếu người ngoài giường
            if bed_areas:
                for bed_area in bed_areas:
                    if is_person_outside_bed(person, bed_area):
                        send_alert("Cảnh báo: Trẻ đã rời khỏi giường!")

            # Vẽ bounding box cho mỗi người
            draw_bounding_boxes(frame, persons)

            # Trong vòng lặp chính của hàm process_video_feed
            persons = detect_person(frame, bed_areas)  # Phát hiện người
            for i, person in enumerate(persons):
                # Phát hiện khuôn mặt trong vùng bounding box của người
                x1, y1, x2, y2 = map(int, person)
                person_frame = frame[y1:y2, x1:x2]
                face_bboxes, _ = detect_faces(person_frame)

                # Vẽ bounding box của khuôn mặt nếu phát hiện thấy
                frame = draw_faces(frame, face_bboxes)

                # Kiểm tra điều kiện để cảnh báo nếu khuôn mặt bị che hoặc không phát hiện được
                if not face_bboxes:
                    send_alert("Cảnh báo: Khuôn mặt bị che hoặc không phát hiện!")


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
