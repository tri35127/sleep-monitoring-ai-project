import cv2
from person_detection import detect_person, draw_bed_area, save_bed_area, load_bed_area, select_bed_area

# Vẽ bounding box xung quanh người
def draw_bounding_boxes(frame, persons):
    for person in persons:
        x1, y1, x2, y2 = map(int, person)  # Tọa độ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hộp màu xanh lá

# Vòng lặp chính để xử lý video
def process_video_feed(video_path):
    cap = cv2.VideoCapture(video_path)
    bed_area = load_bed_area()  # Tải vùng giường từ file config

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Đảm bảo kích thước khung hình không bị thay đổi
        height, width = frame.shape[:2]

        if bed_area is None:
            bed_area = select_bed_area(frame)  # Chọn vùng giường nếu chưa có
            if bed_area is not None:
                save_bed_area(bed_area)  # Lưu vào file config sau khi chọn

        # Vẽ vùng giường lên khung hình
        draw_bed_area(frame, bed_area)

        # Phát hiện người
        persons = detect_person(frame, bed_area)

        # Vẽ bounding box cho người
        draw_bounding_boxes(frame, persons)

        # Hiển thị video feed
        cv2.imshow("Video Feed", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "../data/video/IMG_1671.MOV"  # Thay thế bằng đường dẫn tới video
    process_video_feed(video_path)
