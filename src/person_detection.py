from ultralytics import YOLO
from alert_system import send_alert
import cv2

# Khởi tạo mô hình YOLO v10
model = YOLO("../data/yolov10n.pt")

# Định nghĩa vùng giường (tọa độ x1, y1, x2, y2)
BED_AREA = (0, 0, 600, 800)  # Ví dụ về tọa độ, có thể điều chỉnh theo kích thước thực tế

# Vẽ bounding box cho vùng giường
def draw_bed_area(frame):
    bed_x1, bed_y1, bed_x2, bed_y2 = BED_AREA
    cv2.rectangle(frame, (bed_x1, bed_y1), (bed_x2, bed_y2), (255, 0, 0), 2)  # Vẽ hộp màu xanh dương

# Hàm kiểm tra người có **ở ngoài** vùng giường không
def is_person_outside_bed(person_bbox):
    x1, y1, x2, y2 = map(int, person_bbox)
    bed_x1, bed_y1, bed_x2, bed_y2 = BED_AREA

    # Kiểm tra nếu người nằm **hoàn toàn ngoài** vùng giường
    if x2 < bed_x1 or x1 > bed_x2 or y2 < bed_y1 or y1 > bed_y2:
        return True
    return False

# Hàm phát hiện người và cảnh báo khi trẻ **ở ngoài** giường
def detect_person(frame):
    results = model(frame)
    persons = []

    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls[0])  # Lấy nhãn của đối tượng
            if class_id == 0:  # 0 là nhãn của "người"
                bbox = detection.xyxy[0].cpu().numpy()
                persons.append(bbox)

                # Kiểm tra nếu người **ở ngoài** chỗ ngủ và đưa ra cảnh báo
                if is_person_outside_bed(bbox):
                    send_alert("Cảnh báo: Trẻ đã rời khỏi giường!")

    return persons
