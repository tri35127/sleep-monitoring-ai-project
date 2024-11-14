import cv2
import json
from ultralytics import YOLO
from alert_system import send_alert
import torch

device = torch.device("cuda:0")  # Set the device to GPU
model = YOLO("D:/sleep-monitoring-ai-project/data/yolo11l.pt").to(device)
model.export(format="onnx")
# Load the exported ONNX model
onnx_model = YOLO("D:/sleep-monitoring-ai-project/data/yolo11l.onnx")
CONFIG_FILE = "D:/sleep-monitoring-ai-project/config/config.json"


# Vẽ bounding box cho mỗi người
def draw_bounding_boxes(frame, persons):
    for person in persons:
        x1, y1, x2, y2 = map(int, person)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Xanh lá

# Đọc danh sách vùng giường từ file config
def load_bed_area():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)["bed_areas"]
    except FileNotFoundError:
        return None

# Lưu danh sách vùng giường vào file config
def save_bed_area(bed_areas):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"bed_areas": bed_areas}, f)

# Vẽ bounding box cho vùng giường
def draw_bed_area(frame, bed_area):
    bed_x1, bed_y1, bed_x2, bed_y2 = bed_area
    cv2.rectangle(frame, (bed_x1, bed_y1), (bed_x2, bed_y2), (255, 0, 0), 2)  # Xanh dương

# Tạo vùng giường từ bounding box của người, với tỉ lệ phóng to 1.05x
def create_bed_area_from_person_bbox(bbox, scale_factor=1.05):
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    # Tính kích thước phóng to
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Tính tọa độ mới dựa trên kích thước phóng to
    new_x1 = x1 - (new_width - width) // 2
    new_y1 = y1 - (new_height - height) // 2
    new_x2 = new_x1 + new_width
    new_y2 = new_y1 + new_height

    return new_x1, new_y1, new_x2, new_y2

# Tính diện tích của hình chữ nhật
def calculate_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)

# Tính diện tích giao nhau giữa bounding box của người và vùng giường
def calculate_intersection_area(person_bbox, bed_area):
    p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
    b_x1, b_y1, b_x2, b_y2 = bed_area

    # Tính các tọa độ giao nhau
    inter_x1 = max(p_x1, b_x1)
    inter_y1 = max(p_y1, b_y1)
    inter_x2 = min(p_x2, b_x2)
    inter_y2 = min(p_y2, b_y2)

    # Tính diện tích giao nhau
    return calculate_area(inter_x1, inter_y1, inter_x2, inter_y2)

# Kiểm tra nếu người ở ngoài vùng giường
def is_person_outside_bed(person_bbox, bed_area, threshold=0.5):
    person_area = calculate_area(*map(int, person_bbox))
    intersection_area = calculate_intersection_area(person_bbox, bed_area)

    # Tính tỉ lệ diện tích giao nhau so với diện tích của người
    if person_area > 0:
        overlap_ratio = intersection_area / person_area
        return overlap_ratio < threshold  # True nếu người ở ngoài vùng giường
    return False

# Kiểm tra trạng thái ngồi dựa trên giao nhau 90 độ và box gần vuông
def is_sitting(person_bbox, bed_area, overlap_threshold=0.5, aspect_ratio_threshold=0.9):
    p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
    b_x1, b_y1, b_x2, b_y2 = bed_area

    # Kiểm tra overlap dựa trên diện tích giao nhau
    person_width = abs(p_x2 - p_x1)
    person_height = abs(p_y2 - p_y1)
    bed_width = abs(b_x2 - b_x1)
    bed_height = abs(b_y2 - b_y1)

    # Tính các tọa độ giao nhau
    inter_x1 = max(p_x1, b_x1)
    inter_y1 = max(p_y1, b_y1)
    inter_x2 = min(p_x2, b_x2)
    inter_y2 = min(p_y2, b_y2)
    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    person_area = person_width * person_height
    overlap_ratio = intersection_area / person_area if person_area > 0 else 0

    # Kiểm tra box gần vuông
    aspect_ratio = min(person_width, person_height) / max(person_width, person_height)

    # Kết hợp hai điều kiện
    return overlap_ratio > overlap_threshold and aspect_ratio > aspect_ratio_threshold

# Phát hiện người và cảnh báo khi người ở ngoài giường hoặc đang ngồi
def detect_person(frame, bed_areas=None):
    results = model(frame, verbose=False, imgsz=320, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    persons = []

    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls[0])  # Lấy nhãn của đối tượng
            if class_id == 0:  # 0 là nhãn của "người"
                bbox = detection.xyxy[0].cpu().numpy()
                persons.append(bbox)

                # Kiểm tra nếu có vùng giường và phát cảnh báo
                if bed_areas:
                    for bed_area in bed_areas:
                        if is_person_outside_bed(bbox, bed_area):
                            send_alert("Cảnh báo: Trẻ đã rời khỏi giường!")
                        elif is_sitting(bbox, bed_area):
                            send_alert("Cảnh báo: Trẻ đang ngồi!")

    return persons
