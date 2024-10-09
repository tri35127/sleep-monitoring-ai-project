import cv2
import json
from ultralytics import YOLO
from alert_system import send_alert

# Khởi tạo mô hình YOLO v10
model = YOLO("../data/yolov10n.pt")

# Đường dẫn tới file config để lưu tọa độ vùng giường
CONFIG_FILE = "config/bed_config.json"


# Đọc vùng giường từ file config
def load_bed_area():
    try:
        with open(CONFIG_FILE, "r") as f:
            return tuple(json.load(f)["bed_area"])
    except FileNotFoundError:
        return None  # Trả về None nếu file chưa tồn tại


# Lưu vùng giường vào file config
def save_bed_area(bed_area):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"bed_area": bed_area}, f)
    except FileNotFoundError:
        load_bed_area()


# Vẽ bounding box cho vùng giường
def draw_bed_area(frame, bed_area):
    bed_x1, bed_y1, bed_x2, bed_y2 = bed_area
    cv2.rectangle(frame, (bed_x1, bed_y1), (bed_x2, bed_y2), (255, 0, 0), 2)  # Màu xanh dương


# Vẽ bounding box cho người
def draw_person_bboxes(frame, persons):
    for bbox in persons:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Màu xanh lá


def calculate_area(x1, y1, x2, y2):
    """Tính diện tích của hình chữ nhật."""
    return max(0, x2 - x1) * max(0, y2 - y1)

def calculate_intersection_area(person_bbox, bed_area):
    """Tính diện tích giao nhau giữa bounding box của người và vùng giường."""
    p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
    b_x1, b_y1, b_x2, b_y2 = bed_area

    # Tính các tọa độ giao nhau
    inter_x1 = max(p_x1, b_x1)
    inter_y1 = max(p_y1, b_y1)
    inter_x2 = min(p_x2, b_x2)
    inter_y2 = min(p_y2, b_y2)

    # Tính diện tích giao nhau
    return calculate_area(inter_x1, inter_y1, inter_x2, inter_y2)


def calculate_area(x1, y1, x2, y2):
    """Tính diện tích của hình chữ nhật."""
    return max(0, x2 - x1) * max(0, y2 - y1)

def calculate_intersection_area(person_bbox, bed_area):
    """Tính diện tích giao nhau giữa bounding box của người và vùng giường."""
    p_x1, p_y1, p_x2, p_y2 = map(int, person_bbox)
    b_x1, b_y1, b_x2, b_y2 = bed_area

    # Tính các tọa độ giao nhau
    inter_x1 = max(p_x1, b_x1)
    inter_y1 = max(p_y1, b_y1)
    inter_x2 = min(p_x2, b_x2)
    inter_y2 = min(p_y2, b_y2)

    # Tính diện tích giao nhau
    return calculate_area(inter_x1, inter_y1, inter_x2, inter_y2)

# Hàm kiểm tra người có ở ngoài vùng giường không
def is_person_outside_bed(person_bbox, bed_area, threshold=0.5):
    """Kiểm tra xem người có ở ngoài vùng giường với tỉ lệ nhất định không."""
    person_area = calculate_area(*map(int, person_bbox))
    intersection_area = calculate_intersection_area(person_bbox, bed_area)

    # Tính tỉ lệ diện tích giao nhau so với diện tích của người
    if person_area > 0:  # Kiểm tra diện tích của người
        overlap_ratio = intersection_area / person_area
        return overlap_ratio < threshold  # Trả về True nếu người ở ngoài vùng giường theo tỉ lệ
    return False  # Nếu không có diện tích của người thì trả về True (điều này có thể được điều chỉnh nếu cần)


# Hàm để vẽ các điểm điều chỉnh trên khung hình
def draw_adjustable_area(frame, points):
    for (x, y) in points:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Vẽ các điểm điều chỉnh
    # Vẽ đường nối giữa các điểm
    for i in range(len(points)):
        cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (255, 0, 0), 2)


# Hàm xử lý việc kéo chỉnh vùng
def adjust_bed_area(frame, points):
    dragging = False
    dragged_point = None

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, dragged_point

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (px, py) in enumerate(points):
                if abs(x - px) < 10 and abs(y - py) < 10:  # Nếu nhấp gần một điểm điều chỉnh
                    dragging = True
                    dragged_point = i
                    break

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            if dragged_point is not None:
                points[dragged_point] = (x, y)  # Cập nhật tọa độ điểm đang kéo

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            dragged_point = None

    cv2.namedWindow("Adjust Bed Area")
    cv2.setMouseCallback("Adjust Bed Area", on_mouse)

    while True:
        clone_frame = frame.copy()
        draw_adjustable_area(clone_frame, points)

        cv2.imshow("Adjust Bed Area", clone_frame)

        # Nhấn 'q' để thoát sau khi điều chỉnh
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Adjust Bed Area")
    return points  # Trả về các điểm đã chỉnh sửa


# Chọn và điều chỉnh vùng giường
def select_bed_area(frame):
    # Chọn vùng giường ban đầu bằng cv2.selectROI
    bed_area = cv2.selectROI("Select Bed Area", frame, fromCenter=False)
    bed_area = list(map(int, bed_area))  # Chuyển sang list

    if bed_area[2] > 0 and bed_area[3] > 0:  # Kiểm tra xem có chọn hay không
        # Khởi tạo các điểm cho vùng giường (hình chữ nhật)
        points = [
            (bed_area[0], bed_area[1]),  # Top-left
            (bed_area[0] + bed_area[2], bed_area[1]),  # Top-right
            (bed_area[0] + bed_area[2], bed_area[1] + bed_area[3]),  # Bottom-right
            (bed_area[0], bed_area[1] + bed_area[3])  # Bottom-left
        ]
        adjusted_points = adjust_bed_area(frame, points)  # Cho phép điều chỉnh tự do sau khi chọn

        # Cập nhật bed_area với tọa độ mới từ adjusted_points
        updated_bed_area = (
            min(adjusted_points[0][0], adjusted_points[3][0]),  # X1
            min(adjusted_points[0][1], adjusted_points[1][1]),  # Y1
            max(adjusted_points[1][0], adjusted_points[2][0]),  # X2
            max(adjusted_points[2][1], adjusted_points[3][1])   # Y2
        )

        save_bed_area(updated_bed_area)  # Lưu tọa độ mới vào file
        return updated_bed_area  # Trả về tọa độ mới cho bed_area

    return None


# Phát hiện người và cảnh báo khi người ở ngoài giường
def detect_person(frame, bed_area):
    results = model(frame, verbose=False)
    persons = []

    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls[0])  # Lấy nhãn của đối tượng
            if class_id == 0:  # 0 là nhãn của "người"
                bbox = detection.xyxy[0].cpu().numpy()
                persons.append(bbox)

                if is_person_outside_bed(bbox, bed_area):
                    send_alert("Cảnh báo: Trẻ đã rời khỏi giường!")

    return persons