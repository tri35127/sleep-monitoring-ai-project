from ultralytics import YOLO

# Khởi tạo mô hình YOLO v10 từ Ultralytics
model = YOLO("../data/yolov10n.pt")  # Đường dẫn tới tệp mô hình đã huấn luyện


# Hàm phát hiện người sử dụng YOLO v10
def detect_person(frame):
    results = model(frame)  # Dự đoán trên khung hình

    persons = []
    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls[0])  # Lấy nhãn của đối tượng
            if class_id == 0:  # 0 là nhãn của "người" trong mô hình YOLO
                bbox = detection.xyxy[0].cpu().numpy()  # Tọa độ hộp bao quanh đối tượng
                persons.append(bbox)

    return persons

