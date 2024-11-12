import cv2
import numpy as np
import face_recognition
from alert_system import send_alert

# Hàm phát hiện khuôn mặt và kiểm tra xem mặt có bị che hay không
def detect_face(frame, upsample=1, model="hog"):
    # Chuyển khung hình từ OpenCV (BGR) sang RGB cho thư viện face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện vị trí khuôn mặt
    face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=upsample, model=model)

    if face_locations:
        # Kiểm tra nếu khuôn mặt bị che một phần
        for (top, right, bottom, left) in face_locations:
            face_frame = frame[top:bottom, left:right]
            if is_face_obstructed(face_frame):
                send_alert("Cảnh báo: Khuôn mặt bị che!")
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        send_alert("Cảnh báo: Không phát hiện khuôn mặt!")

    return frame

# Hàm kiểm tra xem khuôn mặt có bị che không
def is_face_obstructed(face_frame, obstruction_threshold=0.5):
    gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_face, 100, 255, cv2.THRESH_BINARY_INV)  # Tạo mask cho vùng bị che

    # Tính tỷ lệ vùng bị che
    obstructed_area = np.sum(mask == 255)
    total_area = face_frame.shape[0] * face_frame.shape[1]
    obstruction_ratio = obstructed_area / total_area

    # Trả về True nếu tỷ lệ che vượt ngưỡng
    return obstruction_ratio > obstruction_threshold
