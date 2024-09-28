import cv2
import mediapipe as mp
from alert_system import send_alert
# Khởi tạo mô hình nhận diện khuôn mặt
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# Hàm phát hiện khuôn mặt
def detect_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển đổi khung hình sang RGB
    results = mp_face_detection.process(rgb_frame)

    if not results.detections:
        # Không tìm thấy khuôn mặt, phát cảnh báo
        send_alert("Cảnh báo: Khuôn mặt bị che, không thể nhận diện!")
        return False
    return True
