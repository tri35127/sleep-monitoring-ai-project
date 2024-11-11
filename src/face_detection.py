import cv2
import numpy as np
from yoloface import face_detector

# Khởi tạo mô hình yoloface
model = face_detector.YoloDetector(target_size=720, device="mps", min_face=90)


def detect_faces(frame):
    # Chuyển đổi frame sang định dạng cần thiết cho yoloface
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes, points = model.predict(frame_rgb)

    # Trả về danh sách các bounding box khuôn mặt
    return bboxes, points


def draw_faces(frame, bboxes):
    # Vẽ bounding box của khuôn mặt trên khung hình
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame
