import cv2
from person_detection import detect_person, draw_bed_area
from pose_estimation import estimate_pose
from face_detection import detect_face
import mediapipe as mp

# Khởi tạo vẽ khung xương sử dụng MediaPipeq

# Vẽ bounding box xung quanh người
def draw_bounding_boxes(frame, persons):
    for person in persons:
        x1, y1, x2, y2 = map(int, person)  # Tọa độ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hộp màu xanh lá

# Vẽ khung xương người trên khung hình
def draw_pose(frame, keypoints):
    if keypoints:
        mp_drawing.draw_landmarks(frame, keypoints, mp_pose.POSE_CONNECTIONS)


# Hàm chính xử lý khung hình từ camera
def process_camera_feed():
    cap = cv2.VideoCapture(1)  # Sử dụng camera mặc định

    if not cap.isOpened():
        print("Error: Không thể mở camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break

        # Phát hiện người
        persons = detect_person(frame)
        draw_bounding_boxes(frame, persons)  # Vẽ bounding box xung quanh người

        # Vẽ bounding box cho vùng giường
        draw_bed_area(frame)

        # Phát hiện khuôn mặt và cảnh báo nếu khuôn mặt bị che
        detect_face(frame)

        # Nhận diện khung xương
        keypoints = estimate_pose(frame)
        draw_pose(frame, keypoints)  # Vẽ khung xương lên khung hình

        # Hiển thị khung hình
        cv2.imshow("Camera Feed - Bounding Box and Pose", frame)

        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_camera_feed()
