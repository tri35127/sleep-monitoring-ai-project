import numpy as np
import cv2
import mediapipe as mp

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose

# Khởi tạo đối tượng pose estimation
pose = mp_pose.Pose()

# Hàm nhận diện khung xương
def estimate_pose(frame):
    # Chuyển đổi khung hình sang RGB vì MediaPipe yêu cầu định dạng RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý khung hình để nhận diện tư thế
    results = pose.process(rgb_frame)

    # Kiểm tra nếu có keypoints
    if results.pose_landmarks:
        return results.pose_landmarks
    else:
        return None



# Kiểm tra tư thế bất thường
def is_abnormal_pose(keypoints, threshold=0.5):
    abnormal_pose_detected = False

    # Lấy keypoint cho đầu, cổ và vai
    head = keypoints[mp_pose.PoseLandmark.NOSE.value]  # Keypoint của mũi (tương đương đầu)
    left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Kiểm tra nếu đầu thấp hơn vai (trẻ nằm sấp)
    if head.y > left_shoulder.y and head.y > right_shoulder.y:
        abnormal_pose_detected = True

    # Kiểm tra vai không đều (ví dụ: tư thế tay chân co rút bất thường)
    if np.abs(left_shoulder.y - right_shoulder.y) > 0.1:  # 0.1 là ngưỡng chênh lệch giữa vai
        abnormal_pose_detected = True

    return abnormal_pose_detected

