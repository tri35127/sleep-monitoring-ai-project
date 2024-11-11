import cv2
import numpy as np
from alert_system import send_alert  # Đảm bảo module này có sẵn
import torch
from ultralytics import YOLO

# Kiểm tra xem CUDA có khả dụng không
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Đang sử dụng GPU: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Đang sử dụng MPS")
else:
    device = torch.device("cpu")
    print("CUDA không khả dụng. Đang sử dụng CPU.")

# Tải mô hình YOLO Pose
model = YOLO("D:/sleep-monitoring-ai-project/data/yolo11x-pose.pt")  # Thay thế bằng đường dẫn đúng đến mô hình YOLO của bạn

initial_posture = None  # Theo dõi tư thế ban đầu

def estimate_pose(frame):
    # Thực hiện suy luận với YOLO trên thiết bị đã chỉ định (GPU hoặc CPU)
    results = model.predict(frame, verbose=False, device=device)

    keypoints = []
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # Lấy keypoints dưới dạng mảng numpy
    return keypoints if len(keypoints) > 0 else None  # Trả về None nếu không phát hiện keypoints

# Phân loại tư thế dựa trên vị trí tương đối của keypoints vai và hông
def classify_posture(keypoints):
    if keypoints is None:
        return "unknown"

    # Xác định các keypoints cho vai trái, vai phải, hông trái, hông phải
    left_shoulder, right_shoulder, left_hip, right_hip = None, None, None, None

    # Giả sử keypoints[0] chứa tất cả các keypoints cho các bộ phận cơ thể
    for i, keypoint in enumerate(keypoints[0]):
        if i == 5:  # Vai trái
            left_shoulder = keypoint
        elif i == 6:  # Vai phải
            right_shoulder = keypoint
        elif i == 11:  # Hông trái
            left_hip = keypoint
        elif i == 12:  # Hông phải
            right_hip = keypoint

    # Đảm bảo tất cả keypoints đều không phải None và có tọa độ hợp lệ
    if (left_shoulder is not None and right_shoulder is not None and
        left_hip is not None and right_hip is not None and
        left_shoulder.size > 0 and right_shoulder.size > 0 and
        left_hip.size > 0 and right_hip.size > 0):
        
        # Nếu vai trái ở bên phải vai phải và hông trái ở bên phải hông phải,
        # điều này có thể chỉ ra tư thế nằm ngửa (supine).
        if left_shoulder[0] > right_shoulder[0] and left_hip[0] > right_hip[0]:
            return "supine"
        
        # Nếu vai trái ở bên trái vai phải và hông trái ở bên trái hông phải,
        # điều này có thể chỉ ra tư thế nằm sấp (prone).
        elif left_shoulder[0] < right_shoulder[0] and left_hip[0] < right_hip[0]:
            return "prone"

    return "unknown"  # Trả về unknown nếu các keypoints không đủ để phân loại tư thế

# Phát hiện nếu tư thế đã thay đổi từ nằm ngửa sang nằm sấp hoặc ngược lại
def has_posture_changed(current_posture):
    global initial_posture
    if initial_posture is None:
        initial_posture = current_posture
        return False  # Không thay đổi vì đây là tư thế lần đầu được phát hiện

    # Kiểm tra nếu tư thế đã thay đổi
    if current_posture != initial_posture:
        initial_posture = current_posture  # Cập nhật tư thế ban đầu với tư thế mới
        return True  # Tư thế đã thay đổi
    return False

def draw_pose(frame, keypoints, offset_x=0, offset_y=0):
    if keypoints is not None:
        for i, keypoint in enumerate(keypoints[0]):
            if (keypoint[0], keypoint[1]) != (0, 0):  # Kiểm tra nếu keypoint hợp lệ
                x, y = int(keypoint[0]) + offset_x, int(keypoint[1]) + offset_y
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Vẽ keypoint (màu đỏ)
                cv2.putText(frame, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def prone_alert(frame):
    keypoints = estimate_pose(frame)
    current_posture = classify_posture(keypoints)

    if current_posture == "prone":
        # Nếu phát hiện tư thế nằm sấp, gửi thông báo
        send_alert("Tư thế nằm sấp phát hiện!")

    if has_posture_changed(current_posture):
        print(f"Tư thế đã thay đổi thành: {current_posture}")

    draw_pose(frame, keypoints)





