from flask import Flask, request, jsonify
import cv2
import os
from datetime import datetime
import threading

app = Flask(__name__)

# Biến toàn cục để quản lý camera và trạng thái video
camera_status = {"is_active": False, "camera_id": 0}
recorded_videos = []  # Danh sách các video đã ghi lại
stats_data = {"alerts": 0, "detections": 0, "postures": {"lying": 0, "sitting": 0, "standing": 0}}


@app.route("/")
def home():
    return jsonify({"message": "Welcome to the AI Sleep Monitoring System API!"})


@app.route("/check_cam", methods=["GET"])
def check_cam():
    """API để kiểm tra trạng thái camera."""
    try:
        cam_id = camera_status["camera_id"]
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            return jsonify({"status": "Camera not available"}), 400
        cap.release()
        return jsonify({"status": "Camera is active", "camera_id": cam_id}), 200
    except Exception as e:
        return jsonify({"status": "Error", "details": str(e)}), 500


@app.route("/replay", methods=["GET"])
def replay():
    """API để phát lại video đã ghi."""
    video_name = request.args.get("video_name")
    if not video_name or video_name not in recorded_videos:
        return jsonify({"error": "Video not found!"}), 404

    video_path = os.path.join("recorded_videos", video_name)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found!"}), 404

    return jsonify({"message": "Video replay ready", "video_path": video_path})


@app.route("/view_stats", methods=["GET"])
def view_stats():
    """API để xem thống kê."""
    return jsonify({"stats": stats_data})


@app.route("/record_video", methods=["POST"])
def record_video():
    """API để bắt đầu ghi video."""
    global camera_status, recorded_videos

    if camera_status["is_active"]:
        return jsonify({"error": "Camera is already in use!"}), 400

    # Lấy ID camera từ yêu cầu hoặc dùng mặc định
    camera_id = request.json.get("camera_id", 0)
    camera_status["camera_id"] = camera_id
    camera_status["is_active"] = True

    # Tạo file video
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"record_{now}.avi"
    video_path = os.path.join("recorded_videos", video_name)
    os.makedirs("recorded_videos", exist_ok=True)

    # Thread để ghi video
    def record():
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            camera_status["is_active"] = False
            return

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        while camera_status["is_active"]:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break

        cap.release()
        out.release()
        recorded_videos.append(video_name)
        camera_status["is_active"] = False

    threading.Thread(target=record).start()
    return jsonify({"message": "Recording started", "video_name": video_name})


@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    """API để dừng ghi video."""
    global camera_status
    if not camera_status["is_active"]:
        return jsonify({"error": "No active recording!"}), 400

    camera_status["is_active"] = False
    return jsonify({"message": "Recording stopped"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
