from flask import Flask, jsonify, Response, request
import threading
import os
import cv2
from combine import process_video_feed
from alert_system import send_alert, display_alert_statistics
app = Flask(__name__)

# Biến toàn cục
camera_status = {"is_active": False, "camera_id": 0}
recorded_videos = []  # Danh sách video đã quay
alerts_count = 0  # Biến lưu số lượng cảnh báo


### Phần CheckCam ###
@app.route("/checkcam/source", methods=["GET"])
def checkcam_source():
    """Phát luồng camera và xử lý AI từ combine.py."""
    camera_id = camera_status["camera_id"]
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        return jsonify({"error": "Camera not available"}), 400

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                # Xử lý khung hình qua combine.py
                processed_frame = process_video_feed()
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in video processing: {e}")
                break

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/checkcam/alert", methods=["GET"])
def checkcam_alert():
    """Trả về cảnh báo hiện tại từ backend."""
    global alerts_count
    return jsonify({"alerts_count": alerts_count})


@app.route("/checkcam/record", methods=["POST"])
def checkcam_record():
    """Quay video và thiết lập vùng giường."""
    global camera_status, recorded_videos

    if camera_status["is_active"]:
        return jsonify({"error": "Camera is already in use!"}), 400

    # Lấy ID camera từ yêu cầu
    camera_id = request.json.get("camera_id", 0)
    camera_status["camera_id"] = camera_id
    camera_status["is_active"] = True

    # Tạo file video
    video_name = f"record_{len(recorded_videos) + 1}.avi"
    video_path = os.path.join("recorded_videos", video_name)
    os.makedirs("recorded_videos", exist_ok=True)

    # Thread để quay video
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
                processed_frame = process_video_feed(frame)  # Xử lý khung hình trong combine.py
                out.write(processed_frame)
            else:
                break

        cap.release()
        out.release()
        recorded_videos.append(video_name)
        camera_status["is_active"] = False

    threading.Thread(target=record).start()
    return jsonify({"message": "Recording started", "video_name": video_name})


@app.route("/checkcam/resetbeds", methods=["POST"])
def checkcam_resetbeds():
    """Reset vùng giường."""
    from combine import save_bed_area  # Import tại đây để tránh dependency loop
    save_bed_area([])  # Reset vùng giường
    return jsonify({"message": "Bed areas reset successfully"})


### Phần Replay ###
@app.route("/replay/replay", methods=["GET"])
def replay_video():
    """Replay video đã quay."""
    video_name = request.args.get("video_name")
    if not video_name or video_name not in recorded_videos:
        return jsonify({"error": "Video not found!"}), 404

    video_path = os.path.join("recorded_videos", video_name)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found!"}), 404

    return jsonify({"message": "Video replay ready", "video_path": video_path})


@app.route("/replay/viewstats", methods=["GET"])
def replay_viewstats():
    """Trả về thống kê dựa trên cảnh báo."""
    global alerts_count
    return jsonify({
        "stats": {
            "alerts_count": alerts_count}  # Cập nhật theo nhu cầu
        }
    )


### Phần ViewStats ###
@app.route("/viewstats", methods=["GET"])
def viewstats():
    """Xem tổng hợp thống kê hệ thống."""
    return jsonify({"alerts_count": display_alert_statistics()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
