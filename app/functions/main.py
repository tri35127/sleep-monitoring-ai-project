from flask import Flask, jsonify, Response, request
import threading
import os
import cv2
import queue
import json
from datetime import datetime
from person_detection import detect_person, create_bed_area_from_person_bbox, save_bed_area
import time
from combine import process_video_feed
from alert_system import display_last_alert
app = Flask(__name__)

event_queue = queue.Queue()  # Queue để lưu trữ các sự kiện cần gửi

# Biến toàn cục
camera_status = {"is_active": False, "camera_id": 0}
recorded_videos = []  # Danh sách video đã quay
alerts_count = 0  # Biến lưu số lượng cảnh báo
@app.route('/checkcam/source', methods=['GET'])
def video_feed():
    cap = cv2.VideoCapture(0)
    def generate_frames():
        while True:
            success, frame = process_video_feed(cap)
            if not success:
                break
            else:
                # Mã hóa khung hình thành JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                # Truyền dữ liệu khung hình
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/checkcam/resetbeds", methods=["POST"])
def checkcam_resetbeds():
    """Reset vùng giường."""
    cap = cv2.VideoCapture(0)
    success, frame = process_video_feed(cap)
    bed_areas = []
    persons = detect_person(frame, None)
    for person in persons:
        bed_area = create_bed_area_from_person_bbox(person)
        bed_areas.append(bed_area)
    save_bed_area(bed_areas)
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

# Background task để cập nhật dữ liệu vào queue
def push_updates_to_queue():
    while True:
        stats = display_last_alert()
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = {"message": {stats}}
        event_queue.put(f"data: {json.dumps(str(data), ensure_ascii=False)}\n\n")
        time.sleep(5) # Cập nhật mỗi 5 giây


# Endpoint SSE để gửi dữ liệu realtime
@app.route('/viewstats', methods=['GET'])
def viewstats():
    def event_stream():
        while True:
            # Lấy sự kiện từ queue và gửi đến client
            event = event_queue.get()
            yield event

    return Response(event_stream(), content_type='text/event-stream')

# Khởi chạy background task
threading.Thread(target=push_updates_to_queue, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)