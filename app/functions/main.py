from flask import Flask, jsonify, Response, request
import threading
import cv2
import queue
import json
from person_detection import detect_person, create_bed_area_from_person_bbox, save_bed_area
import time
from combine import process_video_feed
from alert_system import display_last_alert
import configparser
import os

# Construct the relative path to config.ini
config_path = os.path.realpath("../config/config.ini")
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)

app = Flask(__name__)
event_queue = queue.Queue()  # Queue để lưu trữ các sự kiện cần gửi

@app.route('/checkcam/source', methods=['GET'])
def video_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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




### Phần ViewStats ###
# Background task để cập nhật dữ liệu vào queue
def push_updates_to_queue():
    last_stats = None
    while True:
        stats = display_last_alert()
        if stats != last_stats:
            last_stats = stats
            data = {"message": {stats}}
            event_queue.put(f"data: {json.dumps(str(data), ensure_ascii=False)}\n\n")
        else:
            event_queue.put(None)
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