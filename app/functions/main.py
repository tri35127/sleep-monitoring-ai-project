from flask import Flask, jsonify, Response, request
import threading
import cv2
import queue
import json
from person_detection import detect_person, create_bed_area_from_person_bbox, save_bed_area
import time
from combine import process_video_feed
from alert_system import display_last_alert
from collections import Counter
import configparser
import torch
import os
if torch.backends.mps.is_available():
    config_path = os.path.realpath("../config/config.ini")
else:
    config_path = os.path.realpath("../sleep-monitoring-ai-project/app/config/config.ini")
# Create a configuration object
config = configparser.ConfigParser()    
config.read(config_path)

app = Flask(__name__)
event_queue = queue.Queue()  # Queue để lưu trữ các sự kiện cần gửi
camera_lock = threading.Lock()
def access_camera():
    with camera_lock:
        cap = cv2.VideoCapture(config.getint("camera", "camera_id"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.getint("camera", "width"))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.getint("camera", "height"))
        return cap


@app.route("/checkcam/source", methods=['GET'])
def video_feed():
    cap = access_camera()
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



@app.route("/resetbeds", methods=["POST"])
def checkcam_resetbeds():
    """Reset vùng giường."""
    cap = access_camera()
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

alert_counter = Counter()
alert = []

def push_updates_to_queue():
    last_stats = None
    while True:
        stats = display_last_alert()
        stats = str(stats)
        if stats != last_stats:
            last_stats = stats
            data = {"message": stats}
            alert_counter[stats] += 1  # Tăng số lượng cho loại cảnh báo này
            alert.append(stats)
            event_queue.put(f"data: {json.dumps(data, ensure_ascii=False)}\n\n")
        else:  
            event_queue.put(":\n\n")
        time.sleep(5) # Cập nhật mỗi 5 giây


# Endpoint SSE để gửi dữ liệu realtime
@app.route("/viewstats", methods=['GET'])
def viewstats():
    def event_stream():
        while True:
            # Lấy sự kiện từ queue và gửi đến client
            event = event_queue.get()
            yield event

    return Response(event_stream(), content_type='text/event-stream')


@app.route("/viewall", methods=["GET"])
def view_all():
    """Trả về thống kê dạng danh sách."""
    # Loại bỏ mục 'None'
    filtered_alert_counter = {key: value for key, value in alert_counter.items() if key != "None"}
    
    # Tạo danh sách các cặp cảnh báo
    stats = [{key: value} for key, value in filtered_alert_counter.items()]
    
    # Trả về dưới dạng JSON
    return jsonify({"stats": stats})

# Khởi chạy background task
threading.Thread(target=push_updates_to_queue, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)