import cv2
import time
import psutil
import subprocess
import matplotlib.pyplot as plt
from person_detection import detect_person, draw_bed_area, load_bed_area, create_bed_area_from_person_bbox, save_bed_area, draw_bounding_boxes, is_person_outside_bed, is_sitting
from keypoint import estimate_pose, detect_convulsive_movement, draw_pose
from alert_system import send_alert, display_alert_statistics
import numpy as np
# Biến lưu các metric
fps_list = []
response_times = []
cpu_usages = []
memory_usages = []
gpu_usages = []
gpu_memory_usages = []

def get_gpu_usage():
    try:
        # Chạy lệnh `nvidia-smi` và lấy thông tin GPU
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        # Kết quả từ nvidia-smi sẽ là chuỗi dạng "30, 1024" với các giá trị GPU utilization và memory used
        gpu_util, gpu_mem = result.strip().split(", ")
        return int(gpu_util), int(gpu_mem)
    except Exception as e:
        print("Unable to retrieve GPU information:", e)
        return None, None

def process_video_feed(cap):
    bed_areas = load_bed_area()
    prev_frame_time = 0

    while cap.isOpened():
        start_time = time.time()  # Bắt đầu đếm thời gian xử lý frame
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Tính FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        fps_list.append(fps)

        # Thêm chỉ số CPU và RAM
        cpu_usages.append(psutil.cpu_percent())
        memory_usages.append(psutil.virtual_memory().percent)

        # Lấy thông số GPU nếu có
        gpu_util, gpu_mem = get_gpu_usage()
        if gpu_util is not None:
            gpu_usages.append(gpu_util)
            gpu_memory_usages.append(gpu_mem)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('b'):
            bed_areas = []
            persons = detect_person(frame, None)
            for person in persons:
                bed_area = create_bed_area_from_person_bbox(person)
                bed_areas.append(bed_area)
            save_bed_area(bed_areas)

        if bed_areas:
            for bed_area in bed_areas:
                draw_bed_area(frame, bed_area)

        persons = detect_person(frame, bed_areas)
        for person in persons:
            x1, y1, x2, y2 = map(int, person)
            person_frame = frame[y1:y2, x1:x2]
            draw_bounding_boxes(frame, persons)
            if bed_areas:
                for bed_area in bed_areas:
                    if is_sitting(person, bed_area):
                        send_alert("Canh bao tre dang ngoi!")
                    elif is_person_outside_bed(person, bed_area):
                        send_alert("Canh bao tre roi khoi giuong!")
                    else:
                        keypoints = estimate_pose(person_frame)
                    if keypoints is not None:
                            
                        if detect_convulsive_movement(keypoints):
                            send_alert("Tre ngu khong ngon!")  # Alert for erratic movement
                        draw_pose(frame, keypoints, x1, y1)

        # Tính thời gian phản hồi cho mỗi khung hình
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)

        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Monitoring System", frame)
        cv2.waitKey(0)
        if key == ord('q'):
            break
        return ret, frame
    cap.release()
    cv2.destroyAllWindows()

    # Vẽ biểu đồ các thông số
    #plot_performance_metrics(fps_list, response_times, cpu_usages, memory_usages, gpu_usages, gpu_memory_usages)

def plot_performance_metrics(fps, response_times, cpu, memory, gpu, gpu_memory, fps_avg=30):
    skip_frames = int(0.75 * fps_avg)  # Số khung hình bỏ qua để tương ứng với 0.75 giây

    # Cắt bỏ các giá trị trong 0.75 giây đầu
    fps = fps[skip_frames:]
    response_times = response_times[skip_frames:]
    cpu = cpu[skip_frames:]
    memory = memory[skip_frames:]
    gpu = gpu[skip_frames:]
    gpu_memory = gpu_memory[skip_frames:]

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(fps, label="FPS")
    plt.xlabel("Time")
    plt.ylabel("FPS")
    plt.title("FPS over Time")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(response_times, label="Response Time (s)", color='orange')
    plt.xlabel("Time")
    plt.ylabel("Time (s)")
    plt.title("Response Time per Frame")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(cpu, label="CPU Usage (%)", color='green')
    plt.xlabel("Time")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage over Time")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(memory, label="Memory Usage (%)", color='red')
    plt.xlabel("Time")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage over Time")
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(gpu, label="GPU Usage (%)", color='purple')
    plt.xlabel("Time")
    plt.ylabel("GPU Usage (%)")
    plt.title("GPU Usage over Time")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(gpu_memory, label="GPU Memory Usage (MiB)", color='brown')
    plt.xlabel("Time")
    plt.ylabel("GPU Memory Usage (MiB)")
    plt.title("GPU Memory Usage over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_statistics(data, label):
    """Tính toán và in các thông số thống kê cho dữ liệu sau 0.75 giây."""
    if len(data) == 0:
        print(f"{label} Statistics: No data available")
        return
    
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_dev = np.std(data)
    
    print(f"{label} Statistics:")
    print(f"  Min: {min_val:.2f}")
    print(f"  Max: {max_val:.2f}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Std Dev (Stability): {std_dev:.2f}")
    print()

def display_performance_statistics(fps_list, response_times, cpu_usages, memory_usages, gpu_usages, gpu_memory_usages, fps_avg=30):
    skip_frames = int(0.75 * fps_avg)  # Số khung hình bỏ qua để tương ứng với 0.75 giây

    # Cắt bỏ các giá trị trong 0.75 giây đầu
    fps_list = fps_list[skip_frames:]
    response_times = response_times[skip_frames:]
    cpu_usages = cpu_usages[skip_frames:]
    memory_usages = memory_usages[skip_frames:]
    gpu_usages = gpu_usages[skip_frames:]
    gpu_memory_usages = gpu_memory_usages[skip_frames:]

    # Tính toán và hiển thị thống kê cho các danh sách đã cắt
    calculate_statistics(fps_list, "FPS")
    calculate_statistics(response_times, "Response Time (s)")
    calculate_statistics(cpu_usages, "CPU Usage (%)")
    calculate_statistics(memory_usages, "Memory Usage (%)")
    calculate_statistics(gpu_usages, "GPU Usage (%)")
    calculate_statistics(gpu_memory_usages, "GPU Memory Usage (MiB)")

if __name__ == "__main__":
    process_video_feed()
    display_alert_statistics()
    display_performance_statistics(fps_list, response_times, cpu_usages, memory_usages, gpu_usages, gpu_memory_usages)