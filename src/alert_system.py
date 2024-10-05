from datetime import datetime, timedelta

# Biến lưu thời gian lần cảnh báo cuối cùng
last_alert_time = None

# Thời gian tối thiểu giữa các lần cảnh báo (10 giây)
ALERT_INTERVAL = timedelta(seconds=10)

# Hiển thị cảnh báo bằng cách in ra màn hình
def show_alert(message):
    print(f"ALERT: {message}")

# Ghi log cảnh báo vào file
def log_alert(message, log_file="alert_log.txt"):
    with open(log_file, "a", encoding="utf-8") as f:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{time_now} - {message}\n")
    print(f"Logged alert: {message}")


# Hàm kiểm tra xem đã đủ thời gian giữa các lần cảnh báo chưa
def can_send_alert():
    global last_alert_time
    current_time = datetime.now()

    if last_alert_time is None or (current_time - last_alert_time) >= ALERT_INTERVAL:
        return True
    return False

# Hàm chính để gửi cảnh báo
def send_alert(message):
    global last_alert_time  # Sử dụng biến toàn cục để cập nhật thời gian
    # Chỉ gửi cảnh báo nếu đủ thời gian (10 giây)
    if can_send_alert():
        # In thông báo ra màn hình
        show_alert(message)

        # Ghi log mỗi lần có cảnh báo
        log_alert(message)

        # Cập nhật thời gian cảnh báo cuối cùng
        last_alert_time = datetime.now()
    else:
       pass
