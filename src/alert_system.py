import tkinter as tk
from datetime import datetime

# Hiển thị cảnh báo trên giao diện ứng dụng (popup)
def show_popup_alert(message):
    root = tk.Tk()
    root.title("Alert")

    # Thiết lập giao diện đơn giản cho popup
    label = tk.Label(root, text=message, padx=20, pady=20, font=("Arial", 14))
    label.pack()

    button = tk.Button(root, text="OK", command=root.destroy, padx=10, pady=5)
    button.pack()

    root.mainloop()

# Ghi log cảnh báo vào file
def log_alert(message, log_file="alert_log.txt"):
    with open(log_file, "a") as f:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{time_now} - {message}\n")
    print(f"Logged alert: {message}")

# Hàm chính để gửi cảnh báo
def send_alert(message):
    # Chỉ sử dụng popup để thông báo
    show_popup_alert(message)

    # Ghi log mỗi lần có cảnh báo
    log_alert(message)


