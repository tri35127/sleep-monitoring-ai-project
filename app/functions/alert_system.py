from datetime import datetime, timedelta
from collections import Counter
import os
import configparser
from database import Database
# Construct the relative path to config.ini
config_path = os.path.realpath("../sleep-monitoring-ai-project/app/config/config.ini")
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)

# Variable to store the last alert time
last_alert_time = None
alert_counter = Counter()
alert = []
# Minimum time between alerts (10 seconds)
ALERT_INTERVAL = timedelta(seconds=config.getint("alert_system","timedelta"))

# Display alert by printing to the console
def show_alert(message):
    print(f"ALERT: {message}")

def alert_to_db(message):
    db = Database()
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.insert_alert_to_db(time_now, message)
    db.close_connection()
# Function to check if enough time has passed between alerts
def can_send_alert():
    global last_alert_time
    current_time = datetime.now()

    if last_alert_time is None or (current_time - last_alert_time) >= ALERT_INTERVAL:
        return True
    return False

# Main function to send alerts
alerts_count = 0  # Thêm biến này vào alert_system.py

# Main function to send alerts
def send_alert(message):
    global last_alert_time, alerts_count, alert
    if can_send_alert():
        show_alert(message)
        #alert_to_db(message)
        last_alert_time = datetime.now()
        alerts_count += 1  # Tăng tổng số lượng cảnh báo
        alert_counter[message] += 1  # Tăng số lượng cho loại cảnh báo này
        alert.append(message)


# Hàm hiển thị thống kê cảnh báo
def display_alert_statistics():
    """Hiển thị thống kê số lượng và loại cảnh báo."""
    print("Alert Statistics:")
    for alert_type, count in alert_counter.items():
        print(f"  {alert_type}: {count}")
    return alert_counter



def display_last_alert():
    if alert:
        return alert[-1]
    else:
        return None