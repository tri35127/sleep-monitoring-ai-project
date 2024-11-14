from datetime import datetime, timedelta

# Variable to store the last alert time
last_alert_time = None

# Minimum time between alerts (10 seconds)
ALERT_INTERVAL = timedelta(seconds=10)

# Display alert by printing to the console
def show_alert(message):
    print(f"ALERT: {message}")

# Log alert to a file
def log_alert(message, log_file="alert_log.txt"):
    with open(log_file, "a", encoding="utf-8") as f:
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{time_now} - {message}\n")
    print(f"Logged alert: {message}")

# Function to check if enough time has passed between alerts
def can_send_alert():
    global last_alert_time
    current_time = datetime.now()

    if last_alert_time is None or (current_time - last_alert_time) >= ALERT_INTERVAL:
        return True
    return False

# Main function to send alerts
def send_alert(message):
    global last_alert_time  # Use global variable to update the time
    # Only send alert if enough time has passed (10 seconds)
    if can_send_alert():
        # Print the message to the console
        show_alert(message)

        # Log each alert
        log_alert(message)

        # Update the last alert time
        last_alert_time = datetime.now()
    # else:  # No need for an explicit 'else' if you're just passing
    #     pass
