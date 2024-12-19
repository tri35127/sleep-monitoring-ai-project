import configparser
import os
import mysql.connector

# Construct the relative path to config.ini
config_path = os.path.realpath("../sleep-monitoring-ai-project/app/config/config.ini")
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)


class Database:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=config.get('database', 'db_host'),
            user=config.get('database', 'db_user'),
            password=config.get('database', 'db_password'),
            database=config.get('database', 'db_name')
        )
        self.cursor = self.conn.cursor()


    def insert_alert_to_db(self, time_now, message):
        try:
            query = """
            INSERT INTO logs (Timestamp, events_type)
            VALUES (%s, %s)
            """
            self.cursor.execute(query, (time_now, str(message)))
            self.conn.commit()
            print(f"Alert logged to database: {message}")
        except mysql.connector.Error as e:
            print(f"Error logging alert to database: {e}")

    def export_alert_from_db(self, date):
        """
        Lấy dữ liệu Timestamp và events_type từ bảng 'logs' cho một ngày cụ thể.
        """
        try:
            query = """
            SELECT Timestamp, events_type 
            FROM logs 
            WHERE DATE(Timestamp) = %s
            ORDER BY `Timestamp` DESC
            """
            self.cursor.execute(query, (date,))
            results = self.cursor.fetchall()

            # Format kết quả thành danh sách các cặp thời gian và sự kiện
            alerts = [{"Timestamp": str(row[0])[-6:], "events_type": row[1]} for row in results]
            print(f"Alerts exported successfully for date: {date}")
            return alerts
        except mysql.connector.Error as e:
            print(f"Error exporting alerts from database: {e}")
            return []
        
    def close_connection(self):
        if self.conn.is_connected():
            self.cursor.close()
            self.conn.close()