import configparser
import os
import sys
import mysql.connector

# Construct the relative path to config.ini
config_path = os.path.realpath("../config.ini")
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)


class Database:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="events_logs"
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

    def close_connection(self):
        if self.conn.is_connected():
            self.cursor.close()
            self.conn.close()