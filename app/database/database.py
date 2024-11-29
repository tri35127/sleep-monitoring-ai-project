import configparser
import os

import mysql.connector

# Construct the relative path to config.ini
config_path = os.path.realpath("../config/config.ini")
# Create a configuration object
config = configparser.ConfigParser()
config.read(config_path)


def connect_to_phpmyadmin():
    try:
        # Update these parameters if your phpMyAdmin uses non-default settings
        connection = mysql.connector.connect(
            host="localhost",     # Default phpMyAdmin server host
            user="root",          # Default username for phpMyAdmin
            password="",          # Default password (empty for XAMPP/MAMP default setup)
            database="events_logs"  # Replace with your database name
        )
        if connection.is_connected():
            print("Successfully connected to the database.")
            return connection
    except mysql.connector.Error as e:
        print(f"Error connecting to database: {e}")
        return None
connect_to_phpmyadmin

# Insert alert into the database
def insert_alert_to_db(connection, time_now, message):
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO logs (Timestamp, events_types)
        VALUES (%s, %s)
        """
        cursor.execute(query, (time_now, message))
        connection.commit()
        print(f"Alert logged to database: {message}")
    except mysql.connector.Error as e:
        print(f"Error logging alert to database: {e}")

