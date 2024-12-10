# Application of Al technology for monitoring preschool children’s naps and alerting teachers of anomalies
A comprehensive AI-powered system for monitoring sleep patterns, detecting anomalies, and analyzing movements during sleep. Designed for real-time or recorded video input, this project uses computer vision and deep learning techniques to deliver accurate and actionable insights.

## Table of Contents

- [Features](#features)
- [Quick Guide](#quick-guide)
    - [Pre-requisites](#pre-requisites)
    - [Installation](#installation)
    - [Running the Application](#running-the-application)
    - [API Usage](#api-usage)
- [Default Database Information ](#default-database-information)
- [Database Tables](#database-tables)
    - [Events Logs Table](#events-logs-table)
- [Model - Library](#model---library)
    - [`Yolo v11` and `Yolo v11 pose` model](#Yolo-v11-and-yolo-v11-pose-model)
- [Endpoint Usage](#endpoint-usage)
- [Config](#config)

## Features

- **Video Analysis**: Supports live monitoring via camera streams or processing of pre-recorded videos.
- **Customizable Monitoring Zones**: Define and adjust bed areas for focused analysis.
- **Posture and Movement Detection**: Tracks sleep positions and identifies significant movements.
- **Configurable Input Settings**: Easily modify resolution, input source, and other parameters.
- **Detailed Logs**: Provides logs of events for post-analysis.
## Quick Guide

### Pre-requisites
#### System Requirements
- **Operating System**: Windows 10 or later, MacOS Ventura or later 
- **Python Version**: 3.10 or later
- **Libraries**: Listed in `requirements.txt`
- **Hardware**:
  - Minimum: 
    - CPU: Intel® Core™ i5-8500 / Apple M1
    - GPU: NVIDIA GeForce GTX 1650 / Apple M1
    - RAM: 8GB
  - Recommended:
    - CPU: Intel® Core™ i5-11400 / Apple M1 Pro
    - GPU: NVIDIA GeForce GTX 3050 / Apple M2 Pro
    - RAM: 16GB
### Installation

1. Clone the project from the repository:

    ```bash
    git clone -b hosting https://github.com/tri35127/sleep-monitoring-ai-project.git
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
4. Create a database with name is 'events_logs'

5. Set up the database by run SQL script:
    ```
   database/database.sql
    ```

### Running the Application
- Please replace `path_to_the_project` with your actual path
Start the application using the provided run script:

    ```bash
    cd path_to_the_project/sleep-monitoring-ai-project
    python app/main/main.py
    ```

3. The application will start, and you can now access the API at http://localhost:8000 or http://127.0.0.1:8000.

### API Usage

- Use tools like `curl`, `Postman`, or your preferred HTTP client to send requests to the API endpoints.
  See the [Endpoint Usage](#endpoint-usage) for more information



## Default Database Information

- **Database Host**: localhost or 127.0.0.1
- **Database Port**: 8001
- **Database Name**: events_logs
- **Database User**: root
- **Database Password**: No password required

## Database Tables

### Events Logs Table

- **Fields**:
    - `Timestamp` datetime DEFAULT NULL,
    - `events_type` varchar(50) DEFAULT NULL
- **Description**: Stores event with time after request.


## Model - Library
### `Yolo v11` and `Yolo v11 pose` model
- **Model name**: YOLO
- **Version**: v11
- **Install command line**:
    ```bash
    pip install ultralytics
    ```
- **import syntax**:
    ```python lines
    from ultralytics import YOLO
    ```
- **Project documentation**: https://docs.ultralytics.com/


## Endpoint Usage

### Video streaming

- To view processed video stream, make a `GET` request to `{baseurl}/checkcam/source`.
- You can configure the AI threshold according to your needs in the `config.ini` file.

### Reset beds

- To set/reset beds area based on the child's position, make a `POST` request to `{baseurl}/checkcam/resetbeds`.
- The results will notify about the bed zone change success or failure

### View stats
- To get the unusual notification , make a `GET` request to `{baseurl}/viewstats`.
- The results will notify about the alert of anomalies

### View all
- To get all the unusual notification and counting of them , make a `GET` request to `{baseurl}/viewall`.
- The results will show a list of alert of anomalies and frequency of each type of notification

## Config
```ini
[camera]
camera_id = 0               # The ID of the camera to use (usually 0 if only one camera is connected)
width = 1920                # The width of the camera frame
height = 1080               # The height of the camera frame

[database]
db_host = localhost          # The hostname or address of the database server
db_user = root               # The username for database access
db_password =                # The password for database access (left blank here)
db_name = events_logs        # The name of the database used to store event logs

[person_detection]
yolo_model_detection_path = ../sleep-monitoring-ai-project/data/yolo11l.pt  # Path to the YOLO model for person detection
bed_scale_factor = 1.15              # Scale factor for enlarging the bed area bounding box
is_person_outside_bed_threshold = 0.4  # Threshold for detecting if a person is outside the bed (higher = more sensitive)
is_sitting_overlap_threshold = 0.45   # Overlap ratio threshold for detecting if a person is sitting
is_sitting_aspect_ratio_threshold = 0.6  # Aspect ratio threshold for detecting if a person is sitting

[keypoint]
yolo_model_pose_path = ../sleep-monitoring-ai-project/data/yolo11m-pose.pt  # Path to the YOLO model for pose estimation
frame_to_analyze_sleep_movement = 10       # Number of frames to analyze for sleep movement
max_standard_deviation_velocity = 3.0      # Maximum allowable standard deviation of velocity for keypoints
max_velocity_of_one_keypoint = 20          # Maximum allowable velocity for a single keypoint
max_velocity = 5                           # Maximum velocity threshold for normal movement
number_of_frame = 5                        # Number of frames used for analysis
max_sustained_spike = 2                    # Maximum number of sustained spikes indicating restlessness
max_movement_cluster = 2                   # Maximum number of movement clusters allowed
max_movement_count = 10                    # Maximum number of movements allowed before being flagged

[alert_system]
timedelta = 10                                              # Time interval for triggering alerts in seconds
is_sitting_alert = Canh bao tre dang ngoi!                  # Alert message when the child is sitting
is_person_outside_bed_alert = Canh bao tre roi khoi giuong! # Alert message for a child outside the bed
is_face_covered_alert = Canh bao tre bi che mat!            # Alert message for covered face
poor_sleep_movement_alert = Tre ngu khong ngon!             # Alert message for poor sleep movements
```