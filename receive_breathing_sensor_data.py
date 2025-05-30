import threading
import serial
import time
import csv
from datetime import datetime
import os
from collections import deque

class SmoothingFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
    
    def apply(self, new_value):
        self.data_window.append(new_value)
        smoothed_value = sum(self.data_window) / len(self.data_window)
        return smoothed_value

# Create an instance of the smoothing filter with a window size of 5
smoothing_filter = SmoothingFilter(window_size=20)

# Global variable to store the latest breathing value
latest_breathing_value = 0
stop_event = threading.Event()

log_file_path = "breathing_log.csv"
sensor_thread = None

def save_breathing_value(value):
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value"])
    timestamp = time.time()
    with open(log_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, value])

def receive_breathing_data():
    global latest_breathing_value

    try:
        ser = serial.Serial('COM3', 9600, timeout=0.001)
        ser.write(bytes([0x20, 0x32]))
        time.sleep(0.1)

        while not stop_event.is_set():
            byte = ser.read(1)
            if byte:
                decimal_value = int(byte.hex(), 16)
                latest_breathing_value = round(smoothing_filter.apply(decimal_value))  # Update the global variable
    finally:
        ser.close()

def start_sensor_thread():
    global sensor_thread
    stop_event.clear()
    sensor_thread = threading.Thread(target=receive_breathing_data)
    sensor_thread.daemon = True
    sensor_thread.start()

    with open(log_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "value"])

def stop_sensor():
    stop_event.set()
    if sensor_thread:
        sensor_thread.join()

