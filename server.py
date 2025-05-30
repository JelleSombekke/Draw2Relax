from flask import Flask, request, send_file, send_from_directory, jsonify
from flask_cors import CORS
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import sys
from datetime import datetime
from ultralytics import YOLO
import receive_breathing_sensor_data
import logging

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from functions import make_circ_animation_frames

app = Flask(__name__)
CORS(app, origins=["https://jellesombekke.github.io"])

# Define a custom log filter
class IgnoreGetBreathingDataFilter(logging.Filter):
    def filter(self, record):
        # Ignore logs that contain 'GET /get-breathing-data'
        if "GET /get-breathing-data" in record.getMessage():
            return False  # Filter out these logs
        return True

# Set up the custom log filter
log = logging.getLogger('werkzeug')
log.addFilter(IgnoreGetBreathingDataFilter())

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Extract image data
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        # Prepare and save drawing
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        white_bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, pil_img)
        gray_img = composite.convert("L")
        img = np.array(gray_img)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # image saving
        os.makedirs('drawings', exist_ok=True)
        gray_img.save(f'drawings/drawing_{timestamp}.png')

        # Run your processing pipeline
        base64_frames, file_path_list= run_pipeline(img, timestamp)
        return jsonify({"frames": base64_frames})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear-frames', methods=['POST'])
def clear_frames():
    folder = os.path.join('static', 'guidance_flow_img')
    os.makedirs(folder, exist_ok=True)
    try:
        for file in os.listdir(folder):
            if file.endswith('.png'):
                os.remove(os.path.join(folder, file))
        return '', 204
    except Exception as e:
        print(f"Failed to clear frames: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/run-sensor', methods=['POST'])
def start_breathing_sensor():
    # Start the breathing sensor in a separate thread
    try:
        receive_breathing_sensor_data.start_sensor_thread()
        return jsonify({"status": "Sensor started"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-breathing-data', methods=['GET'])
def get_breathing_data():
    # Fetch the latest breathing data from the global variable
    value = receive_breathing_sensor_data.latest_breathing_value

    # Save breathing data
    # receive_breathing_sensor_data.save_breathing_value(value)

    return jsonify({"breathing_value": value})

@app.route('/stop-sensor', methods=['POST'])
def stop_breathing_sensor():
    # Stop the sensor thread
    receive_breathing_sensor_data.stop_sensor()
    return jsonify({"status": "Sensor stopped"}), 200

def run_pipeline(img, timestamp):
    start_N = 2000
    end_N = 500
    n_iterations = 100
    growth_constant = 15000

    # Load the model once (outside of your draw loop)
    model = YOLO("../trained_model/weights/best.pt")
    results = model.predict(source=f"drawings/drawing_{timestamp}.png", save=False, name="", exist_ok=True)
    padding = 5
    circular_structures = []

    for r in results:

        if r.boxes is not None:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                confidence = r.boxes.conf[i]
                #if confidence < 0.6:
                #    continue

                x1, y1, x2, y2 = box
                x1 -= padding
                y1 -= padding
                x2 += padding
                y2 += padding

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                rx = (x2 - x1) / 2
                ry = (y2 - y1) / 2

                circular_structures.append((cx, cy, rx, ry))

    base64_frames, file_path_list = make_circ_animation_frames(img, start_N, end_N, n_iterations, growth_constant, circular_structures, location='static/guidance_flow_img', mode=None)

    return base64_frames, file_path_list

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)