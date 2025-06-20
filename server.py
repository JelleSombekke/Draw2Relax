from flask import Flask, request, send_file, send_from_directory, jsonify, Response
import socket
import qrcode
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import sys
from datetime import datetime
from ultralytics import YOLO
import breathing_sensor_handling
from functions import make_circ_animation_frames
import logging

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

latest_image_data = None
new_drawing_available = False
drawing_meta = {"width": 0, "height": 0}

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Define a custom log filter
class IgnoreGetBreathingDataFilter(logging.Filter):
    def filter(self, record):
        # Ignore certain logs
        if "GET /get-breathing-data" in record.getMessage():
            return False
        elif "GET /sample-breathing-data" in record.getMessage():
            return False
        elif "GET /new-drawing" in record.getMessage():
            return False
        return True

# Set up the custom log filter
log = logging.getLogger('werkzeug')
log.addFilter(IgnoreGetBreathingDataFilter())

@app.route('/ip')
def get_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return jsonify({"ip": local_ip})

@app.route('/qrcode')
def generate_qr():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    ws_url = f"http://{local_ip}:5000"  # your WebSocket URL or backend URL

    img = qrcode.make(ws_url)
    buf = BytesIO()
    img.save(buf)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/upload', methods=['POST'])
def upload():
    global latest_image_data, drawing_meta, new_drawing_available
    data = request.json
    img_data = data['image'].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    latest_image_data = img
    drawing_meta['width'], drawing_meta['height'] = img.size
    new_drawing_available = True
    return jsonify({"status": "success"})

@app.route('/draw')
def get_draw():
    global latest_image_data
    if latest_image_data:
        img_io = BytesIO()
        latest_image_data.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    else:
        return '', 404

@app.route('/drawing-meta')
def get_meta():
    global drawing_meta
    return jsonify(drawing_meta)

@app.route('/new-drawing')
def new_drawing():
    global new_drawing_available
    flag = new_drawing_available
    new_drawing_available = False
    return jsonify({"new_drawing": flag})

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
        base64_frames, file_path_list = run_pipeline(img, timestamp)
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
        breathing_sensor_handling.start_sensor_thread()
        return jsonify({"status": "Sensor started"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-breathing-data', methods=['GET'])
def get_breathing_data():
    # Fetch the latest breathing data from the global variable
    value = breathing_sensor_handling.latest_breathing_value
    unsmoothed_value = breathing_sensor_handling.latest_breathing_value_unsmoothed
    breathing_sensor_handling.save_breathing_value(value, unsmoothed_value)
    return jsonify({"breathing_value": value, "breathing_value_unsmoothed": unsmoothed_value})

@app.route('/sample-breathing-data', methods=['GET'])
def sample_breathing_data():
    # Fetch the latest breathing data from the global variable
    value = breathing_sensor_handling.latest_breathing_value
    unsmoothed_value = breathing_sensor_handling.latest_breathing_value_unsmoothed
    return jsonify({"breathing_value": value, "breathing_value_unsmoothed": unsmoothed_value})


@app.route('/stop-sensor', methods=['POST'])
def stop_breathing_sensor():
    # Stop the sensor thread
    breathing_sensor_handling.stop_sensor()
    return jsonify({"status": "Sensor stopped"}), 200

def run_pipeline(img, timestamp):
    start_N = 2000
    end_N = 500
    n_iterations = 100
    growth_constant = 15000

    # Load the model once (outside of your draw loop)
    model = YOLO("../circle_detection_model/runs/segment/train3/weights/best.pt")
    results = model.predict(source=f"drawings/drawing_{timestamp}.png", save=False, project="../circle_detection_model/runs/segment", name="", exist_ok=True)
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
    host_ip = socket.gethostbyname(socket.gethostname())
    app.run(host=host_ip, port=5000, debug=False, threaded=True)