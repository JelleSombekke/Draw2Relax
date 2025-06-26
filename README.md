# Draw2Relax

## Description

Draw2Relax is an interactive WebUI application that integrates user-drawn sketches with real-time breathing sensor input. It uses machine learning and computer vision to process user's drawings, enabling a unique, meditative experience that links creativity and relaxation. 

This application must be run locally, as GitHub Pages does not support the required back-end processing. For best results, set up a virtual environment, install the required dependencies, and run the application on your local machine.

### Document Descriptions
### Code
- index.html: Contains the front-end HTML code for the WebUI. It is viewable on [Draw2Relax](https://jellesombekke.github.io/Draw2Relax/). Note: This page only includes front-end logic and does not support any back-end functionality.
- server.py: The primary back-end script powering the WebUI.
- functions.py:  Includes core pipeline functions such as contour calculation and manipulation, contour refilling, and computing displacement fields for circular structures.
- receive_breathing_sensor_data.py: Handles the reception, smoothing, and storage of breathing sensor data.

### Files
- requirements.txt: Lists all required packages to run both the front-end and back-end components of the WebUI.
- trained_model: Contains a model trained to detect various circular structures (circle, swirl, blob, oval, and half-circle). The model is based on Ultralytics’ YOLOv8n-seg architecture.

### WebUI
The WebUI contains the following buttons:
- Clear: Removes all drawn strokes.
- Undo: Removes the last drawn stroke.
- Submit: Submits the drawing for processing and starts running pipeline with breathing sensor (when submitting on non laptop device the drawing is sent to the laptop it is connected to).
- Simulate: Submits the drawing for processing and simulates a breathing wave. (Button only exists on laptops)
- Connect device: Pops up a QR-code so devices could connect to the laptops IP and submit drawings from the device towards the laptop. (Button only exists on laptops)
- Palette: Blues: Selects the color palette for the pipeline (default: Blues). (Button only exists on laptops)
- Click Draw Mode: OFF: Changes drawing logic to click to start and stop drawing (default: OFF). (Button only exists on laptops)
- Fullscreen: OFF: Toggles fullscreen (default: OFF).
- Breathing Sensor Inactive: This displays if the breathing sensor is active or not (default: Inactive). (Unclickable, Button only exists on laptops)

<img width="1512" alt="webUI" src="https://github.com/user-attachments/assets/0a4d6971-6126-4353-be78-b097eb4daf15" />

## Running the System

To run Draw2Relax on your local machine, follow these steps:

Clone the Repository
<pre>git clone https://github.com/JelleSombekke/Draw2Relax.git
cd Draw2Relax</pre>

Create a Virtual Environment (Recommended)
<pre>python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate</pre>


Install Dependencies
<pre>pip install -r requirements.txt</pre>

Run the Backend Server
<pre>python server.py</pre>

Using the Breathing Sensor (Optional)
- Connect the breathing sensor via a USB port. (Currently only compatible with the HKH-11C respiratory wave sensor)
- Check which COM port the sensor is connected to, and update the script if needed. (Default is set to COM3 in receive_breathing_sensor_data.py)

## Author
* Jelle Sombekke - jellesombekke@gmail.com
