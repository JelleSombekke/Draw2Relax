# Draw2Relax

## Description

Draw2Relax is an interactive WebUI application that integrates user-drawn sketches with real-time breathing sensor input. It uses machine learning to detect and process circular structures from drawings, enabling a unique, meditative experience that links creativity and relaxation. 

This application must be run locally, as GitHub Pages does not support the required back-end processing. For best results, set up a virtual environment, install the required dependencies, and run the application on your local machine.

### Document Descriptions
### Code
- index.html: Contains the front-end HTML code for the WebUI. It is viewable on [Draw2Relax](https://jellesombekke.github.io/Draw2Relax/). Note: This file only includes front-end logic and does not support any back-end functionality.
- server.py: The primary back-end script powering the WebUI.
- functions.py:  Includes core pipeline functions such as contour calculation and manipulation, contour refilling, and computing displacement fields for circular structures.
- receive_breathing_sensor_data.py: Handles the reception, smoothing, and storage of breathing sensor data.

### Files
- requirements.txt: Lists all required packages to run both the front-end and back-end components of the WebUI.
- trained_model: Contains a model trained to detect various circular structures (circle, swirl, blob, oval, and half-circle). The model is based on Ultralytics’ YOLOv8n-seg architecture.

### WebUI
The WebUI contains the following buttons/:
- Clear: Removes all drawn strokes.
- Undo: Removes the last drawn stroke.
- Submit: Submits the drawing for processing and starts running pipeline with breathing sensor (when submitting on non laptop device the drawing is sent to the laptop it is connected to).
- Simulate: Submits the drawing for processing and simulates a breathing wave. (Button only exists on laptops)
- Connect device: Pops up a QR-code so devices could connect to the laptops IP and submit drawings from the device towards the laptop. (Button only exists on laptops)
- Palette: Blues: Selects the color palette for the pipeline (default: Blues). (Button only exists on laptops)
- Mousepad Mode: OFF: Changes drawing logic to click to start and stop drawing (default: OFF). (Button only exists on laptops)
- Fullscreen: OFF: Toggles fullscreen (default: OFF).
- Breathing Sensor Inactive: This displays if the breathing sensor is active or not (default: Inactive). (Unclickable, Button only exists on laptops)

<img width="1512" alt="webUI" src="https://github.com/user-attachments/assets/633f5229-f878-4846-8780-c0a6fb86b222" />


## Author
* Jelle Sombekke - jellesombekke@gmail.com
