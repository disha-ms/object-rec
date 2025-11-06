# object-rec
it detects the objects in live with its name and accuracy

Object Identification using YOLOv8 + OpenCV
Overview
This project performs real-time object detection using the YOLOv8 model and OpenCV.
It identifies and labels objects through the webcam feed.
The YOLOv8n model detects 80 common objects such as people, cars, laptops, etc.

Features
Detects multiple objects in real-time.
Displays bounding boxes with confidence levels.
Uses YOLOv8n for faster inference.
Press ‘q’ to quit detection window.
Easy to set up and run.

Requirements
Python 3.8 or above
Required libraries:
pip install ultralytics opencv-python

Project Files
object_identify.py → Main Python script
yolov8n.pt → Pre-trained YOLO model (downloaded automatically)
README.md → Documentation

How to Run
Clone this repository
git clone https://github.com/<your-username>/object-identification.git
cd object-identification
Install dependencies
pip install ultralytics opencv-python
Run the program
python object_identify.py
Press ‘q’ to exit the webcam window.

How It Works
Loads the YOLOv8n model.
Captures live frames from webcam.
Runs YOLO detection on each frame.
Draws bounding boxes and class labels on detected objects.
Displays the live annotated video feed.

Tech Stack
Python
OpenCV
Ultralytics YOLOv8
