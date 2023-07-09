import torch
import numpy as np
import cv2
import csv
import time
from torchvision.ops.boxes import nms
import RPi.GPIO as GPIO

# Set up GPIO pins
object_pins = {
    'cat': 17,
    'person': 18,
    'dog': 19,
    'car': 20,
    'bus': 21
}
GPIO.setmode(GPIO.BCM)
for pin in object_pins.values():
    GPIO.setup(pin, GPIO.OUT)

# Load the YOLOv5 model
weights = "yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

# Load the dataset configuration
data = "data/coco128.yaml"
model.yaml = data

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)  # Use webcam (change the index if you have multiple cameras)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# CSV setup for saving detection results
csv_file = 'detection_results.csv'
csv_fields = ['timestamp', 'class', 'confidence', 'x', 'y', 'width', 'height']
csv_output = open(csv_file, 'w')
csv_writer = csv.DictWriter(csv_output, fieldnames=csv_fields)
csv_writer.writeheader()

# Folder setup for saving cropped detections
output_folder = 'detection_crops'

# Load class labels
class_labels = model.names

# Initialize variables
object_detected = {}
object_detection_time = {}

# Object detection loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get detection information
    detections = results.pandas().xyxy[0]

    # Filter out bench detections
    for object_label, object_pin in object_pins.items():
        object_detections = detections[detections['name'] == object_label]

        if len(object_detections) > 0:
            # Apply non-maximum suppression (NMS)
            boxes = object_detections[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
            scores = object_detections['confidence'].values.astype(np.float32)
            keep_indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.5)
            keep_indices = keep_indices.cpu().numpy().astype(np.int32)  # Convert to NumPy array of integers

            # Get the timestamp
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # Save detection results in CSV and crop images
            for idx in keep_indices:
                detection = object_detections.iloc[idx]
                class_label = detection['name']
                confidence = detection['confidence']
                x = detection['xmin']
                y = detection['ymin']
                width = detection['xmax'] - detection['xmin']
                height = detection['ymax'] - detection['ymin']

                # Write to CSV
                csv_writer.writerow({
                    'timestamp': current_time,
                    'class': class_label,
                    'confidence': confidence,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })

                # Crop image
                crop = frame[int(y):int(y + height), int(x):int(x + width)]
                cv2.imwrite(f"{output_folder}/{class_label}_{current_time}.jpg", crop)

                # Store object detection and time
                object_detected[object_label] = True
                object_detection_time[object_label] = time.time()

    # Check if the object detection time has exceeded 12 seconds
    for object_label in object_detected.keys():
        if object_label in object_pins:
            pin = object_pins[object_label]
            if object_detected[object_label] and (time.time() - object_detection_time[object_label]) >= 12:
                # Turn off the corresponding LED
                GPIO.output(pin, GPIO.LOW)
                object_detected[object_label] = False

    # Show the frame with detections
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(100) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
csv_output.close()
GPIO.cleanup()
