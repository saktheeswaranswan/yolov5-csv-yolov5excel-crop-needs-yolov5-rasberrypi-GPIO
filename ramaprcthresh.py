import torch
import numpy as np
import cv2
import csv
import time
import os
from datetime import datetime

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
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# CSV setup for saving detection results
csv_file = 'detection_results.csv'
csv_fields = ['timestamp', 'class', 'confidence', 'x', 'y', 'width', 'height']
csv_output = open(csv_file, 'w')
csv_writer = csv.DictWriter(csv_output, fieldnames=csv_fields)
csv_writer.writeheader()

# Folder setup for saving cropped detections
output_folder = 'detection_crops'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load class labels
class_labels = model.names

# Load classes and their numbers from CSV
class_mapping = {}
with open('class_mapping.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        class_mapping[row[0]] = int(row[1])

# Load thresholds from threshold.csv
thresholds = {}
with open('threshold.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        class_name = row[0]
        threshold = int(row[1])
        thresholds[class_name] = threshold

# Initialize variables for counting
counters = {class_name: 0 for class_name in class_mapping.keys()}

# Time variables
start_time = time.time()
previous_time = start_time

# Object detection loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - previous_time

    if elapsed_time < 0.1:
        continue

    # Perform object detection
    results = model(frame)

    # Get detection information
    detections = results.pandas().xyxy[0]

    # Filter detections for specified classes
    filtered_detections = detections[detections['name'].isin(class_mapping.keys())]

    # Check if any specified classes are detected
    if len(filtered_detections) > 0:
        for idx, detection in filtered_detections.iterrows():
            class_name = detection['name']
            class_number = class_mapping[class_name]

            # Check if the class number is 999
            if class_number == 999:
                # Save detection results in CSV and crop the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_writer.writerow({
                    'timestamp': timestamp,
                    'class': class_name,
                    'confidence': detection['confidence'],
                    'x': detection['xmin'],
                    'y': detection['ymin'],
                    'width': detection['xmax'] - detection['xmin'],
                    'height': detection['ymax'] - detection['ymin']
                })

                crop = frame[int(detection['ymin']):int(detection['ymax']),
                             int(detection['xmin']):int(detection['xmax'])]

                # Check if the counter exceeds the threshold
                if counters[class_name] >= thresholds[class_name]:
                    # Save the cropped image with timestamp and location
                    location = "bus_station_village_district"  # Replace with the actual location
                    image_name = f"{class_name}_{timestamp}_{location}.jpg"
                    cv2.imwrite(os.path.join(output_folder, image_name), crop)

                    # Reset the counter
                    counters[class_name] = 0

                # Increment the counter
                counters[class_name] += 1

    # Display the frame with bounding boxes and labels
    if len(results.pred) > 0:
        pred = results.pred[0]
        for det in pred:
            bbox = det[:4].int().cpu().numpy()  # Convert bbox to int values
            label = int(det[5])
            confidence = float(det[4])

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_labels[label]} {confidence:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Object Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

    previous_time = current_time

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
csv_output.close()

