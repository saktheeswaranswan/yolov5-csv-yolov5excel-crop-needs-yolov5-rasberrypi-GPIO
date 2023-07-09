#Please note that this modified code assumes you have connected the LEDs to GPIO pins on your hardware and installed the necessary dependencies. You may need to adjust the GPIO pin numbers based on your setup. Additionally, make sure you have the gpiozero library installed (you can install it using pip install gpiozero).
#Please note that this modified code assumes you have connected the LEDs to GPIO pins on your hardware and installed the necessary dependencies. You may need to adjust the GPIO pin numbers based on your setup. Additionally, make sure you have the gpiozero library installed (you can install it using pip install gpiozero).

import torch
import numpy as np
import cv2
import csv
import time
from torchvision.ops.boxes import nms
from gpiozero import LED

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

# LED setup
led_cat = LED(17)  # GPIO pin number for cat LED
led_dog = LED(18)  # GPIO pin number for dog LED
led_bus = LED(19)  # GPIO pin number for bus LED
led_car = LED(20)  # GPIO pin number for car LED
led_person = LED(21)  # GPIO pin number for person LED

# Object detection loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get detection information
    detections = results.pandas().xyxy[0]

    # Get the timestamp
    timestamp = time.time()

    # Detect and process each class individually
    for class_label in ['cat', 'dog', 'bus', 'car', 'person']:
        class_indices = detections[detections['name'] == class_label].index

        if len(class_indices) > 0:
            # Turn on the LED for the current class
            if class_label == 'cat':
                led_cat.on()
            elif class_label == 'dog':
                led_dog.on()
            elif class_label == 'bus':
                led_bus.on()
            elif class_label == 'car':
                led_car.on()
            elif class_label == 'person':
                led_person.on()

            # Save detection results in CSV and crop images
            for idx in class_indices:
                detection = detections.loc[idx]
                confidence = detection['confidence']
                x = detection['xmin']
                y = detection['ymin']
                width = detection['xmax'] - detection['xmin']
                height = detection['ymax'] - detection['ymin']

                # Write to CSV
                csv_writer.writerow({
                    'timestamp': timestamp,
                    'class': class_label,
                    'confidence': confidence,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })

                # Crop image
                crop = frame[int(y):int(y + height), int(x):int(x + width)]
                cv2.imwrite(f"{output_folder}/{class_label}_{timestamp}.jpg", crop)

            # Wait for 10 seconds
            time.sleep(10)

            # Turn off the LED after 10 seconds
            if class_label == 'cat':
                led_cat.off()
            elif class_label == 'dog':
                led_dog.off()
            elif class_label == 'bus':
                led_bus.off()
            elif class_label == 'car':
                led_car.off()
            elif class_label == 'person':
                led_person.off()

    # Display the frame with bounding boxes and labels
    if len(results.pred) > 0:
        pred = results.pred[0]
        for det in pred:
            bbox = det[:4].int().cpu().numpy()  # Convert bbox to int values
            label = int(det[5])
            confidence = det[4]
            class_label = class_labels[label]

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
csv_output.close()
