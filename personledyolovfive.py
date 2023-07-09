import torch
import numpy as np
import cv2
import csv
import time
from torchvision.ops.boxes import nms
import RPi.GPIO as GPIO

# Set up GPIO pins
red_led_pin = 17
green_led_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(red_led_pin, GPIO.OUT)
GPIO.setup(green_led_pin, GPIO.OUT)

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
person_detected = False
person_detection_time = 0

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
    person_detections = detections[detections['name'] != 'bench']
    cat_detections = detections[detections['name'] == 'cat']

    if len(person_detections) > 0:
        # Apply non-maximum suppression (NMS)
        boxes = person_detections[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
        scores = person_detections['confidence'].values.astype(np.float32)
        keep_indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.5)
        keep_indices = keep_indices.cpu().numpy().astype(np.int32)  # Convert to NumPy array of integers

        # Get the timestamp
        timestamp = time.strftime("%I:%M %p")

        # Save detection results in CSV and crop images
        for idx in keep_indices:
            detection = person_detections.iloc[idx]
            class_label = detection['name']
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

            # Turn on red LED
            GPIO.output(red_led_pin, GPIO.HIGH)
            GPIO.output(green_led_pin, GPIO.LOW)

            # Display the frame with bounding boxes and labels
            cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (255, 0, 0), 2)
            cv2.putText(frame, f'{class_label}: {confidence:.2f}', (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Update person detection time
            if not person_detected:
                person_detected = True
                person_detection_time = time.time()
    
    elif len(cat_detections) > 0:
        # Apply non-maximum suppression (NMS)
        boxes = cat_detections[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
        scores = cat_detections['confidence'].values.astype(np.float32)
        keep_indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.5)
        keep_indices = keep_indices.cpu().numpy().astype(np.int32)  # Convert to NumPy array of integers

        # Get the timestamp
        timestamp = time.strftime("%I:%M %p")

        # Save detection results in CSV and crop images
        for idx in keep_indices:
            detection = cat_detections.iloc[idx]
            class_label = detection['name']
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

            # Turn on green LED
            GPIO.output(red_led_pin, GPIO.LOW)
            GPIO.output(green_led_pin, GPIO.HIGH)

            # Display the frame with bounding boxes and labels
            cv2.rectangle(frame, (int(x), int(y)), (int(x + width), int(y + height)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_label}: {confidence:.2f}', (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    else:
        # Turn off both LEDs if no detections
        GPIO.output(red_led_pin, GPIO.LOW)
        GPIO.output(green_led_pin, GPIO.LOW)
        
        # Reset person detection flag and time
        person_detected = False
        person_detection_time = 0
    
    # Check if person detected and elapsed time is greater than 12 seconds
    if person_detected and (time.time() - person_detection_time) >= 12:
        # Turn off red LED
        GPIO.output(red_led_pin, GPIO.LOW)

    # Show the frame with detections
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(100) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
csv_output.close()
GPIO.cleanup()
