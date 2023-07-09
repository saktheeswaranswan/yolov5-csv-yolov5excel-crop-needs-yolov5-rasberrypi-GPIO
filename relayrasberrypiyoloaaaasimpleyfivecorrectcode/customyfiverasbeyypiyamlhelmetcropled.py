import RPi.GPIO as GPIO
import time
import torch
import cv2
import csv
import datetime
import os
import torchvision.ops as ops
import numpy as np

# GPIO pin numbers for the LEDs
RED_LED_PIN = 17
GREEN_LED_PIN = 18

# Initialize GPIO settings
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)

# Load the YOLOv5 model with custom weights
weights = "yolov5s.pt"  # Replace with the path to your custom weights file
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)  # Use webcam (change the index if you have multiple cameras)

# CSV setup for saving detection results
csv_file = 'detection_results.csv'
csv_fields = ['timestamp', 'class', 'confidence']

# Initialize CSV file
with open(csv_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_fields)

# Create folder for cropped images
folder_name = 'cropped_images'
os.makedirs(folder_name, exist_ok=True)

# Object detection loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Get detection information
    detections = results.xyxy[0].cpu()

    # Filter detections based on selected classes
    cat_detection = None
    person_detection = None
    for detection in detections:
        class_label = model.module.names[int(detection[5])] if hasattr(model, 'module') else model.names[int(detection[5])]

        if class_label == 'cat':
            cat_detection = detection
        elif class_label == 'person':
            person_detection = detection

    # Get the current system date and time
    current_time = datetime.datetime.now().strftime("%b %d %H:%M %p")

    # Save detection results in CSV with current time
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        if cat_detection:
            class_label = 'cat'
            confidence = cat_detection[4]

            # Write to CSV with current time
            csv_writer.writerow([current_time, class_label, confidence])

            # Crop and save the detected cat image
            xmin, ymin, xmax, ymax = map(int, cat_detection[:4])
            cat_image = frame[ymin:ymax, xmin:xmax]
            cat_image_filename = f'{class_label}_{confidence:.2f}_{current_time.replace(":", "").replace(" ", "_")}.jpg'
            cat_image_path = os.path.join(folder_name, cat_image_filename)
            cv2.imwrite(cat_image_path, cat_image)

        if person_detection:
            class_label = 'person'
            confidence = person_detection[4]

            # Write to CSV with current time
            csv_writer.writerow([current_time, class_label, confidence])

            # Crop and save the detected person image
            xmin, ymin, xmax, ymax = map(int, person_detection[:4])
            person_image = frame[ymin:ymax, xmin:xmax]
            person_image_filename = f'{class_label}_{confidence:.2f}_{current_time.replace(":", "").replace(" ", "_")}.jpg'
            person_image_path = os.path.join(folder_name, person_image_filename)
            cv2.imwrite(person_image_path, person_image)

            # Control LEDs based on detected classes
            GPIO.output(RED_LED_PIN, GPIO.LOW)
            GPIO.output(GREEN_LED_PIN, GPIO.HIGH)

    # Check if the cat is missing
    if cat_detection is None and person_detection is not None:
        class_label = 'person'
        confidence = person_detection[4]

        # Write to CSV with current time
        with open(csv_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([current_time, class_label, confidence])

        # Crop and save the detected person image
        xmin, ymin, xmax, ymax = map(int, person_detection[:4])
        person_image = frame[ymin:ymax, xmin:xmax]
        person_image_filename = f'{class_label}_{confidence:.2f}_{current_time.replace(":", "").replace(" ", "_")}.jpg'
        person_image_path = os.path.join(folder_name, person_image_filename)
        cv2.imwrite(person_image_path, person_image)

        # Control LEDs based on detected classes
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        GPIO.output(GREEN_LED_PIN, GPIO.LOW)

    # Display the frame with bounding boxes and labels
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()  # Clean up the GPIO pins

