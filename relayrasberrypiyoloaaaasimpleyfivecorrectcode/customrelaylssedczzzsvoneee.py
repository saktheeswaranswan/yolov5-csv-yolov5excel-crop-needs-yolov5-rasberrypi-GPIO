import torch
import cv2
import csv
import datetime
import os
import RPi.GPIO as GPIO
import torchvision.ops as ops
import numpy as np

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

# Specify the desired classes to detect and crop
classes_to_detect = ['person', 'cat', 'dog', 'car', 'bus']

# Create folder for cropped images
folder_name = 'cropped_images'
os.makedirs(folder_name, exist_ok=True)

# Raspberry Pi GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# LED pin setup
led_pins = {
    'person': 18,
    'cat': 23,
    'dog': 24,
    'car': 25,
    'bus': 12
}

# Initialize LED pins
for pin in led_pins.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Function to glow LED based on class
def glow_led(class_label):
    if class_label in led_pins:
        pin = led_pins[class_label]
        GPIO.output(pin, GPIO.HIGH)

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
    detected_classes = set()
    for detection in detections:
        class_label = model.module.names[int(detection[5])] if hasattr(model, 'module') else model.names[int(detection[5])]
        if class_label in classes_to_detect:
            detected_classes.add(class_label)
            glow_led(class_label)

    # Check if all classes are detected in the frame
    if set(classes_to_detect).issubset(detected_classes):
        # Sleep for 10 seconds
        cv2.waitKey(10000)
        # Turn off all LEDs
        for pin in led_pins.values():
            GPIO.output(pin, GPIO.LOW)

    # Get the current system date and time
    current_time = datetime.datetime.now().strftime("%b %d %H:%M %p")

    # Save detection results in CSV with current time
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        for detection in detections:
            class_label = model.module.names[int(detection[5])] if hasattr(model, 'module') else model.names[int(detection[5])]
            confidence = detection[4]

            # Write to CSV with current time
            csv_writer.writerow([current_time, class_label, confidence])

            # Crop and save the detected object image
            xmin, ymin, xmax, ymax = map(int, detection[:4])
            object_image = frame[ymin:ymax, xmin:xmax]
            object_image_filename = f'{class_label}_{confidence:.2f}_{current_time.replace(":", "").replace(" ", "_")}.jpg'
            object_image_path = os.path.join(folder_name, object_image_filename)
            cv2.imwrite(object_image_path, object_image)

            # Draw bounding boxes
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_label}: {confidence:.2f}', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()

