import torch
import cv2
import csv
import time
import os
import RPi.GPIO as GPIO

# GPIO setup for LED
LED_PIN = 18  # GPIO pin number
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Load the YOLOv5 model
weights = "yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)  # Use webcam (change the index if you have multiple cameras)

# CSV setup for saving detection results
csv_file = 'detection_results.csv'
csv_fields = ['timestamp', 'class', 'confidence', 'x', 'y', 'width', 'height']

# Folder setup for saving cropped detections
output_folder = 'detection_crops'
os.makedirs(output_folder, exist_ok=True)

# Initialize CSV file
with open(csv_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_fields)

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
    
    cat_detected = False
    person_detected = False
    
    # Save detection results in CSV and crop images
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        for _, detection in detections.iterrows():
            class_label = detection['name']
            confidence = detection['confidence']
            x = detection['xmin']
            y = detection['ymin']
            width = detection['xmax'] - detection['xmin']
            height = detection['ymax'] - detection['ymin']
            
            # Write to CSV
            csv_writer.writerow([timestamp, class_label, confidence, x, y, width, height])
            
            if class_label == 'cat':
                cat_detected = True
            elif class_label == 'person':
                person_detected = True
                # Crop person image
                crop = frame[int(y):int(y + height), int(x):int(x + width)]
                cv2.imwrite(f"{output_folder}/person_{timestamp}.jpg", crop)
    
    if person_detected and not cat_detected:
        # Turn on LED for 10 seconds
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(10)
        GPIO.output(LED_PIN, GPIO.LOW)
    
    # Display the frame
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()  # Cleanup GPIO pins

