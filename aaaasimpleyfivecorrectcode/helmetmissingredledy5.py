import torch
import cv2
import csv
import time
import datetime
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
csv_fields = ['timestamp', 'class', 'confidence']

# Folder setup for saving cropped detections
output_folder = 'detection_crops'
os.makedirs(output_folder, exist_ok=True)

# Initialize CSV file
with open(csv_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_fields)

# Object detection loop
start_time = time.time()
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)

    # Get detection information
    detections = results.pandas().xyxy[0]

    # Get the current computer time
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    
    # Save detection results in CSV and crop images
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        cat_detected = False
        person_detected = False
        for _, detection in detections.iterrows():
            class_label = detection['name']
            confidence = detection['confidence']
            x = detection['xmin']
            y = detection['ymin']
            width = detection['xmax'] - detection['xmin']
            height = detection['ymax'] - detection['ymin']
            
            # Write to CSV with current time
            csv_writer.writerow([current_time, class_label, confidence])

            # Crop person image when cat is missing
            if class_label == 'cat':
                cat_detected = True
            elif class_label == 'person':
                person_detected = True
                if not cat_detected:
                    # Crop person image
                    crop = frame[int(y):int(y + height), int(x):int(x + width)]
                    cv2.imwrite(f"{output_folder}/person_{current_time}.jpg", crop)

            # Draw bounding boxes
            bbox = detection[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Control LED if cat is missing
    if person_detected and not cat_detected:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(10)
        GPIO.output(LED_PIN, GPIO.LOW)
    
    # Display the frame with bounding boxes and labels
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()  # Cleanup GPIO pins

