#In this updated code, the RPi.GPIO library is used to control the LED connected to a GPIO pin. Here are the modifications made to the code:

#Added GPIO setup code to configure the LED pin as an output.
#Added an LED control section inside the loop that checks if a person is detected. When a person is detected, the LED is turned on for 10 seconds and then turned off.
#Added a GPIO.cleanup() call at the end of the code to release the GPIO resources.
#Make sure you have the necessary dependencies installed, including torch, cv2, csv, and RPi.GPIO. Also, connect an LED to the specified GPIO pin (LED_PIN) on your Raspberry Pi.

import torch
import cv2
import csv
import time
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
    
    # Save detection results in CSV and control LED
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        for _, detection in detections.iterrows():
            class_label = detection['name']
            confidence = detection['confidence']
            
            # Write to CSV
            csv_writer.writerow([timestamp, class_label, confidence])

            if class_label == 'person':
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
