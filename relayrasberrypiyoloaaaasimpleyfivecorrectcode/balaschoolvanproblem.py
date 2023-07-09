import torch
import cv2
import csv
import time
import datetime

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
csv_fields = ['start_time', 'end_time', 'class', 'confidence']

# Initialize CSV file
with open(csv_file, 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_fields)

# Object detection loop
start_time = None
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)

    # Get detection information
    detections = results.pandas().xyxy[0]

    # Get the current computer time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Draw bounding boxes
    for _, detection in detections.iterrows():
        class_label = detection['name']
        confidence = detection['confidence']
        bbox = detection[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Check if any objects are detected
        if start_time is None:
            # Start of a new detection
            start_time = current_time
        else:
            # End of a detection
            end_time = current_time

            # Save detection results in CSV with start and end times
            with open(csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([start_time, end_time, class_label, confidence])

            # Reset start time
            start_time = None
    
    # Display the frame with bounding boxes and labels
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

