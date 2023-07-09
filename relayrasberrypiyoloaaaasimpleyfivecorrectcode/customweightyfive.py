import torch
import cv2
import csv
import datetime
import os
import torchvision.ops as ops
import numpy as np

# Load the YOLOv5 model with custom weights
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

# Define the maximum number of classes to crop
max_classes = 10

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
    detections = results.pandas().xyxy[0]

    # Convert coordinates to torch tensor
    boxes = torch.tensor(detections[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(float), dtype=torch.float32)
    confidences = torch.tensor(detections['confidence'].values.astype(float), dtype=torch.float32)
    class_labels = [model.names[int(cls)] for cls in detections['class'].values]

    # Apply non-maximum suppression
    keep_indices = ops.nms(boxes, confidences, iou_threshold=0.5)

    # Filter detections based on NMS results
    detections = detections.iloc[keep_indices]

    # Get the current system date and time
    current_time = datetime.datetime.now().strftime("%b %d %H:%M %p")

    # Save detection results in CSV with current time
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        for _, detection in detections.iterrows():
            class_label = detection['name']
            confidence = detection['confidence']
            
            # Write to CSV with current time
            csv_writer.writerow([current_time, class_label, confidence])

            if max_classes > 0:
                # Crop and save the detected object image
                xmin, ymin, xmax, ymax = detection[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
                object_image = frame[ymin:ymax, xmin:xmax]
                object_image_filename = f'{class_label}_{confidence:.2f}_{current_time.replace(":", "").replace(" ", "_")}.jpg'
                object_image_path = os.path.join(folder_name, object_image_filename)
                cv2.imwrite(object_image_path, object_image)
                max_classes -= 1
    
    # Draw bounding boxes
    for _, detection in detections.iterrows():
        class_label = detection['name']
        confidence = detection['confidence']
        bbox = detection[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame with bounding boxes and labels
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

