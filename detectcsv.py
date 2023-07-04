#pip install torch torchvision opencv-python pandas yolov5

import torch
import numpy as np
import cv2
import csv

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
csv_fields = ['class', 'confidence', 'x', 'y', 'width', 'height']
csv_output = open(csv_file, 'w')
csv_writer = csv.DictWriter(csv_output, fieldnames=csv_fields)
csv_writer.writeheader()

# Folder setup for saving cropped detections
output_folder = 'detection_crops/'

# Object detection loop
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform object detection
    results = model(frame)

    # Get detection information
    detections = results.pandas().xyxy[0]
    
    # Save detection results in CSV
    for _, detection in detections.iterrows():
        csv_writer.writerow({
            'class': detection['name'],
            'confidence': detection['confidence'],
            'x': detection['xmin'],
            'y': detection['ymin'],
            'width': detection['xmax'] - detection['xmin'],
            'height': detection['ymax'] - detection['ymin']
        })
    
    # Save cropped detections
    for i, (_, detection) in enumerate(detections.iterrows()):
        crop = frame[int(detection['ymin']):int(detection['ymax']),
                     int(detection['xmin']):int(detection['xmax'])]
        cv2.imwrite(output_folder + f'detection_{i}.jpg', crop)

    # Display the frame with bounding boxes
    if len(results.pred) > 0:
        pred = results.pred[0]
        for det in pred:
            bbox = det[:4].int().cpu().numpy()  # Convert bbox to int values
            label = int(det[5])
            confidence = det[4]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
csv_output.close()

