import torch
import numpy as np
import cv2
import csv
import time
from torchvision.ops.boxes import nms

# Load the YOLOv5 model
weights = "yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

# Load the dataset configuration
data = "data/coco128.yaml"
model.yaml = data

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)  # Use webcam (change the index if you have multiple cameras)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# CSV setup for saving detection results
csv_file = 'detection_results.csv'
csv_fields = ['timestamp', 'class', 'confidence', 'x', 'y', 'width', 'height']

# Folder setup for saving cropped detections
output_folder = 'detection_crops'

# Load class labels
class_labels = model.names

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

    # Apply non-maximum suppression (NMS)
    boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
    scores = detections['confidence'].values.astype(np.float32)
    keep_indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.5)
    keep_indices = keep_indices.cpu().numpy().astype(np.int32)  # Convert to NumPy array of integers

    # Get the timestamp
    timestamp = time.time()
    
    # Save detection results in CSV and crop images
    with open(csv_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        for idx in keep_indices:
            detection = detections.iloc[idx]
            class_label = detection['name']
            confidence = detection['confidence']
            x = detection['xmin']
            y = detection['ymin']
            width = detection['xmax'] - detection['xmin']
            height = detection['ymax'] - detection['ymin']
            
            # Write to CSV
            csv_writer.writerow([timestamp, class_label, confidence, x, y, width, height])
            
            # Crop image
            crop = frame[int(y):int(y + height), int(x):int(x + width)]
            cv2.imwrite(f"{output_folder}/{class_label}_{timestamp}.jpg", crop)

    # Display the frame with bounding boxes and labels
    if len(results.pred) > 0:
        pred = results.pred[0][keep_indices]
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
