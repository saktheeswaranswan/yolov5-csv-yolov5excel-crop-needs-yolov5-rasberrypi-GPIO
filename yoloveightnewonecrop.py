import csv
import time
import cv2
import supervision as sv
from ultralytics import YOLO


def main():
    # to save the video
    writer = cv2.VideoWriter('webcam_yolo.mp4',
                             cv2.VideoWriter_fourcc(*'DIVX'),
                             7,
                             (1280, 720))

    # define resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # specify the model
    model = YOLO("yolov8n.pt")

    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Create a CSV file to log the detections
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_file = open(f"detections_{timestamp}.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Label", "Confidence", "Timestamp", "Crop"])

    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        person_detected = False
        cat_detected = False
        person_box = None

        for detection, label in zip(detections, labels):
            _, confidence, class_id, box = detection
            label_name = model.model.names[class_id]

            if label_name == 'person':
                person_detected = True
                person_box = box

            if label_name == 'cat':
                cat_detected = True

            csv_writer.writerow([label_name, confidence, timestamp, box])

        if person_detected and not cat_detected and person_box is not None:
            # Crop the person region
            x, y, w, h = person_box
            crop = frame[y:y + h, x:x + w]
            cv2.imwrite(f"person_{timestamp}.jpg", crop)

        writer.write(frame)
        cv2.imshow("yolov8", frame)

        if cv2.waitKey(30) == 27:  # break with escape key
            break

    # Close the CSV file
    csv_file.close()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

