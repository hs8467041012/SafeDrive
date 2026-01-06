import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")  # <-- update path if needed

# Open webcam (0 = default camera, or replace with video path)
cap = cv2.VideoCapture(0)  # for webcam
# cap = cv2.VideoCapture("your_video.mp4")  # for video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)   # bounding boxes
        classes = r.boxes.cls.cpu().numpy().astype(int)  # class IDs
        confs = r.boxes.conf.cpu().numpy()               # confidence

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            label = model.names[int(cls)]  # get class name (Alert / Drowsy)
            conf_txt = f"{label} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, conf_txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show result
    cv2.imshow("YOLOv8 Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
