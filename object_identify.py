import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (detects 80 common objects)
model = YOLO("yolov8n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # confidence
            cls = int(box.cls[0])      # class id
            label = model.names[cls]   # class name
            
            # Draw rectangle & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.imshow("Object Identification", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup (only runs after loop breaks)
cap.release()
cv2.destroyAllWindows()
# run as : python object_identify.py 