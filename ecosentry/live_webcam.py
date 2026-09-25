import cv2
from ultralytics import YOLO

model = YOLO("ecosentry/models/yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("ECOSENTRY webcam started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        "INPUT: NORMAL WEBCAM",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "THERMAL DATA: NOT AVAILABLE",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("ECOSENTRY LIVE MONITOR", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("ECOSENTRY webcam stopped.")