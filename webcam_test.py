from ultralytics import YOLO
import cv2
import time

model = YOLO("./output/train3/weights/best.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        print(f"Object bbox size: width={w:.1f}, height={h:.1f}")

    cv2.imshow("YOLOv8 - BBox Size", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

    elapsed = time.time() - start
    if elapsed < 0.1:
        time.sleep(0.1 - elapsed)

cap.release()
cv2.destroyAllWindows()
