import cv2
from object_detection import ObjectDetection
import object_tracking
import time

# Initialize Object Detection
od = ObjectDetection(weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg")

# Initialize Object Tracking
ot = object_tracking()

# Video capture
cap = cv2.VideoCapture("eatery5.mp4")

if not cap.isOpened():
    print("Error: Video file cannot be opened.")
    exit()

frame_skip = 1  # Process every 2nd frame if set to 1
frame_count = 0
resize_scale = 0.5

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print("End of video or failed to read frame.")
        break

    frame_count += 1
    if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
        continue

    frame = cv2.resize(frame, (int(frame.shape[1] * resize_scale), int(frame.shape[0] * resize_scale)))
    class_ids, scores, boxes = od.detect(frame)
    print(f"Frame {frame_count}: Detected {len(class_ids)} objects.")

    for class_id, score, box in zip(class_ids, scores, boxes):
        if od.classes[class_id] == "diningtable":
            x, y, w, h = box

            padding = 10
            x1, y1 = max(x - padding, 0), max(y - padding, 0)
            x2, y2 = min(x + w + padding, frame.shape[1]), min(y + h + padding, frame.shape[0])

            table_crop = frame[y1:y2, x1:x2]
            inner_class_ids, _, _ = od.detect(table_crop)
            is_occupied = any(od.classes[c] in ["person", "chair", "spoon", "cup", "bowl"] for c in inner_class_ids)

            color = (0, 255, 0) if not is_occupied else (0, 0, 255)
            status = "Available" if not is_occupied else "Occupied"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    end_time = time.time()
    print(f"Frame {frame_count} processed in {end_time - start_time:.2f} seconds.")
    cv2.imshow("Eatery Table Availability Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()