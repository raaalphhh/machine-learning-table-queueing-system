import cv2
import numpy as np
from object_detection import ObjectDetection
import math

od = ObjectDetection()
cap = cv2.VideoCapture("eatery5.mp4")

frame_count = 0
tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        break

    current_frame_centers = []
    class_ids, scores, boxes = od.detect(frame)

    for class_id, score, box in zip(class_ids, scores, boxes):
        if od.classes[class_id] == "diningtable":
            x, y, w, h = box
            cx, cy = int((x + x + w) / 2), int((y + y + h) / 2)
            current_frame_centers.append((cx, cy, (x, y, w, h)))

    if frame_count <= 2:
        for center in current_frame_centers:
            tracking_objects[track_id] = center
            track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()

        for object_id, tracked_center in list(tracking_objects_copy.items()):
            object_exists = False

            for center in current_frame_centers:
                distance = math.hypot(tracked_center[0] - center[0], tracked_center[1] - center[1])

                if distance < 20:
                    tracking_objects[object_id] = center
                    object_exists = True
                    current_frame_centers.remove(center)
                    break

            if not object_exists:
                tracking_objects.pop(object_id)

        for center in current_frame_centers:
            tracking_objects[track_id] = center
            track_id += 1

    for object_id, (cx, cy, (x, y, w, h)) in tracking_objects.items():
        padding = 10
        x1, y1 = max(x - padding, 0), max(y - padding, 0)
        x2, y2 = min(x + w + padding, frame.shape[1]), min(y + h + padding, frame.shape[0])
        table_crop = frame[y1:y2, x1:x2]

        inner_class_ids, _, _ = od.detect(table_crop)
        is_occupied = any(od.classes[c] in ["person", "chair", "spoon", "cup", "bowl"] for c in inner_class_ids)

        status = "Occupied" if is_occupied else "Available"
        color = (0, 0, 255) if is_occupied else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"ID {object_id}: {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Eatery Table Availability Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()