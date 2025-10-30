# yolo_track_cars_demo.py
# YOLOv8 + simple centroid tracking for vehicle detection
# Input: traffic.mp4 (video)
# Output: live display + saved video in runs/detect/

from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial import distance

# ----------------------------
# Settings
# ----------------------------
VIDEO_PATH = "traffic.mp4"       # input video
MODEL_PATH = "yolov8n.pt"       # small YOLOv8 model
CONF_THRESH = 0.25               # confidence threshold for detection
IOU_THRESH = 0.45                # NMS threshold
CLASS_CAR = 2                    # COCO class id for "car"
TRACK_DIST_THRESHOLD = 50        # pixels to match centroids between frames
OUTPUT_FPS = 30                  # for saving video (approx)

# ----------------------------
# Load YOLOv8 model
# ----------------------------
model = YOLO(MODEL_PATH)

# ----------------------------
# Open video
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

# Video writer (same size as input)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('vehicle_detection_output.mp4', fourcc, OUTPUT_FPS, (frame_width, frame_height))

# ----------------------------
# Tracking setup
# ----------------------------
next_id = 0
tracks = {}  # id -> (cx, cy)

# ----------------------------
# Process video
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection on frame
    results = model.predict(source=frame, conf=CONF_THRESH, iou=IOU_THRESH, classes=[CLASS_CAR], verbose=False)
    
    boxes = []
    for r in results:
        for det in r.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            boxes.append((x1, y1, x2, y2))

    # Compute centroids
    centroids_new = []
    for (x1, y1, x2, y2) in boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centroids_new.append((cx, cy, x1, y1, x2, y2))

    # Match new centroids to existing tracks
    new_tracks = {}
    for cx, cy, x1, y1, x2, y2 in centroids_new:
        if len(tracks) == 0:
            new_tracks[next_id] = (cx, cy)
            track_id = next_id
            next_id += 1
        else:
            ids = list(tracks.keys())
            dists = [distance.euclidean((cx, cy), tracks[i]) for i in ids]
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < TRACK_DIST_THRESHOLD:
                track_id = ids[min_idx]
            else:
                track_id = next_id
                next_id += 1
            new_tracks[track_id] = (cx, cy)

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"VEHICLE ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    tracks = new_tracks

    # Status bar
    status_text = f"Video: ACTIVE — {len(tracks)} vehicles"
    cv2.rectangle(frame, (0, 0), (frame_width, 30), (40, 40, 40), -1)
    cv2.putText(frame, status_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Vehicle Detection", frame)
    out.write(frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Done — video saved as 'vehicle_detection_output.mp4'")
