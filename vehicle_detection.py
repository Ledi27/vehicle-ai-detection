# vehicle_detection_simple.py
# Simple vehicle detection demo that clearly shows "Camera: ACTIVE" status
# and draws a green box + "VEHICLE" label for each detection.
# Press 'q' to quit.

import cv2
import time

# Use local haarcascade file named 'haarcascade_car.xml' in same folder
CASCADE_PATH = "haarcascade_car.xml"

# Load cascade - use local file to avoid missing default cascade
car_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if car_cascade.empty():
    raise IOError(f"Could not load cascade classifier from '{CASCADE_PATH}'. Make sure the file exists in the project folder.")

# Open default webcam (0). Replace with "traffic.mp4" to use a file.
cap = cv2.VideoCapture("traffic.mp4")
if not cap.isOpened():
    raise IOError("Cannot open webcam. Check camera connection or try a video file path.")

# For FPS calculation
prev_time = time.time()
fps = 0.0

print("Camera opened. Press 'q' in the video window to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received. Exiting.")
            break

        # Resize to a reasonable width for speed/display (optional)
        frame = cv2.resize(frame, (960, 540))

        # Convert to grayscale for the cascade detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect vehicles (returns list of rectangles)
        # Tune scaleFactor and minNeighbors if detection is too noisy
        vehicles = car_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8)

        # Draw a status bar at the top (dark rectangle)
        cv2.rectangle(frame, (0, 0), (960, 40), (40, 40, 40), -1)

        # Compute FPS
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time)) if current_time != prev_time else fps
        prev_time = current_time
        fps_text = f"FPS: {fps:.1f}"

        if len(vehicles) == 0:
            # No vehicles detected: show camera active + no vehicle message
            status_text = "Camera: ACTIVE — No vehicle"
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, fps_text, (760, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            # At least one vehicle detected: draw green boxes and label "VEHICLE"
            status_text = "Camera: ACTIVE — VEHICLE"
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.putText(frame, fps_text, (760, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            for (x, y, w, h) in vehicles:
                # Draw green bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put label "VEHICLE" above the box
                cv2.putText(frame, "VEHICLE", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Vehicle Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user (KeyboardInterrupt).")

finally:
    # Always release camera and close windows
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")
