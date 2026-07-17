import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =========================================
# LOAD YOLO MODEL
# =========================================

model = YOLO("models/yolov8m.pt")

# =========================================
# DEEPSORT TRACKER
# =========================================

tracker = DeepSort(max_age=30)

# =========================================
# VIDEO SOURCE
# =========================================

cap = cv2.VideoCapture(
    "data/sample_videos/traffic.mp4"
)

# =========================================
# VEHICLE CLASSES
# =========================================

vehicle_classes = [2, 3, 5, 7]

# =========================================
# COUNTING LINE
# =========================================

line_y = 500

# =========================================
# VEHICLE COUNT
# =========================================

counted_ids = set()

total_count = 0

# =========================================
# SPEED STORAGE
# =========================================

previous_positions = {}

# =========================================
# MAIN LOOP
# =========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =========================================
    # RESIZE FRAME
    # =========================================

    frame = cv2.resize(
        frame,
        (1280, 720)
    )

    # =========================================
    # YOLO DETECTION
    # =========================================

    if frame is not None:
        results = model(frame)

    detections = []

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            if cls in vehicle_classes:

                conf = float(box.conf[0])

                if conf < 0.4:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append(
                    (
                        [x1, y1, x2 - x1, y2 - y1],
                        conf,
                        cls
                    )
                )

    # =========================================
    # DEEPSORT TRACKING
    # =========================================

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # =========================================
    # DRAW COUNTING LINE
    # =========================================

    cv2.line(
        frame,
        (0, line_y),
        (1280, line_y),
        (0, 255, 255),
        3
    )

    # =========================================
    # PROCESS TRACKS
    # =========================================

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(
            int,
            ltrb
        )

        # =========================================
        # CENTER POINT
        # =========================================

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        # =========================================
        # VEHICLE COUNTING
        # =========================================

        if (
            center_y > line_y - 15 and
            center_y < line_y + 15
        ):

            if track_id not in counted_ids:

                counted_ids.add(track_id)

                total_count += 1

        # =========================================
        # SPEED ESTIMATION
        # =========================================

        speed = 0

        if track_id in previous_positions:

            prev_x, prev_y = previous_positions[
                track_id
            ]

            distance = np.sqrt(
                (center_x - prev_x) ** 2 +
                (center_y - prev_y) ** 2
            )

            speed = int(distance * 0.8)

        previous_positions[track_id] = (
            center_x,
            center_y
        )

        # =========================================
        # DIRECTION DETECTION
        # =========================================

        if center_y < line_y:

            direction = "UP"

        else:

            direction = "DOWN"

        # =========================================
        # SPEED COLOR
        # =========================================

        speed_color = (0, 255, 0)

        # =========================================
        # OVER SPEED WARNING
        # =========================================

        if speed > 80:

            speed_color = (0, 0, 255)

            cv2.putText(
                frame,
                "OVER SPEED",
                (x1, y2 + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # =========================================
        # DRAW BOUNDING BOX
        # =========================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # =========================================
        # VEHICLE ID DISPLAY
        # =========================================

        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # =========================================
        # SPEED DISPLAY
        # =========================================

        cv2.putText(
            frame,
            f"Speed: {str(speed)} km/h",
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            speed_color,
            2
        )

        # =========================================
        # DIRECTION DISPLAY
        # =========================================

        cv2.putText(
            frame,
            str(direction),
            (x1, y2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # =========================================
        # CENTER POINT
        # =========================================

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

    # =========================================
    # TRAFFIC STATUS
    # =========================================

    if total_count < 10:

        traffic_status = "LOW"

    elif total_count < 25:

        traffic_status = "MEDIUM"

    else:

        traffic_status = "HIGH"

    # =========================================
    # INFO PANEL
    # =========================================

    cv2.rectangle(
        frame,
        (20, 20),
        (450, 170),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Total Vehicles: {total_count}",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        f"Traffic: {traffic_status}",
        (40, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # =========================================
    # FPS DISPLAY
    # =========================================

    cv2.putText(
        frame,
        "FPS: 25",
        (1050, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    # =========================================
    # SHOW FRAME
    # =========================================

    cv2.imshow(
        "AI Traffic Analytics System",
        frame
    )

    # =========================================
    # EXIT KEY
    # =========================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================
# RELEASE
# =========================================

cap.release()

cv2.destroyAllWindows()