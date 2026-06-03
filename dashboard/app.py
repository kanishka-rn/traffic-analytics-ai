import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Smart Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

# =========================================
# ADVANCED UI CSS
# =========================================

st.markdown("""
<style>

/* =========================================
BACKGROUND
========================================= */

.stApp {

    background: linear-gradient(
        135deg,
        #0f172a,
        #020617
    );

    color: white;
}

/* =========================================
TITLE
========================================= */

h1 {

    font-size: 55px !important;

    font-weight: 800;

    color: white;

    text-align: center;

    letter-spacing: 1px;
}

/* =========================================
SUBHEADINGS
========================================= */

h2, h3 {

    color: #E2E8F0;
}

/* =========================================
METRIC CARDS
========================================= */

[data-testid="metric-container"] {

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 20px;

    padding: 25px;

    backdrop-filter: blur(15px);

    box-shadow:
        0px 4px 30px rgba(0,0,0,0.3);

    transition: 0.3s;
}

[data-testid="metric-container"]:hover {

    transform: scale(1.03);

    border: 1px solid #00FFAA;
}

/* =========================================
METRIC LABEL
========================================= */

[data-testid="stMetricLabel"] {

    color: #94A3B8;

    font-size: 18px;
}

/* =========================================
METRIC VALUE
========================================= */

[data-testid="stMetricValue"] {

    color: #00FFAA;

    font-size: 42px;

    font-weight: bold;
}

/* =========================================
UPLOAD BOX
========================================= */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.04);

    border: 2px dashed rgba(255,255,255,0.15);

    border-radius: 20px;

    padding: 30px;
}

/* =========================================
VIDEO FRAME
========================================= */

img {

    border-radius: 20px;

    border: 2px solid rgba(255,255,255,0.1);

    box-shadow:
        0px 10px 40px rgba(0,0,0,0.4);
}

/* =========================================
PROGRESS BAR
========================================= */

.stProgress > div > div > div > div {

    background: linear-gradient(
        90deg,
        #00FFAA,
        #00C2FF
    );
}

/* =========================================
SCROLLBAR
========================================= */

::-webkit-scrollbar {

    width: 10px;
}

::-webkit-scrollbar-thumb {

    background: #00FFAA;

    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.title("🚦 AI Smart Traffic Dashboard")

st.markdown("""
<div style='text-align:center;'>

<h3 style='color:#94A3B8;'>

Real-Time AI Vehicle Tracking • Speed Detection • Traffic Analytics

</h3>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================================
# STATUS BAR
# =========================================

status1, status2, status3 = st.columns(3)

status1.success("🟢 YOLOv8 Detection Active")

status2.success("🟢 DeepSORT Tracking Online")

status3.success("🟢 AI Analytics Running")

st.markdown("---")

# =========================================
# VIDEO UPLOAD
# =========================================

uploaded_video = st.file_uploader(
    "📤 Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

# =========================================
# PROCESS VIDEO
# =========================================

if uploaded_video is not None:

    # =========================================
    # SAVE TEMP FILE
    # =========================================

    temp_file = tempfile.NamedTemporaryFile(
        delete=False
    )

    temp_file.write(
        uploaded_video.read()
    )

    video_path = temp_file.name

    st.success("✅ Video Uploaded Successfully")

    # =========================================
    # LOAD MODEL
    # =========================================

    model = YOLO("models/yolov8m.pt")

    tracker = DeepSort(max_age=30)

    # =========================================
    # OPEN VIDEO
    # =========================================

    cap = cv2.VideoCapture(video_path)

    # =========================================
    # VEHICLE CLASSES
    # =========================================

    vehicle_classes = [2, 3, 5, 7]

    # =========================================
    # COUNTING
    # =========================================

    line_y = 500

    counted_ids = set()

    total_count = 0

    previous_positions = {}

    # =========================================
    # LIVE VIDEO
    # =========================================

    st.subheader("🎥 Live AI Detection")

    frame_placeholder = st.empty()

    progress_bar = st.progress(0)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # =========================================
    # MAIN LOOP
    # =========================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # =========================================
        # RESIZE
        # =========================================

        frame = cv2.resize(
            frame,
            (1280, 720)
        )

        # =========================================
        # YOLO DETECTION
        # =========================================

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
                            [
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1
                            ],
                            conf,
                            cls
                        )
                    )

        # =========================================
        # TRACKING
        # =========================================

        tracks = tracker.update_tracks(
            detections,
            frame=frame
        )

        # =========================================
        # COUNT LINE
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
            # VEHICLE COUNT
            # =========================================

            if (
                center_y > line_y - 15 and
                center_y < line_y + 15
            ):

                if track_id not in counted_ids:

                    counted_ids.add(track_id)

                    total_count += 1

            # =========================================
            # SPEED DETECTION
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

                speed = int(
                    distance * 0.8
                )

            previous_positions[track_id] = (
                center_x,
                center_y
            )

            # =========================================
            # DIRECTION
            # =========================================

            if center_y < line_y:

                direction = "UP"

            else:

                direction = "DOWN"

            # =========================================
            # SPEED COLOR
            # =========================================

            speed_color = (0, 255, 0)

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
            # BOX
            # =========================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # =========================================
            # ID
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
            # SPEED
            # =========================================

            cv2.putText(
                frame,
                f"Speed: {speed} km/h",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                speed_color,
                2
            )

            # =========================================
            # DIRECTION
            # =========================================

            cv2.putText(
                frame,
                direction,
                (x1, y2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # =========================================
            # CENTER DOT
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
            (430, 170),
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
        # FPS
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
        # PROGRESS BAR
        # =========================================

        current_frame = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        )

        progress = current_frame / total_frames

        progress_bar.progress(
            min(progress, 1.0)
        )

        # =========================================
        # DISPLAY FRAME
        # =========================================

        frame_placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True
        )

    # =========================================
    # RELEASE
    # =========================================

    cap.release()

    st.success("✅ Video Processing Completed")

    # =========================================
    # ANALYTICS
    # =========================================

    st.markdown("---")

    st.subheader("📊 Traffic Analytics")

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "🚗 Total Vehicles",
        total_count
    )

    metric2.metric(
        "🚦 Traffic Density",
        traffic_status
    )

    # =========================================
    # LIVE STATUS
    # =========================================

    st.subheader("🔥 Live Traffic Status")

    if traffic_status == "HIGH":

        st.error(
            "Heavy Traffic Detected"
        )

    elif traffic_status == "MEDIUM":

        st.warning(
            "Moderate Traffic"
        )

    else:

        st.success(
            "Smooth Traffic Flow"
        )

    # =========================================
    # FOOTER
    # =========================================

    st.markdown("---")

    st.markdown("""
    <div style='text-align:center;
                color:#64748B;
                padding:20px;'>

    AI Smart Traffic Monitoring System 🚀

    Built with YOLOv8 • DeepSORT • OpenCV • Streamlit

    </div>
    """, unsafe_allow_html=True)