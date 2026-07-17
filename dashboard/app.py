import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Smart Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

# =========================================
# UI CSS
# =========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
h1 {
    font-size: 48px !important;
    font-weight: 800;
    color: white;
    text-align: center;
    letter-spacing: 1px;
}
h2, h3 { color: #E2E8F0; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(15px);
    box-shadow: 0px 4px 30px rgba(0,0,0,0.3);
    transition: 0.3s;
}
[data-testid="metric-container"]:hover {
    transform: scale(1.03);
    border: 1px solid #00FFAA;
}
[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 18px; }
[data-testid="stMetricValue"] { color: #00FFAA; font-size: 42px; font-weight: bold; }
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00FFAA, #00C2FF);
}
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: #00FFAA; border-radius: 20px; }
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

c1, c2, c3 = st.columns(3)
c1.success("🟢 YOLOv8 Detection Active")
c2.success("🟢 ByteTrack Tracking Online")
c3.success("🟢 AI Analytics Running")
st.markdown("---")

# =========================================
# MODEL LOADER
# =========================================

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "..", "models", "yolov8n.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    return YOLO("yolov8n.pt")  # auto-download fallback

# =========================================
# VIDEO UPLOAD
# =========================================

uploaded_video = st.file_uploader(
    "� Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()
    video_path = tfile.name

    st.success("✅ Video Uploaded Successfully")

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    # ByteTrack tracker (pure Python, no torch dependency)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video file.")
        st.stop()

    VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorbike, bus, truck
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    # counting line at 75% of frame height
    LINE_Y_RATIO = 0.75

    counted_ids = set()
    total_count = 0
    previous_positions = {}

    st.subheader("🎥 Live AI Detection")
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    frame_idx = 0
    FRAME_SKIP = 2  # process every 2nd frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            continue

        frame = cv2.resize(frame, (854, 480))
        line_y = int(480 * LINE_Y_RATIO)

        # YOLO inference
        results = model(frame, verbose=False)[0]

        # filter vehicle classes only
        mask = np.isin(results.boxes.cls.cpu().numpy().astype(int),
                       list(VEHICLE_CLASSES))
        filtered_boxes = results.boxes[mask]

        # build supervision Detections
        detections = sv.Detections(
            xyxy=filtered_boxes.xyxy.cpu().numpy(),
            confidence=filtered_boxes.conf.cpu().numpy(),
            class_id=filtered_boxes.cls.cpu().numpy().astype(int),
        )

        # track
        tracks = tracker.update_with_detections(detections)

        # draw counting line
        cv2.line(frame, (0, line_y), (854, line_y), (0, 255, 255), 2)

        for i in range(len(tracks)):
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            track_id = int(tracks.tracker_id[i])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # count crossing
            if line_y - 15 < center_y < line_y + 15:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1

            # speed estimate
            speed = 0
            if track_id in previous_positions:
                px, py = previous_positions[track_id]
                speed = int(np.hypot(center_x - px, center_y - py) * 0.8)
            previous_positions[track_id] = (center_x, center_y)

            direction = "UP" if center_y < line_y else "DOWN"
            speed_color = (0, 0, 255) if speed > 80 else (0, 255, 0)

            if speed > 80:
                cv2.putText(frame, "OVER SPEED", (x1, y2 + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"{speed}km/h", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 2)
            cv2.putText(frame, direction, (x1, y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        # traffic status
        if total_count < 10:
            traffic_status = "LOW"
        elif total_count < 25:
            traffic_status = "MEDIUM"
        else:
            traffic_status = "HIGH"

        # info overlay
        cv2.rectangle(frame, (8, 8), (260, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Vehicles: {total_count}", (16, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Traffic: {traffic_status}", (16, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        progress_bar.progress(min(frame_idx / total_frames, 1.0))
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    try:
        os.unlink(video_path)
    except Exception:
        pass

    st.success("✅ Processing Complete")
    st.markdown("---")
    st.subheader("📊 Traffic Analytics")

    m1, m2 = st.columns(2)
    m1.metric("🚗 Total Vehicles", total_count)
    m2.metric("🚦 Traffic Density", traffic_status)

    st.subheader("🔥 Live Traffic Status")
    if traffic_status == "HIGH":
        st.error("Heavy Traffic Detected")
    elif traffic_status == "MEDIUM":
        st.warning("Moderate Traffic")
    else:
        st.success("Smooth Traffic Flow")

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#64748B; padding:20px;'>
AI Smart Traffic Monitoring System 🚀<br>
Built with YOLOv8 • ByteTrack • OpenCV • Streamlit
</div>
""", unsafe_allow_html=True)
