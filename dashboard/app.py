import streamlit as st
import cv2
import tempfile
import os
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
# UI CSS
# =========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
h1 {
    font-size: 55px !important;
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
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 30px;
}
img {
    border-radius: 20px;
    border: 2px solid rgba(255,255,255,0.1);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
}
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

# =========================================
# STATUS BAR
# =========================================

status1, status2, status3 = st.columns(3)
status1.success("🟢 YOLOv8 Detection Active")
status2.success("🟢 DeepSORT Tracking Online")
status3.success("🟢 AI Analytics Running")

st.markdown("---")

# =========================================
# MODEL LOADER (cached across sessions)
# =========================================

@st.cache_resource
def load_model():
    # resolve model path relative to this file
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "..", "models", "yolov8n.pt")
    if not os.path.exists(model_path):
        # fallback: let ultralytics auto-download yolov8n
        return YOLO("yolov8n.pt")
    return YOLO(model_path)

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

    # Save to temp file
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

    # DeepSort per-session (not cached — it holds mutable state)
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Could not open video. Please try a different file.")
        st.stop()

    vehicle_classes = [2, 3, 5, 7]
    line_y = 360  # adjusted for 640-height display
    counted_ids = set()
    total_count = 0
    previous_positions = {}

    st.subheader("🎥 Live AI Detection")
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        total_frames = 1

    frame_skip = 2   # process every 2nd frame to reduce load on cloud
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (854, 480))
        display_h = 480

        results = model(frame, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in vehicle_classes:
                    continue
                conf = float(box.conf[0])
                if conf < 0.4:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        # draw counting line
        cv2.line(frame, (0, line_y), (854, line_y), (0, 255, 255), 2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # count vehicles crossing the line
            if line_y - 15 < center_y < line_y + 15:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1

            # speed estimate
            speed = 0
            if track_id in previous_positions:
                px, py = previous_positions[track_id]
                speed = int(np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2) * 0.8)
            previous_positions[track_id] = (center_x, center_y)

            direction = "UP" if center_y < line_y else "DOWN"
            speed_color = (0, 0, 255) if speed > 80 else (0, 255, 0)

            if speed > 80:
                cv2.putText(frame, "OVER SPEED", (x1, y2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"{speed}km/h", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, speed_color, 2)
            cv2.putText(frame, direction, (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        # traffic status
        if total_count < 10:
            traffic_status = "LOW"
        elif total_count < 25:
            traffic_status = "MEDIUM"
        else:
            traffic_status = "HIGH"

        # info panel overlay
        cv2.rectangle(frame, (10, 10), (300, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Vehicles: {total_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Traffic: {traffic_status}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        progress_bar.progress(min(frame_idx / total_frames, 1.0))
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    os.unlink(video_path)

    st.success("✅ Video Processing Completed")
    st.markdown("---")
    st.subheader("📊 Traffic Analytics")

    c1, c2 = st.columns(2)
    c1.metric("🚗 Total Vehicles", total_count)
    c2.metric("🚦 Traffic Density", traffic_status)

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
Built with YOLOv8 • DeepSORT • OpenCV • Streamlit
</div>
""", unsafe_allow_html=True)
