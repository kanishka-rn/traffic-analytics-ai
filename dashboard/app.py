import streamlit as st
import tempfile
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
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

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #020617); color: white; }
h1 { font-size: 48px !important; font-weight: 800; color: white; text-align: center; }
h2, h3 { color: #E2E8F0; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 25px;
    box-shadow: 0px 4px 30px rgba(0,0,0,0.3); transition: 0.3s;
}
[data-testid="metric-container"]:hover { transform: scale(1.03); border: 1px solid #00FFAA; }
[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 18px; }
[data-testid="stMetricValue"] { color: #00FFAA; font-size: 42px; font-weight: bold; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #00FFAA, #00C2FF); }
</style>
""", unsafe_allow_html=True)

st.title("🚦 AI Smart Traffic Dashboard")
st.markdown("""
<div style='text-align:center;'>
<h3 style='color:#94A3B8;'>Real-Time AI Vehicle Tracking • Speed Detection • Traffic Analytics</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.success("🟢 YOLOv8 Detection Active")
c2.success("🟢 ByteTrack Tracking Online")
c3.success("🟢 AI Analytics Running")
st.markdown("---")

# =========================================
# HELPERS
# =========================================

def draw_frame(img_array, tracks, line_y, total_count, traffic_status, prev_positions):
    """Draw annotations on a numpy RGB frame using PIL (no cv2)."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # counting line
    draw.line([(0, line_y), (img.width, line_y)], fill=(0, 255, 255), width=2)

    # info panel
    draw.rectangle([(8, 8), (260, 100)], fill=(0, 0, 0))
    draw.text((16, 14), f"Vehicles: {total_count}", fill=(0, 255, 255))
    status_color = (255, 50, 50) if traffic_status == "HIGH" else (255, 200, 0) if traffic_status == "MEDIUM" else (0, 200, 100)
    draw.text((16, 60), f"Traffic: {traffic_status}", fill=status_color)

    speeds = {}
    for i in range(len(tracks)):
        x1, y1, x2, y2 = map(int, tracks.xyxy[i])
        track_id = int(tracks.tracker_id[i])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        speed = 0
        if track_id in prev_positions:
            px, py = prev_positions[track_id]
            speed = int(np.hypot(cx - px, cy - py) * 0.8)
        prev_positions[track_id] = (cx, cy)
        speeds[track_id] = speed

        direction = "UP" if cy < line_y else "DOWN"
        box_color = (255, 50, 50) if speed > 80 else (0, 255, 0)

        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=2)
        draw.text((x1, max(y1 - 16, 2)), f"ID:{track_id}", fill=box_color)
        draw.text((x1, y2 + 2), f"{speed}km/h {direction}", fill=box_color)
        if speed > 80:
            draw.text((x1, y2 + 18), "OVER SPEED", fill=(255, 50, 50))
        draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], fill=(255, 0, 0))

    return np.array(img), speeds

# =========================================
# MODEL
# =========================================

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "..", "models", "yolov8n.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    return YOLO("yolov8n.pt")

# =========================================
# VIDEO UPLOAD
# =========================================

uploaded_video = st.file_uploader("📤 Upload Traffic Video", type=["mp4", "avi", "mov"])

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

    tracker = sv.ByteTrack()

    VEHICLE_CLASSES = {2, 3, 5, 7}
    LINE_Y_RATIO = 0.75
    FRAME_SKIP = 2

    counted_ids = set()
    total_count = 0
    previous_positions = {}

    st.subheader("🎥 Live AI Detection")
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
        meta = reader.get_meta_data()
        total_frames = meta.get("nframes", None) or int(meta.get("duration", 60) * meta.get("fps", 25))
        total_frames = max(total_frames, 1)
    except Exception as e:
        st.error(f"Could not read video: {e}")
        st.stop()

    frame_idx = 0
    for frame_rgb in reader:
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            continue

        # resize to 854x480 using PIL
        pil_img = Image.fromarray(frame_rgb).resize((854, 480))
        frame_rgb = np.array(pil_img)
        line_y = int(480 * LINE_Y_RATIO)

        # YOLO expects BGR — convert
        frame_bgr = frame_rgb[:, :, ::-1].copy()
        results = model(frame_bgr, verbose=False)[0]

        # filter vehicles
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        mask = np.isin(cls_ids, list(VEHICLE_CLASSES))
        filtered = results.boxes[mask]

        if len(filtered) > 0:
            detections = sv.Detections(
                xyxy=filtered.xyxy.cpu().numpy(),
                confidence=filtered.conf.cpu().numpy(),
                class_id=filtered.cls.cpu().numpy().astype(int),
            )
            tracks = tracker.update_with_detections(detections)
        else:
            tracks = sv.Detections.empty()

        # count crossings
        for i in range(len(tracks)):
            _, y1, _, y2 = map(int, tracks.xyxy[i])
            cy = (y1 + y2) // 2
            tid = int(tracks.tracker_id[i])
            if line_y - 15 < cy < line_y + 15:
                if tid not in counted_ids:
                    counted_ids.add(tid)
                    total_count += 1

        # traffic status
        if total_count < 10:
            traffic_status = "LOW"
        elif total_count < 25:
            traffic_status = "MEDIUM"
        else:
            traffic_status = "HIGH"

        annotated, _ = draw_frame(frame_rgb, tracks, line_y, total_count, traffic_status, previous_positions)

        progress_bar.progress(min(frame_idx / total_frames, 1.0))
        frame_placeholder.image(annotated, use_container_width=True)

    reader.close()
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
Built with YOLOv8 • ByteTrack • imageio • Streamlit
</div>
""", unsafe_allow_html=True)
