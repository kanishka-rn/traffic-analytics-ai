# 🚦 AI Smart Traffic Analytics System

An AI-powered real-time traffic monitoring and analytics system built using:

- YOLOv8
- DeepSORT
- OpenCV
- Streamlit
- Plotly

This project performs intelligent vehicle detection, tracking, speed estimation, and traffic analytics using Computer Vision and Deep Learning.

---

# 🔥 Features

## ✅ Real-Time Vehicle Detection

Detects:

- Cars
- Bikes
- Buses
- Trucks

Using:

- YOLOv8
- OpenCV

---

## ✅ Vehicle Tracking

Uses:

- DeepSORT

Features:

- Unique vehicle IDs
- Real-time tracking
- Stable detection
- No duplicate counting

---

## ✅ Full Frame Vehicle Counting

Counts vehicles instantly when they enter the frame.

Features:

- Accurate counting
- Real-time analytics
- Traffic density estimation

---

## ✅ Speed Detection

Features:

- Real-time speed estimation
- Overspeed warning system
- Dynamic speed display

---

## ✅ Traffic Density Analysis

Automatically detects:

- LOW traffic
- MEDIUM traffic
- HIGH traffic

---

## ✅ Streamlit Dashboard

Interactive dashboard with:

- Video upload
- Live processed output
- Traffic analytics
- Clean dark UI

---

## ✅ Heatmap Visualization

Visualizes:

- Traffic hotspots
- Vehicle movement density

---

# 🧠 Tech Stack

| Purpose | Technology |
|---|---|
| Detection | YOLOv8 |
| Tracking | DeepSORT |
| Computer Vision | OpenCV |
| Dashboard | Streamlit |
| Graphs | Plotly |
| Backend | Python |
| Data Handling | Pandas |
| Numerical Operations | NumPy |

---

# 📂 Project Structure

```bash
traffic-analytics-ai/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── input/
│   ├── output/
│   └── sample_videos/
│
├── models/
│   └── yolov8m.pt
│
├── tracking/
│   └── counting.py
│
├── utils/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

