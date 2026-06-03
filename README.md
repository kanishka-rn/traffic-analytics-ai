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
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/traffic-analytics-ai.git
```

---

## 2️⃣ Open Project Folder

```bash
cd traffic-analytics-ai
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 4️⃣ Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 5️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

## Run Main Detection System

```bash
python main.py
```

---

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📊 Dashboard Features

- Upload traffic video
- Real-time vehicle tracking
- Speed analytics
- Traffic density monitoring
- Live processed video output

---

# 🚀 Future Improvements

- Lane detection
- Helmet detection
- Accident detection
- Number plate recognition
- Emergency vehicle prioritization
- Traffic prediction using AI

---

# 📸 Sample Output

Features shown in output:

✅ Vehicle Tracking  
✅ Vehicle Counting  
✅ Speed Detection  
✅ Traffic Density  
✅ Heatmaps  
✅ Real-Time Monitoring  

---

# 👨‍💻 Author

Developed by **KANISHKA RN**

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the project  
🚀 Share on LinkedIn
