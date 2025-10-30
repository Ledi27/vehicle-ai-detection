# 🚗 Vehicle AI Detection App

## 🎯 Overview
This is a simple yet powerful Python application that detects and tracks vehicles in a video using **YOLOv8** (from the `ultralytics` library).  
It can process live camera feed or pre-recorded traffic videos and automatically identify cars, drawing green bounding boxes around them.

This project was created as part of an **Artificial Intelligence master’s portfolio**, focusing on real-world applications for **autonomous vehicles and smart traffic systems**.

---

## ⚙️ Features
- Real-time **vehicle detection** using YOLOv8  
- **Tracking** with unique IDs per vehicle  
- Works with **live webcam** or **video files (e.g., `traffic.mp4`)**  
- Adjustable confidence threshold for precision  
- Output video automatically saved as `vehicle_detection_output.mp4`

---

## 🧠 Tech Stack
- **Python 3.11+**
- **OpenCV** for video frame processing
- **Ultralytics YOLOv8** for object detection
- **NumPy & SciPy** for numerical operations

---

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Vehicle_AI_Portfolio.git
   cd Vehicle_AI_Portfolio
