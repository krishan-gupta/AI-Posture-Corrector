# 🏋️‍♂️ AI Posture Corrector & Virtual Gym Trainer

> **An Interdisciplinary Project (IDP)**  
> Developed by students of **Vellore Institute of Technology (VIT), Chennai**

---

## 👥 Project Team

* **Vartika Bhagat**
* **Krishan Gupta**
* **Vanshika Pandey**

---

## 📌 Project Overview

The **AI Powered Posture Corrector for Gym and Home Exercises** is an intelligent, vision-based virtual trainer and workout assistant designed to facilitate safe, guided home workouts. By leveraging computer vision and real-time human pose estimation, the software monitors user movements, tracks key anatomical landmarks, and guides fitness enthusiasts to execute exercises with proper form and posture without needing expensive wearable sensors.

---

## 📊 Project Status (Review 2 — ~20% Milestone)

This project is currently under active development. For the **Review 2** submission, the foundational computer vision and pose tracking pipeline has been implemented:

### ✅ What Has Been Built So Far:
- **Live Video & Camera Feed Processing**: Real-time webcam capture with OpenCV.
- **MediaPipe Pose Landmarker Integration**: Integration with MediaPipe Tasks Vision API (`pose_landmarker.task`) for high-performance landmark extraction.
- **Joint & Landmark Tracking**: Real-time tracking and mapping of crucial body joints:
  - Face
  - Upper Body: Shoulders, Elbows, Wrists
  - Lower Body: Hips, Knees, Ankles
- **Skeleton & Joint Overlay**: Dynamic rendering of connection lines (skeletal structure) and key joint nodes over the video stream.
- **Person Presence Detection**: Visual state indicators (`PERSON DETECTED` / `NO PERSON DETECTED`).
- **Responsive Display**: Resizable OpenCV rendering window supporting various screen resolutions.

---

## 🚀 Future Roadmap & Upcoming Features

The following features and enhancements are scheduled for subsequent development phases:

- 🎯 **Enhanced Accuracy & Angle Analysis**: High-precision geometric and mathematical angle calculations for joint alignment.
- 🧠 **Custom ML Model Training**: Training custom classification models to analyze specific exercise trajectories and exercise states.
- 📋 **Workout Routine Selection (Split Days)**: Modular workout categories such as **Leg Day**, **Chest Day**, **Back & Arms**, and **Core Workouts**.
- 🔍 **Auto-Detection of Exercises**: Automatic classification of exercises (e.g., Squats, Push-ups, Bicep Curls, Lunges) without requiring manual mode switching.
- ⚠️ **Real-Time Form & Posture Feedback**:
  - Clear visual and textual alerts for correct vs. incorrect postures.
  - Pinpointing specific body parts with incorrect placement (e.g., *“Keep your back straight”*, *“Knees overshooting toes”*, *“Elbows flaring”*).
- 🔢 **Automated Rep Counter & Tracker**:
  - Live repetition counting based on repetition phase analysis (eccentric / concentric phases).
  - Tracking completed sets, pending exercises, and workout duration.
- 👤 **Personalized Training & Recommendations**: User-adaptive workout plans with difficulty adjustments and performance analytics.

---

## 🛠️ Technology Stack

- **Programming Language**: Python 3.10+ / 3.13
- **Computer Vision**: OpenCV (`opencv-python`)
- **Pose Estimation**: Google MediaPipe (`mediapipe`)

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone https://github.com/krishan-gupta/AI-Posture-Corrector.git
cd AI-Posture-Corrector
```

### 2. Install Dependencies
Ensure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the pose detection pipeline:
```bash
python main.py
```

### 4. Controls
- **Resize Window**: Drag window borders or maximize the screen.
- **Exit Application**: Press **`q`** on your keyboard while the window is focused.

---

## 📁 Repository Structure

```
AI-Posture-Corrector/
│
├── pose_landmarker.task   # MediaPipe pretrained pose landmarker model
├── pose_detector.py       # Pose detection & landmark extraction module
├── main.py                # Main webcam feed, skeleton rendering & UI loop
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
