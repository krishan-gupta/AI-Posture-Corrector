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

## 📊 Project Features & Capabilities

### ✅ Core Features Implemented:
- **3D Camera-Invariant Angle Analysis**: Calculates real Euclidean 3D joint angles using MediaPipe `pose_world_landmarks`, making flexion and extension tracking completely immune to camera distance, tilt, and perspective distortion.
- **Viewpoint & Orientation Auto-Detection**: Real-time detection of user orientation (`FRONTAL`, `SIDE_LEFT`, `SIDE_RIGHT`, `OBLIQUE`) with dynamic HUD cues.
- **Biomechanical Posture Checks**:
  - **Bicep Curls**: 3D upper-arm torso pinning (prevents cheating with front deltoids), 3D lateral elbow flaring, and peak contraction over-curling alerts.
  - **Squats**: 3D femur parallel angle, true spinal/torso inclination from vertical (prevents excessive forward lean), and knee valgus (knee cave) checks in frontal view.
  - **Tricep Overhead Extensions**: True 3D overhead vertical alignment, elbow tendon hyperflexion protection (<58°), and full overhead lockout detection.
- **Automated Repetition Counter & Tracker**: Biomechanically accurate state machines with phase tracking (`READY`, `CONCENTRIC`, `PEAK`, `ECCENTRIC`) that differentiates between clean reps and reps with posture alerts.
- **Temporal EMA Smoothing & Feedback Debouncing**:
  - Exponential Moving Average (EMA) removes landmark flutter and micro-jitter without latency.
  - Hysteresis feedback debouncer eliminates rapid visual flickering and guarantees readable messages.
- **High-Definition Stream & Modern Glassmorphic HUD**: Stream capture at 960x540 for high landmark resolution with semi-transparent HUD cards, live FPS, rep counts, and target guides.

---

## 🛠️ Technology Stack

- **Programming Language**: Python 3.10+ / 3.13
- **Computer Vision**: OpenCV (`opencv-python`)
- **Pose Estimation**: Google MediaPipe Tasks Vision (`mediapipe`)
- **Scientific Computing**: NumPy (`numpy`)

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
Start the interactive virtual trainer:
```bash
python main.py
```

### 4. Controls
- **Select Exercise**: Press `1` (Bicep Curls), `2` (Squats), or `3` (Tricep Extensions).
- **Return to Menu**: Press **`m`** while in an exercise.
- **Exit Application**: Press **`q`** on your keyboard while the window is focused.

---

## 📁 Repository Structure

```
AI-Posture-Corrector/
│
├── pose_landmarker.task   # MediaPipe pretrained pose landmarker model
├── pose_detector.py       # Pose detection & high-precision timestamping module
├── biomechanics.py        # 3D vector math, viewpoint auto-detection, EMA & rep counting
├── bicep_trainer.py       # Dedicated Bicep Curl Coach (AR Ghost, Peak & Extension ROM)
├── squat_trainer.py       # Dedicated Squat Coach (AR Parallel Thigh Ghost, Hip Reticle)
├── tricep_trainer.py      # Dedicated Tricep Coach (AR Lockout & Stretch Ghost)
├── main.py                # Main interactive menu & unified exercise launcher
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```


