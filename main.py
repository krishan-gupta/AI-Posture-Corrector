"""
AI Gym - Real-Time Posture Corrector & Virtual Trainer
Enhanced with:
- 3D Camera-Invariant Angle and Biomechanical Analysis
- Real-Time Viewpoint Auto-Detection (Frontal / Side / Oblique)
- Temporal EMA Smoothing for Flutter-Free Landmark Tracking
- Hysteresis Feedback Debouncer
- Automatic Repetition Counter & Form Quality Tracker
- High-Definition Stream Processing with Modern HUD
"""

import os
import time
import math
import cv2
import numpy as np

from pose_detector import PoseDetector
import biomechanics as bio
import tricep_trainer
import bicep_trainer
import squat_trainer

# ==============================================================================

# SETTINGS & CONSTANTS
# ==============================================================================


WINDOW_NAME = "AI Gym - Posture Corrector & Virtual Trainer"
CAMERA_INDEX = 0

# High-resolution standard stream for precise full-body joint tracking
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Color Palette (BGR)
GREEN = (40, 210, 40)
RED = (40, 40, 245)
YELLOW = (0, 215, 255)
WHITE = (245, 245, 245)
CYAN = (235, 205, 50)
BLUE = (255, 140, 0)
BLACK = (20, 20, 20)
DARK_GRAY = (35, 35, 35)
ACCENT_PURPLE = (180, 80, 220)

# MediaPipe landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

ARMS = [
    ("LEFT", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP),
    ("RIGHT", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP),
]

LEGS = [
    ("LEFT", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_SHOULDER),
    ("RIGHT", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_SHOULDER),
]


# ==============================================================================
# DRAWING & HUD UTILITIES
# ==============================================================================

def point_2d(landmarks_2d, index, width, height):
    """Convert normalized 2D landmark to pixel coordinate."""
    lm = landmarks_2d[index]
    return int(lm.x * width), int(lm.y * height)


def draw_card(frame, x, y, w, h, bg_color=DARK_GRAY, alpha=0.75, border_color=None):
    """Draw a modern semi-transparent card on the frame."""
    x = max(0, min(frame.shape[1] - 1, x))
    y = max(0, min(frame.shape[0] - 1, y))
    w = max(1, min(frame.shape[1] - x, w))
    h = max(1, min(frame.shape[0] - y, h))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    if border_color:
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1)


def draw_dotted_line(frame, start, end, color=YELLOW):
    """Draw a clean dotted guide line between two pixel points."""
    x1, y1 = start
    x2, y2 = end
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 2:
        return

    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    dash_len = 6
    gap_len = 6
    step = dash_len + gap_len

    for p in range(0, int(dist), step):
        p_end = min(p + dash_len, dist)
        pt1 = (int(x1 + dx * p), int(y1 + dy * p))
        pt2 = (int(x1 + dx * p_end), int(y1 + dy * p_end))
        cv2.line(frame, pt1, pt2, color, 2)


def draw_arm_guide(frame, shoulder, elbow, wrist, side):
    """Draw target alignment guide lines for bicep curls."""
    height, width = frame.shape[:2]
    upper_len = math.hypot(elbow[0] - shoulder[0], elbow[1] - shoulder[1])
    lower_len = math.hypot(wrist[0] - elbow[0], wrist[1] - elbow[1])
    if upper_len < 10 or lower_len < 10:
        return

    target_elbow = (shoulder[0], min(height - 1, int(shoulder[1] + upper_len)))
    extended_wrist = (target_elbow[0], min(height - 1, int(target_elbow[1] + lower_len)))
    direction = -1 if side == "LEFT" else 1
    curled_wrist = (
        max(0, min(width - 1, int(target_elbow[0] + direction * lower_len * 0.72))),
        max(0, int(target_elbow[1] - lower_len * 0.70))
    )

    draw_dotted_line(frame, shoulder, target_elbow)
    draw_dotted_line(frame, target_elbow, extended_wrist)
    draw_dotted_line(frame, target_elbow, curled_wrist)




def draw_squat_guide(frame, hip, knee, ankle, side):
    """Draw target squat depth parallel guide."""
    height, width = frame.shape[:2]
    thigh_len = math.hypot(knee[0] - hip[0], knee[1] - hip[1])
    if thigh_len < 10:
        return

    direction = -1 if side == "LEFT" else 1
    target_hip = (
        max(0, min(width - 1, int(knee[0] + direction * thigh_len * 0.85))),
        int(knee[1])
    )
    draw_dotted_line(frame, ankle, knee)
    draw_dotted_line(frame, knee, target_hip)


def draw_bone(frame, pt1, pt2, color, thickness=4):
    cv2.line(frame, pt1, pt2, color, thickness)
    cv2.circle(frame, pt1, 5, color, -1)
    cv2.circle(frame, pt2, 5, color, -1)


def draw_header_hud(frame, title, viewpoint, rep_stats, fps):
    """Render top status dashboard with exercise info, viewpoint, reps, and FPS."""
    height, width = frame.shape[:2]
    draw_card(frame, 0, 0, width, 50, bg_color=BLACK, alpha=0.88)

    # Title
    cv2.putText(frame, title, (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, CYAN, 2)

    # Viewpoint badge
    view_color = GREEN if "SIDE" in viewpoint else (YELLOW if "OBLIQUE" in viewpoint else WHITE)
    view_str = f"VIEW: {viewpoint.replace('_', ' ')}"
    cv2.putText(frame, view_str, (290, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, view_color, 2)

    # Rep Counter
    reps_str = f"REPS: {rep_stats['total_reps']} (CLEAN: {rep_stats['good_reps']})"
    cv2.putText(frame, reps_str, (520, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREEN, 2)

    # FPS & Controls
    cv2.putText(frame, f"{int(fps)} FPS | M: MENU | Q: QUIT", (width - 240, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)


def draw_bottom_hud(frame, feedback_text, feedback_color, stage_text=""):
    """Render bottom feedback card and repetition stage."""
    height, width = frame.shape[:2]
    card_h = 62
    card_y = height - card_h - 10

    draw_card(frame, 12, card_y, width - 24, card_h, bg_color=DARK_GRAY, alpha=0.90, border_color=(70, 70, 70))

    # Stage Badge if present
    x_offset = 24
    if stage_text:
        stage_w = 110
        cv2.rectangle(frame, (20, card_y + 12), (20 + stage_w, card_y + card_h - 12), (55, 55, 55), -1)
        cv2.putText(frame, stage_text, (28, card_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 2)
        x_offset = 145

    # Feedback Message
    cv2.putText(frame, f"FEEDBACK: {feedback_text}", (x_offset, card_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, feedback_color, 2)


def get_camera(index=CAMERA_INDEX):
    """Open camera with optimal driver, exposure calibration, and resolution."""
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(index)

    if not cam.isOpened():
        return cam

    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)

    try:
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except Exception:
        pass

    try:
        cam.set(cv2.CAP_PROP_AUTO_WB, 1)
    except Exception:
        pass

    try:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    except Exception:
        try:
            cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        except Exception:
            pass

    warmup_ok = False
    for _ in range(5):
        ret, frame = cam.read()
        if ret and frame is not None:
            warmup_ok = True

    if warmup_ok and frame is not None:
        mean_b = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if mean_b < 40:
            for exp_val in [-4.0, -3.0]:
                cam.set(cv2.CAP_PROP_EXPOSURE, exp_val)
                for _ in range(3):
                    ret, frame = cam.read()
                if ret and frame is not None and np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) >= 50:
                    break

    return cam


# ==============================================================================
# MENU SCREEN
# ==============================================================================

def draw_menu():
    """Display interactive exercise selection menu."""
    frame = np.zeros((540, 840, 3), dtype=np.uint8)
    frame[:] = (26, 26, 28)

    # Header Card
    draw_card(frame, 40, 30, 760, 95, bg_color=(40, 40, 44), alpha=0.95, border_color=(60, 60, 65))
    cv2.putText(frame, "AI GYM - VIRTUAL TRAINER & POSTURE CORRECTOR", (60, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.78, WHITE, 2)
    cv2.putText(frame, "Powered by 3D Biomechanical Pose Estimation (MediaPipe Tasks)", (60, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1)

    items = [
        ("1. Bicep Curls", "AR Optimal Ghost, 3D Peak Squeeze & Extension Delta, Multi-View Tracking", GREEN),
        ("2. Squats", "AR Parallel Thigh Ghost, Hip Reticle, Chest Up Neutral Spine, Knee Cave", GREEN),
        ("3. Tricep Extensions", "AR Lockout & Stretch Ghost, Sagittal Profile, Tendon Safety ROM", GREEN),
        ("4. Push-ups", "Coming Soon in Next Module", (150, 150, 150)),
    ]

    y = 155
    for title, desc, color in items:
        draw_card(frame, 40, y, 760, 64, bg_color=(36, 36, 40), alpha=0.90, border_color=(60, 60, 65))
        cv2.putText(frame, title, (65, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        cv2.putText(frame, desc, (65, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1)
        y += 76


    cv2.putText(frame, "Press 1, 2, or 3 to select exercise  |  Press Q to quit", (40, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

    cv2.imshow(WINDOW_NAME, frame)


# ==============================================================================
# 1. BICEP CURLS (Dedicated Virtual Coach Engine)
# ==============================================================================

def run_bicep_curls():
    return bicep_trainer.run_bicep_trainer(CAMERA_INDEX)


# ==============================================================================
# 2. SQUATS (Dedicated Virtual Coach Engine)
# ==============================================================================

def run_squats():
    return squat_trainer.run_squat_trainer(CAMERA_INDEX)



# ==============================================================================
# 3. TRICEP EXTENSIONS (Dedicated Virtual Coach Engine)
# ==============================================================================

def run_tricep_extensions():
    return tricep_trainer.run_tricep_trainer(CAMERA_INDEX)



# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================

def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 600)

    while True:
        draw_menu()
        key = cv2.waitKey(0) & 0xFF

        if key == ord("1"):
            cv2.destroyAllWindows()
            result = run_bicep_curls()
            if result == "quit":
                break

        elif key == ord("2"):
            cv2.destroyAllWindows()
            result = run_squats()
            if result == "quit":
                break

        elif key == ord("3"):
            cv2.destroyAllWindows()
            result = run_tricep_extensions()
            if result == "quit":
                break

        elif key == ord("4"):
            print("Push-ups are scheduled for the next module release.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
