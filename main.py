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
    """Open camera with optimal driver and resolution."""
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(index)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        ("1. Bicep Curls", "Upper Arm Pinning, 3D Peak Contraction, Full Extension, Flare Check", GREEN),
        ("2. Squats", "3D Parallel Depth, Knee Valgus (Cave) Check, Torso Inclination", GREEN),
        ("3. Tricep Extensions", "3D Overhead Vertical Alignment, Tendon Protection, Lockout", GREEN),
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
# 1. BICEP CURLS (3D Camera-Invariant Engine)
# ==============================================================================

def run_bicep_curls():
    detector = PoseDetector()
    camera = get_camera(CAMERA_INDEX)
    if not camera.isOpened():
        print("Cannot open webcam.")
        detector.close()
        return "menu"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1080, 640)

    smoother = bio.EMASmoother(alpha=0.35)
    debouncer = bio.FeedbackDebouncer(required_frames=4, min_hold_seconds=1.2)
    rep_counter = bio.RepetitionCounter("BICEP_CURLS")

    prev_time = time.time()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            now = time.time()
            fps = 1.0 / max(1e-4, now - prev_time)
            prev_time = now

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]

            results = detector.detect_pose(frame)

            detected_issue = None
            feedback_msg = "Keep upper arms still at your sides"
            feedback_color = WHITE
            viewpoint = "FRONTAL"

            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks_2d = results.pose_landmarks[0]
                world_landmarks = results.pose_world_landmarks[0]

                # Auto-detect body viewpoint
                viewpoint, _ = bio.detect_viewpoint(world_landmarks)

                arm_data = []
                for side, s_idx, e_idx, w_idx, h_idx in ARMS:
                    if not (bio.is_landmark_valid(world_landmarks, s_idx) and
                            bio.is_landmark_valid(world_landmarks, e_idx) and
                            bio.is_landmark_valid(world_landmarks, w_idx)):
                        continue

                    # 3D Metric Coordinates in Meters
                    s_3d = bio.get_landmark_3d(world_landmarks, s_idx)
                    e_3d = bio.get_landmark_3d(world_landmarks, e_idx)
                    w_3d = bio.get_landmark_3d(world_landmarks, w_idx)
                    h_3d = bio.get_landmark_3d(world_landmarks, h_idx)

                    # 2D Screen Pixels for drawing overlay
                    s_2d = point_2d(landmarks_2d, s_idx, width, height)
                    e_2d = point_2d(landmarks_2d, e_idx, width, height)
                    w_2d = point_2d(landmarks_2d, w_idx, width, height)

                    # Camera-invariant 3D elbow flexion angle
                    raw_angle = bio.calculate_angle_3d(s_3d, e_3d, w_3d)
                    angle_3d = smoother.smooth(f"{side}_angle", raw_angle)

                    # 3D Upper-Arm Drift relative to torso
                    upper_arm_drift_raw = bio.calculate_angle_3d(e_3d, s_3d, h_3d)
                    arm_drift_3d = smoother.smooth(f"{side}_drift", upper_arm_drift_raw)

                    # 3D Lateral Flare (horizontal distance from shoulder line)
                    lateral_flare_dist = abs(e_3d[0] - s_3d[0])

                    # Form Error Checks
                    issue = None
                    if angle_3d < 36:
                        issue = "OVER-CURLED"
                    elif arm_drift_3d > 35:
                        issue = "ELBOW SWING"
                    elif lateral_flare_dist > 0.22:
                        issue = "ELBOW FLARE"

                    arm_data.append({
                        "side": side,
                        "s_2d": s_2d, "e_2d": e_2d, "w_2d": w_2d,
                        "angle_3d": angle_3d,
                        "good": (issue is None),
                        "issue": issue
                    })

                # Draw Target Guides first
                for arm in arm_data:
                    draw_arm_guide(frame, arm["s_2d"], arm["e_2d"], arm["w_2d"], arm["side"])

                # Draw Bones & Joint Angles
                for i, arm in enumerate(arm_data):
                    color = GREEN if arm["good"] else RED
                    draw_bone(frame, arm["s_2d"], arm["e_2d"], color, 4)
                    draw_bone(frame, arm["e_2d"], arm["w_2d"], color, 4)

                    label_y = 75 + i * 26
                    status_str = f"{arm['side']}: {'GOOD' if arm['good'] else arm['issue']} ({int(arm['angle_3d'])} deg)"
                    cv2.putText(frame, status_str, (16, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

                if arm_data:
                    avg_angle = sum(a["angle_3d"] for a in arm_data) / len(arm_data)
                    any_over_curled = any(a["issue"] == "OVER-CURLED" for a in arm_data)
                    any_swing = any(a["issue"] == "ELBOW SWING" for a in arm_data)
                    any_flare = any(a["issue"] == "ELBOW FLARE" for a in arm_data)

                    if any_over_curled:
                        detected_issue = "OVER-CURLED"
                    elif any_swing:
                        detected_issue = "ELBOW SWING"
                    elif any_flare:
                        detected_issue = "ELBOW FLARE"

                    # Update Repetition Counter
                    rep_stats = rep_counter.update(avg_angle, has_form_issue=(detected_issue is not None), now_time=now)

                    # Debounced Feedback Message
                    active_issue = debouncer.update(detected_issue, now_time=now)
                    if active_issue == "OVER-CURLED":
                        feedback_msg = "Over-curled! Maintain tension; don't rest wrists against shoulders"
                        feedback_color = RED
                    elif active_issue == "ELBOW SWING":
                        feedback_msg = "Pin elbows at your sides; stop swinging elbows forward"
                        feedback_color = RED
                    elif active_issue == "ELBOW FLARE":
                        feedback_msg = "Tuck elbows in; stop flaring outward"
                        feedback_color = RED
                    elif avg_angle <= 55:
                        feedback_msg = "Peak contraction! Squeeze and lower with control"
                        feedback_color = GREEN
                    elif avg_angle >= 142:
                        feedback_msg = "Full extension! Smoothly initiate next curl"
                        feedback_color = GREEN
                    else:
                        feedback_msg = "Good form! Keep a steady, controlled tempo"
                        feedback_color = GREEN
                else:
                    rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                    feedback_msg = "Position both arms clearly inside camera view"
                    feedback_color = YELLOW

            else:
                rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                feedback_msg = "PERSON NOT DETECTED"
                feedback_color = RED

            # Render Modern HUD
            draw_header_hud(frame, "BICEP CURLS (3D)", viewpoint, rep_stats, fps)
            draw_bottom_hud(frame, feedback_msg, feedback_color, stage_text=rep_stats["stage"])

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return "quit"
            if key == ord("m"):
                return "menu"

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()


# ==============================================================================
# 2. SQUATS (3D Camera-Invariant Depth & Valgus Engine)
# ==============================================================================

def run_squats():
    detector = PoseDetector()
    camera = get_camera(CAMERA_INDEX)
    if not camera.isOpened():
        print("Cannot open webcam.")
        detector.close()
        return "menu"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1080, 640)

    smoother = bio.EMASmoother(alpha=0.35)
    debouncer = bio.FeedbackDebouncer(required_frames=4, min_hold_seconds=1.2)
    rep_counter = bio.RepetitionCounter("SQUATS")

    prev_time = time.time()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            now = time.time()
            fps = 1.0 / max(1e-4, now - prev_time)
            prev_time = now

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]

            results = detector.detect_pose(frame)

            detected_issue = None
            feedback_msg = "Stand in frame with full legs visible"
            feedback_color = WHITE
            viewpoint = "FRONTAL"

            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks_2d = results.pose_landmarks[0]
                world_landmarks = results.pose_world_landmarks[0]

                # Auto-detect body viewpoint (Frontal / Side / Oblique)
                viewpoint, _ = bio.detect_viewpoint(world_landmarks)

                legs_data = []
                for side, h_idx, k_idx, a_idx, s_idx in LEGS:
                    if not (bio.is_landmark_valid(world_landmarks, h_idx) and
                            bio.is_landmark_valid(world_landmarks, k_idx) and
                            bio.is_landmark_valid(world_landmarks, a_idx)):
                        continue

                    # 3D Metric Coordinates in Meters
                    h_3d = bio.get_landmark_3d(world_landmarks, h_idx)
                    k_3d = bio.get_landmark_3d(world_landmarks, k_idx)
                    a_3d = bio.get_landmark_3d(world_landmarks, a_idx)

                    # 2D Screen Pixels
                    h_2d = point_2d(landmarks_2d, h_idx, width, height)
                    k_2d = point_2d(landmarks_2d, k_idx, width, height)
                    a_2d = point_2d(landmarks_2d, a_idx, width, height)

                    # True 3D Knee Angle (flexion)
                    raw_knee_angle = bio.calculate_angle_3d(h_3d, k_3d, a_3d)
                    knee_angle_3d = smoother.smooth(f"{side}_knee", raw_knee_angle)

                    # 3D Thigh Angle to Horizontal
                    femur_angle = bio.calculate_femur_angle_to_horizontal_3d(h_3d, k_3d)

                    # 3D Torso / Back Angle from Vertical (0 deg = straight upright)
                    back_lean_3d = None
                    s_2d = None
                    if bio.is_landmark_valid(world_landmarks, s_idx):
                        s_3d = bio.get_landmark_3d(world_landmarks, s_idx)
                        s_2d = point_2d(landmarks_2d, s_idx, width, height)
                        raw_back = bio.calculate_inclination_from_vertical_3d(h_3d, s_3d)
                        back_lean_3d = smoother.smooth(f"{side}_back", raw_back)

                    legs_data.append({
                        "side": side,
                        "h_2d": h_2d, "k_2d": k_2d, "a_2d": a_2d, "s_2d": s_2d,
                        "h_3d": h_3d, "k_3d": k_3d, "a_3d": a_3d,
                        "knee_angle_3d": knee_angle_3d,
                        "femur_angle": femur_angle,
                        "back_lean_3d": back_lean_3d,
                        "good": True,
                        "issue": None
                    })

                # Check Knee Cave (Valgus) in Frontal / Oblique views
                knee_cave = False
                left_leg = next((l for l in legs_data if l["side"] == "LEFT"), None)
                right_leg = next((l for l in legs_data if l["side"] == "RIGHT"), None)
                if left_leg and right_leg and ("FRONTAL" in viewpoint or "OBLIQUE" in viewpoint):
                    knee_dist_3d = np.linalg.norm(left_leg["k_3d"] - right_leg["k_3d"])
                    ankle_dist_3d = np.linalg.norm(left_leg["a_3d"] - right_leg["a_3d"])
                    avg_knee_angle = (left_leg["knee_angle_3d"] + right_leg["knee_angle_3d"]) / 2.0
                    # If knees bend into squat and collapse inward significantly relative to ankles
                    if avg_knee_angle < 135 and ankle_dist_3d > 0.25:
                        if knee_dist_3d < (ankle_dist_3d * 0.72):
                            knee_cave = True

                # Draw Guides first
                for leg in legs_data:
                    draw_squat_guide(frame, leg["h_2d"], leg["k_2d"], leg["a_2d"], leg["side"])

                # Posture Evaluation for each leg
                for i, leg in enumerate(legs_data):
                    issue = None
                    if leg["knee_angle_3d"] < 68:
                        issue = "TOO DEEP"
                    elif leg["back_lean_3d"] is not None and leg["back_lean_3d"] > 48:
                        issue = "CHEST DOWN"
                    elif knee_cave:
                        issue = "KNEE CAVE"

                    leg["good"] = (issue is None)
                    leg["issue"] = issue

                    color = GREEN if leg["good"] else RED
                    draw_bone(frame, leg["h_2d"], leg["k_2d"], color, 4)
                    draw_bone(frame, leg["k_2d"], leg["a_2d"], color, 4)
                    if leg["s_2d"]:
                        draw_bone(frame, leg["s_2d"], leg["h_2d"], color, 3)

                    label_y = 75 + i * 26
                    status_str = f"{leg['side']}: {'GOOD' if leg['good'] else leg['issue']} ({int(leg['knee_angle_3d'])} deg)"
                    cv2.putText(frame, status_str, (16, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

                if legs_data:
                    # If viewed from the side, prioritize the facing leg
                    if viewpoint == "SIDE_LEFT":
                        primary_leg = next((l for l in legs_data if l["side"] == "LEFT"), legs_data[0])
                    elif viewpoint == "SIDE_RIGHT":
                        primary_leg = next((l for l in legs_data if l["side"] == "RIGHT"), legs_data[0])
                    else:
                        primary_leg = legs_data[0]

                    avg_knee_angle = sum(l["knee_angle_3d"] for l in legs_data) / len(legs_data)
                    any_too_deep = any(l["issue"] == "TOO DEEP" for l in legs_data)
                    any_chest_down = any(l["issue"] == "CHEST DOWN" for l in legs_data)

                    if any_chest_down:
                        detected_issue = "CHEST DOWN"
                    elif knee_cave:
                        detected_issue = "KNEE CAVE"
                    elif any_too_deep:
                        detected_issue = "TOO DEEP"

                    # Rep counter update
                    rep_stats = rep_counter.update(primary_leg["knee_angle_3d"], has_form_issue=(detected_issue is not None), now_time=now)

                    # Debounced Feedback
                    active_issue = debouncer.update(detected_issue, now_time=now)
                    if active_issue == "CHEST DOWN":
                        feedback_msg = "Keep chest up and spine neutral! Don't bend forward"
                        feedback_color = RED
                    elif active_issue == "KNEE CAVE":
                        feedback_msg = "Push knees outward in line with toes! Avoid knee collapse"
                        feedback_color = RED
                    elif active_issue == "TOO DEEP":
                        feedback_msg = "Squatting too deep! Stop at parallel (~90 deg) to protect knees"
                        feedback_color = RED
                    elif primary_leg["knee_angle_3d"] <= 100:
                        feedback_msg = "Excellent squat depth! Drive through midfoot to stand up"
                        feedback_color = GREEN
                    elif primary_leg["knee_angle_3d"] <= 138:
                        feedback_msg = "Squat lower until thighs are parallel to ground"
                        feedback_color = YELLOW
                    else:
                        feedback_msg = "Standing tall. Break at hips and knees simultaneously"
                        feedback_color = GREEN
                else:
                    rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                    feedback_msg = "Step back so hips, knees, and ankles are clearly visible"
                    feedback_color = YELLOW

            else:
                rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                feedback_msg = "PERSON NOT DETECTED"
                feedback_color = RED

            # Render Modern HUD
            draw_header_hud(frame, "SQUATS (3D)", viewpoint, rep_stats, fps)
            draw_bottom_hud(frame, feedback_msg, feedback_color, stage_text=rep_stats["stage"])

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return "quit"
            if key == ord("m"):
                return "menu"

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()


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
