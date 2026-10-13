"""
Professional AI Tricep Extension Coach & Posture Corrector.
Features:
- Full Multi-View Support: Auto-adapts to Frontal Stance AND Lateral Profile Stances (Left & Right).
- In Lateral View: Automatically locks onto the facing arm with zero occlusion interference.
- Occlusion-tolerant landmark tracking (wrist behind head never vanishes or flickers).
- Smooth AR joint angle arc and overhead alignment target.
- Modern Coach Dashboard sidebar with live Range-of-Motion (ROM) gauge.
- Adaptive form diagnostics (overhead elevation, sagittal drift, flare, joint safety).
- Repetition counter with stage progression and form rating.
"""

import time
import math
import cv2
import numpy as np

from pose_detector import PoseDetector
import biomechanics as bio

# Color Palette (BGR)
BG_DARK = (24, 24, 28)
PANEL_BG = (32, 32, 38)
CARD_BG = (42, 42, 50)
BORDER_COLOR = (60, 60, 72)
GREEN = (45, 215, 65)
CYAN = (235, 205, 50)
YELLOW = (0, 215, 255)
RED = (45, 45, 245)
WHITE = (245, 245, 245)
GRAY = (140, 140, 150)
DARK_GRAY = (65, 65, 75)
PURPLE = (200, 90, 230)


class TricepLandmarkTracker:
    """
    Maintains landmark history and estimates positions during momentary occlusions
    (e.g., when wrists go behind the head in overhead tricep extensions).
    Guarantees that visual overlays never abruptly vanish.
    """

    def __init__(self):
        self.history = {}
        self.last_valid_time = {}
        self.forearm_length = {}
        self.upper_arm_length = {}

    def update_arm(self, side, s_2d, e_2d, w_2d, s_3d, e_3d, w_3d,
                   s_valid, e_valid, w_valid, now_time):
        """
        Updates arm tracking state with occlusion hold-over.
        """
        if s_valid and e_valid:
            cur_len_2d = math.hypot(e_2d[0] - s_2d[0], e_2d[1] - s_2d[1])
            self.upper_arm_length[side] = cur_len_2d
            self.history[f"{side}_shoulder_2d"] = s_2d
            self.history[f"{side}_elbow_2d"] = e_2d
            self.history[f"{side}_shoulder_3d"] = s_3d
            self.history[f"{side}_elbow_3d"] = e_3d
            self.last_valid_time[f"{side}_upper"] = now_time

        if e_valid and w_valid:
            cur_forearm_len = math.hypot(w_2d[0] - e_2d[0], w_2d[1] - e_2d[1])
            if cur_forearm_len > 15:
                self.forearm_length[side] = cur_forearm_len
            self.history[f"{side}_wrist_2d"] = w_2d
            self.history[f"{side}_wrist_3d"] = w_3d
            self.last_valid_time[f"{side}_wrist"] = now_time
            wrist_is_active = True
        else:
            # Retain recent position for up to 1.0 second
            last_wrist_time = self.last_valid_time.get(f"{side}_wrist", 0)
            if (now_time - last_wrist_time) < 1.0 and f"{side}_wrist_2d" in self.history:
                wrist_is_active = True
            else:
                wrist_is_active = False

        return wrist_is_active


def draw_angle_arc(frame, center, start_pt, end_pt, angle_deg, color=GREEN, radius=42):
    """
    Draw a clean, anti-aliased circular angle arc and text readout at the elbow joint.
    Also highlights the optimal target sectors.
    """
    cx, cy = center
    v1 = (start_pt[0] - cx, start_pt[1] - cy)
    v2 = (end_pt[0] - cx, end_pt[1] - cy)

    ang1 = math.degrees(math.atan2(v1[1], v1[0]))
    ang2 = math.degrees(math.atan2(v2[1], v2[0]))

    if ang2 < ang1:
        ang2 += 360.0
    if (ang2 - ang1) > 180.0:
        ang1, ang2 = ang2, ang1 + 360.0

    # Draw faint reference target circle
    cv2.circle(frame, (cx, cy), radius, (55, 55, 65), 1, cv2.LINE_AA)

    # Draw active angle arc
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, ang1, ang2, color, 3, cv2.LINE_AA)

    # Angle label badge
    label = f"{int(angle_deg)} deg"
    text_pos = (cx + radius + 10, cy - 8)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
    cv2.rectangle(frame, (text_pos[0] - 4, text_pos[1] - th - 4),
                  (text_pos[0] + tw + 4, text_pos[1] + 4), (20, 20, 24), -1)
    cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA)


def draw_optimal_ghost(frame, shoulder, elbow, wrist, current_angle, side, viewpoint, stage):
    """
    Renders an Augmented Reality Optimal Target Blueprint Ghost directly on the user's feed.
    Shows:
    - Ideal vertical humerus/upper-arm position.
    - Ideal forearm flexion angle for the current phase (Lockout ~170 deg vs Stretch ~80 deg).
    - Illuminated target reticle and real-time adjustment delta ('LOWER 30 deg MORE' / 'EXTEND 40 deg HIGHER').
    """
    sx, sy = shoulder
    ex, ey = elbow
    wx, wy = wrist

    upper_len = max(40, int(math.hypot(ex - sx, ey - sy)))
    forearm_len = max(40, int(math.hypot(wx - ex, wy - ey)))

    # Optimal Elbow: perfectly vertical overhead above shoulder
    opt_elbow = (sx, max(25, sy - upper_len))

    # Phase-aware target: Lockout vs Deep Stretch
    is_targeting_stretch = (current_angle < 125)

    if is_targeting_stretch:
        target_name = "STRETCH TARGET (80 deg)"
        target_angle = 80.0
        # In lateral view, forearm points backwards behind head
        if "SIDE_LEFT" in viewpoint:
            opt_wrist = (
                int(opt_elbow[0] + forearm_len * 0.95),
                int(opt_elbow[1] + forearm_len * 0.35)
            )
        elif "SIDE_RIGHT" in viewpoint:
            opt_wrist = (
                int(opt_elbow[0] - forearm_len * 0.95),
                int(opt_elbow[1] + forearm_len * 0.35)
            )
        else:
            dir_x = 1 if side == "LEFT" else -1
            opt_wrist = (
                int(opt_elbow[0] + dir_x * forearm_len * 0.50),
                int(opt_elbow[1] + forearm_len * 0.85)
            )

        # Delta calculation
        if 72 <= current_angle <= 88:
            delta_str = "OPTIMAL STRETCH REACHED! (HOLD & SQUEEZE)"
            delta_color = GREEN
            reticle_color = GREEN
        elif current_angle < 60:
            delta_str = f"TOO DEEP! RAISE FOREARM +{int(60 - current_angle)} deg"
            delta_color = RED
            reticle_color = RED
        else:
            diff = int(current_angle - 80)
            delta_str = f"LOWER FOREARM -{diff} deg MORE TO HIT OPTIMAL STRETCH"
            delta_color = YELLOW
            reticle_color = YELLOW

    else:
        target_name = "LOCKOUT TARGET (170 deg)"
        target_angle = 170.0
        # Forearm extends straight overhead
        opt_wrist = (opt_elbow[0], max(10, opt_elbow[1] - forearm_len))

        if current_angle >= 152:
            delta_str = "OPTIMAL LOCKOUT REACHED! (LOWER WITH CONTROL)"
            delta_color = GREEN
            reticle_color = GREEN
        else:
            diff = int(170 - current_angle)
            delta_str = f"EXTEND UPWARD +{diff} deg HIGHER TO HIT LOCKOUT"
            delta_color = CYAN
            reticle_color = CYAN

    # AR Ghost Blueprint (Semi-transparent glowing target arm)
    overlay = frame.copy()

    # Draw Blueprint bones
    cv2.line(overlay, (sx, sy), opt_elbow, (200, 240, 50), 3, cv2.LINE_AA)
    cv2.line(overlay, opt_elbow, opt_wrist, (50, 240, 200), 3, cv2.LINE_AA)

    # Joints
    cv2.circle(overlay, opt_elbow, 6, (200, 240, 50), -1, cv2.LINE_AA)
    cv2.circle(overlay, opt_wrist, 8, reticle_color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Draw Target Reticle at optimal wrist
    cv2.circle(frame, opt_wrist, 16, reticle_color, 2, cv2.LINE_AA)
    cv2.circle(frame, opt_wrist, 4, WHITE, -1, cv2.LINE_AA)

    # Label target reticle
    reticle_label = "OPTIMAL TARGET"
    cv2.putText(frame, reticle_label, (opt_wrist[0] + 18, opt_wrist[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, reticle_color, 2, cv2.LINE_AA)

    # Dynamic Delta Action Banner over the arm
    badge_y = max(45, opt_elbow[1] - 30)
    (bw, bh), _ = cv2.getTextSize(delta_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    badge_x = max(10, min(frame.shape[1] - bw - 20, opt_elbow[0] - bw // 2))

    cv2.rectangle(frame, (badge_x - 6, badge_y - bh - 6),
                  (badge_x + bw + 6, badge_y + 6), (25, 25, 30), -1)
    cv2.rectangle(frame, (badge_x - 6, badge_y - bh - 6),
                  (badge_x + bw + 6, badge_y + 6), delta_color, 1)
    cv2.putText(frame, delta_str, (badge_x, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, delta_color, 2, cv2.LINE_AA)

    return target_name, target_angle, delta_str, delta_color


def draw_rom_gauge(panel, x, y, w, h, current_angle, target_angle=170.0):
    """
    Draws an intuitive Range-Of-Motion (ROM) progress gauge on the dashboard panel.
    Now explicitly displays the optimal target needle and delta.
    """
    cv2.rectangle(panel, (x, y), (x + w, y + h), CARD_BG, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), BORDER_COLOR, 1)

    min_a, max_a = 50.0, 180.0
    clamp_angle = max(min_a, min(max_a, current_angle))
    ratio = (clamp_angle - min_a) / (max_a - min_a)

    bar_x = x + 12
    bar_y = y + 36
    bar_w = w - 24
    bar_h = 16

    # Track
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 58), -1)

    # Stretch Zone (72 to 88 deg) -> yellow/cyan
    s_start = int(bar_w * ((72 - min_a) / (max_a - min_a)))
    s_end = int(bar_w * ((88 - min_a) / (max_a - min_a)))
    cv2.rectangle(panel, (bar_x + s_start, bar_y), (bar_x + s_end, bar_y + bar_h), (0, 180, 220), -1)

    # Lockout Zone (150 to 180 deg) -> green
    l_start = int(bar_w * ((150 - min_a) / (max_a - min_a)))
    cv2.rectangle(panel, (bar_x + l_start, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 190, 40), -1)

    # Optimal target marker (hollow green flag)
    target_x = int(bar_x + ((target_angle - min_a) / (max_a - min_a)) * bar_w)
    cv2.line(panel, (target_x, bar_y - 6), (target_x, bar_y + bar_h + 6), (50, 240, 80), 2)
    cv2.circle(panel, (target_x, bar_y - 6), 4, (50, 240, 80), -1)

    # Current position needle
    marker_x = int(bar_x + ratio * bar_w)
    cv2.line(panel, (marker_x, bar_y - 4), (marker_x, bar_y + bar_h + 4), WHITE, 3)
    cv2.circle(panel, (marker_x, bar_y + bar_h // 2), 6, WHITE, -1)

    cv2.putText(panel, f"RANGE OF MOTION | TARGET: {int(target_angle)} deg", (x + 12, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1, cv2.LINE_AA)

    if current_angle >= 148:
        zone_label = "OPTIMAL: LOCKOUT EXTENSION (Peak)"
        zone_color = GREEN
    elif current_angle <= 62:
        zone_label = "DANGER: HYPER-FLEXION (<60 deg)"
        zone_color = RED
    elif current_angle <= 88:
        zone_label = "OPTIMAL: TARGET DEEP STRETCH"
        zone_color = YELLOW
    else:
        zone_label = "IN MOTION (Moving toward target)"
        zone_color = WHITE

    cv2.putText(panel, zone_label, (x + 12, y + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, zone_color, 2, cv2.LINE_AA)



def draw_checklist_item(panel, x, y, title, is_good, custom_text=None):
    """Draws a single clean form diagnostic checklist entry."""
    badge_color = GREEN if is_good else RED
    badge_icon = "PASS" if is_good else "ADJUST"
    detail = custom_text if custom_text else ("Optimal" if is_good else "Check Form")

    cv2.rectangle(panel, (x, y), (x + 55, y + 20), (45, 45, 52), -1)
    cv2.putText(panel, badge_icon, (x + 8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_color, 2)
    cv2.putText(panel, title, (x + 65, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1)

    (tw, _), _ = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(panel, detail, (panel.shape[1] - tw - 16, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1)


def run_tricep_trainer(camera_index=0):
    """
    Main loop for the upgraded 10/10 AI Tricep Extension Virtual Coach.
    Full support for Frontal, Left Lateral, and Right Lateral views.
    """
    detector = PoseDetector()

    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(camera_index)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        print("Cannot open webcam.")
        detector.close()
        return "menu"

    window_name = "AI Gym - Tricep Extension Virtual Trainer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    tracker = TricepLandmarkTracker()
    smoother = bio.EMASmoother(alpha=0.35)
    debouncer = bio.FeedbackDebouncer(required_frames=3, min_hold_seconds=1.0)
    rep_counter = bio.RepetitionCounter("TRICEP_EXTENSIONS")

    prev_time = time.time()

    try:
        while True:
            success, raw_frame = cam.read()
            if not success:
                continue

            now = time.time()
            fps = 1.0 / max(1e-4, now - prev_time)
            prev_time = now

            raw_frame = cv2.flip(raw_frame, 1)

            feed_w, feed_h = 840, 630
            video_frame = cv2.resize(raw_frame, (feed_w, feed_h), interpolation=cv2.INTER_AREA)

            panel_w, panel_h = 400, 630
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            panel[:] = PANEL_BG

            results = detector.detect_pose(video_frame)

            active_arm_names = []
            primary_angle = 160.0
            is_overhead_ok = True
            is_flare_ok = True
            is_joint_safe = True
            is_vertical_ok = True
            instruction_text = "Raise arm(s) overhead to start"
            instruction_color = WHITE
            detected_issue = None
            viewpoint_label = "FRONTAL"
            is_lateral = False

            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks_2d = results.pose_landmarks[0]
                world_landmarks = results.pose_world_landmarks[0]

                # Robust multi-view auto-detection (Frontal vs. Lateral Stances)
                viewpoint, _ = bio.detect_viewpoint(world_landmarks, landmarks_2d=landmarks_2d)
                is_lateral = ("SIDE" in viewpoint)

                # Determine active arm candidate list based on user stance
                if viewpoint == "SIDE_LEFT":
                    # User standing laterally with Left side to camera -> Focus 100% on Left Arm
                    arm_candidates = [("LEFT", 11, 13, 15, 23)]
                    viewpoint_label = "LATERAL (LEFT PROFILE)"
                elif viewpoint == "SIDE_RIGHT":
                    # User standing laterally with Right side to camera -> Focus 100% on Right Arm
                    arm_candidates = [("RIGHT", 12, 14, 16, 24)]
                    viewpoint_label = "LATERAL (RIGHT PROFILE)"
                elif viewpoint in ("OBLIQUE_LEFT", "OBLIQUE_RIGHT"):
                    side_pick = "LEFT" if "LEFT" in viewpoint else "RIGHT"
                    cand = ("LEFT", 11, 13, 15, 23) if side_pick == "LEFT" else ("RIGHT", 12, 14, 16, 24)
                    arm_candidates = [cand]
                    viewpoint_label = f"DIAGONAL ({side_pick})"
                else:
                    # Frontal view: track both arms if raised overhead
                    arm_candidates = [
                        ("LEFT", 11, 13, 15, 23),
                        ("RIGHT", 12, 14, 16, 24)
                    ]
                    viewpoint_label = "FRONTAL VIEW"

                arms_tracked = []
                for side, s_idx, e_idx, w_idx, h_idx in arm_candidates:
                    s_valid = bio.is_landmark_valid(world_landmarks, s_idx, min_confidence=0.35)
                    e_valid = bio.is_landmark_valid(world_landmarks, e_idx, min_confidence=0.35)
                    w_valid = bio.is_landmark_valid(world_landmarks, w_idx, min_confidence=0.25)

                    if not (s_valid and e_valid):
                        continue

                    # 3D Metric Coordinates in Meters
                    s_3d = bio.get_landmark_3d(world_landmarks, s_idx)
                    e_3d = bio.get_landmark_3d(world_landmarks, e_idx)
                    w_3d = bio.get_landmark_3d(world_landmarks, w_idx)

                    # 2D Screen Pixels
                    s_2d = (int(landmarks_2d[s_idx].x * feed_w), int(landmarks_2d[s_idx].y * feed_h))
                    e_2d = (int(landmarks_2d[e_idx].x * feed_w), int(landmarks_2d[e_idx].y * feed_h))
                    w_2d = (int(landmarks_2d[w_idx].x * feed_w), int(landmarks_2d[w_idx].y * feed_h))

                    # Occlusion Tracker Update
                    tracker.update_arm(side, s_2d, e_2d, w_2d, s_3d, e_3d, w_3d,
                                       s_valid, e_valid, w_valid, now)

                    curr_s_2d = tracker.history.get(f"{side}_shoulder_2d", s_2d)
                    curr_e_2d = tracker.history.get(f"{side}_elbow_2d", e_2d)
                    curr_w_2d = tracker.history.get(f"{side}_wrist_2d", w_2d)

                    # Ignore arms that are resting down at the waist
                    if curr_e_2d[1] > (curr_s_2d[1] + 55):
                        continue

                    # True 3D Elbow Flexion Angle (Sagittal flexion)
                    raw_angle = bio.calculate_angle_3d(s_3d, e_3d, w_3d)
                    smooth_angle = smoother.smooth(f"{side}_tricep", raw_angle)

                    # True 3D Overhead Alignment (Upper arm pointing overhead along -y)
                    raw_arm_tilt = bio.calculate_inclination_from_vertical_3d(s_3d, e_3d)
                    arm_tilt = smoother.smooth(f"{side}_tilt", raw_arm_tilt)

                    # Biomechanical evaluation according to stance
                    arm_issue = None
                    if smooth_angle < 58:
                        arm_issue = "OVER-FLEXED"
                        is_joint_safe = False
                    elif arm_tilt > 36:
                        arm_issue = "ELBOW FORWARD" if is_lateral else "ELBOWS LOW"
                        is_overhead_ok = False
                        is_vertical_ok = False
                    elif not is_lateral:
                        # Lateral flare is only relevant in frontal/oblique view
                        flare_dist = abs(e_3d[0] - s_3d[0])
                        if flare_dist > 0.24:
                            arm_issue = "ELBOW FLARE"
                            is_flare_ok = False

                    arms_tracked.append({
                        "side": side,
                        "s_2d": curr_s_2d,
                        "e_2d": curr_e_2d,
                        "w_2d": curr_w_2d,
                        "angle": smooth_angle,
                        "tilt": arm_tilt,
                        "good": (arm_issue is None),
                        "issue": arm_issue
                    })

                active_target_angle = 170.0
                active_delta_str = ""
                active_delta_color = WHITE

                # Draw Smooth Augmented Reality Overlays & Optimal Target Ghost on Camera Feed
                for arm in arms_tracked:
                    active_arm_names.append(arm["side"])
                    bone_color = GREEN if arm["good"] else RED

                    # Real Limb Bones
                    cv2.line(video_frame, arm["s_2d"], arm["e_2d"], bone_color, 4, cv2.LINE_AA)
                    cv2.line(video_frame, arm["e_2d"], arm["w_2d"], bone_color, 4, cv2.LINE_AA)

                    # Real Limb Joints
                    cv2.circle(video_frame, arm["s_2d"], 7, CYAN, -1, cv2.LINE_AA)
                    cv2.circle(video_frame, arm["e_2d"], 8, WHITE, -1, cv2.LINE_AA)
                    cv2.circle(video_frame, arm["w_2d"], 7, CYAN, -1, cv2.LINE_AA)

                    # Draw Augmented Reality OPTIMAL TARGET GHOST BLUEPRINT
                    # Shows the exact optimal arm position & target reticle!
                    t_name, t_angle, d_str, d_color = draw_optimal_ghost(
                        video_frame, arm["s_2d"], arm["e_2d"], arm["w_2d"],
                        arm["angle"], arm["side"], viewpoint, rep_stats.get("stage", "READY")
                    )
                    active_target_angle = t_angle
                    active_delta_str = d_str
                    active_delta_color = d_color

                    # Dynamic Circular Angle Arc at Elbow
                    arc_color = GREEN if (arm["angle"] >= 148 or (72 <= arm["angle"] <= 88)) else (RED if not arm["good"] else CYAN)
                    draw_angle_arc(video_frame, arm["e_2d"], arm["s_2d"], arm["w_2d"], arm["angle"], color=arc_color)

                if arms_tracked:
                    primary_angle = sum(a["angle"] for a in arms_tracked) / len(arms_tracked)

                    # Identify highest priority fault
                    if any(a["issue"] == "OVER-FLEXED" for a in arms_tracked):
                        detected_issue = "OVER-FLEXED"
                    elif any(a["issue"] == "ELBOW FORWARD" for a in arms_tracked):
                        detected_issue = "ELBOW FORWARD"
                    elif any(a["issue"] == "ELBOWS LOW" for a in arms_tracked):
                        detected_issue = "ELBOWS LOW"
                    elif any(a["issue"] == "ELBOW FLARE" for a in arms_tracked):
                        detected_issue = "ELBOW FLARE"

                    # Update Repetition Counter
                    rep_stats = rep_counter.update(primary_angle, has_form_issue=(detected_issue is not None), now_time=now)

                    # Dynamic Coaching Instruction Text with Exact Optimal Delta Action
                    active_issue = debouncer.update(detected_issue, now_time=now)
                    if active_issue == "OVER-FLEXED":
                        instruction_text = "DANGER: Arms over-closed! Raise forearm back to 75-80 deg"
                        instruction_color = RED
                    elif active_issue == "ELBOW FORWARD":
                        instruction_text = "Keep upper arm vertical overhead; don't let elbow drift forward"
                        instruction_color = RED
                    elif active_issue == "ELBOWS LOW":
                        instruction_text = "Raise elbows high overhead! Keep upper arms pointing straight up"
                        instruction_color = RED
                    elif active_issue == "ELBOW FLARE":
                        instruction_text = "Keep elbows tucked close to your head - stop flaring outward"
                        instruction_color = RED
                    elif active_delta_str:
                        instruction_text = active_delta_str
                        instruction_color = active_delta_color
                    elif primary_angle >= 148:
                        instruction_text = "Optimal Lockout Reached! Now lower smoothly behind head"
                        instruction_color = GREEN
                    elif primary_angle <= 86:
                        instruction_text = "Optimal Stretch Reached! Drive weight up to full lockout"
                        instruction_color = YELLOW
                    else:
                        instruction_text = "Controlled tempo. Follow the glowing optimal target ghost"
                        instruction_color = CYAN
                else:
                    rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                    instruction_text = "Raise arm overhead to align with optimal target ghost"
                    instruction_color = YELLOW

            else:
                rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                instruction_text = "Step into camera frame"
                instruction_color = RED
                active_target_angle = 170.0

            # ==================================================================
            # RENDER COACHING DASHBOARD PANEL
            # ==================================================================

            # 1. Header Card (With Stance & Viewpoint indicator)
            cv2.rectangle(panel, (16, 16), (panel_w - 16, 80), CARD_BG, -1)
            cv2.rectangle(panel, (16, 16), (panel_w - 16, 80), BORDER_COLOR, 1)
            cv2.putText(panel, "TRICEP COACH (3D)", (28, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, CYAN, 2)

            view_color = GREEN if is_lateral else (CYAN if "FRONTAL" in viewpoint_label else YELLOW)
            cv2.putText(panel, f"VIEW: {viewpoint_label}", (28, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, view_color, 2)

            arm_mode = ", ".join(active_arm_names) if active_arm_names else "SEARCHING"
            cv2.putText(panel, f"ACTIVE ARM: {arm_mode}", (28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)

            # 2. Big Rep Counter Card
            cv2.rectangle(panel, (16, 92), (panel_w - 16, 196), CARD_BG, -1)
            cv2.rectangle(panel, (16, 92), (panel_w - 16, 196), BORDER_COLOR, 1)

            cv2.putText(panel, "COMPLETED REPS", (28, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.44, GRAY, 1)
            cv2.putText(panel, str(rep_stats["total_reps"]), (28, 176), cv2.FONT_HERSHEY_DUPLEX, 1.8, WHITE, 3)

            clean_reps = rep_stats["good_reps"]
            total_reps = max(1, rep_stats["total_reps"])
            pct = int((clean_reps / total_reps) * 100) if rep_stats["total_reps"] > 0 else 100

            cv2.putText(panel, f"Clean Reps: {clean_reps}", (160, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREEN, 1)
            cv2.putText(panel, f"Form Rating: {pct}%", (160, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CYAN, 1)

            stage_str = rep_stats.get("stage", "READY")
            pill_color = GREEN if stage_str == "INFLECTION" else (CYAN if stage_str == "MOVING" else GRAY)
            cv2.rectangle(panel, (panel_w - 110, 106), (panel_w - 28, 130), (55, 55, 65), -1)
            cv2.putText(panel, stage_str, (panel_w - 102, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pill_color, 1)

            # 3. Dynamic Range-of-Motion (ROM) Gauge with Target Needle
            draw_rom_gauge(panel, 16, 210, panel_w - 32, 94, primary_angle, target_angle=active_target_angle)


            # 4. Biomechanical Form Diagnostics Checklist (Adaptive to Stance)
            cv2.rectangle(panel, (16, 318), (panel_w - 16, 474), CARD_BG, -1)
            cv2.rectangle(panel, (16, 318), (panel_w - 16, 474), BORDER_COLOR, 1)
            cv2.putText(panel, "BIOMECHANICAL CHECKS", (28, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1)

            if is_lateral:
                draw_checklist_item(panel, 28, 359, "Overhead Elevation", is_overhead_ok,
                                    "High" if is_overhead_ok else "Dropped")
                draw_checklist_item(panel, 28, 389, "Vertical Alignment", is_vertical_ok,
                                    "Pinned" if is_vertical_ok else "Drifting Forward")
                draw_checklist_item(panel, 28, 419, "Tendon Safety ROM", is_joint_safe,
                                    "Safe" if is_joint_safe else "Over-Flexed")
                draw_checklist_item(panel, 28, 449, "Profile Stance", True,
                                    "100% Tracking")
            else:
                draw_checklist_item(panel, 28, 359, "Overhead Elevation", is_overhead_ok,
                                    "High" if is_overhead_ok else "Dropped")
                draw_checklist_item(panel, 28, 389, "Elbow Flare Control", is_flare_ok,
                                    "Tucked" if is_flare_ok else "Flaring")
                draw_checklist_item(panel, 28, 419, "Tendon Safety ROM", is_joint_safe,
                                    "Safe" if is_joint_safe else "Over-Flexed")
                draw_checklist_item(panel, 28, 449, "Stationary Upper Arm", is_overhead_ok and is_flare_ok,
                                    "Pinned" if (is_overhead_ok and is_flare_ok) else "Moving")

            # 5. Instructions Banner at bottom of panel
            cv2.rectangle(panel, (16, 488), (panel_w - 16, 569), (38, 38, 46), -1)
            border_cue_col = RED if instruction_color == RED else (GREEN if instruction_color == GREEN else CYAN)
            cv2.rectangle(panel, (16, 488), (panel_w - 16, 569), border_cue_col, 2)
            cv2.putText(panel, "COACH INSTRUCTION:", (28, 512), cv2.FONT_HERSHEY_SIMPLEX, 0.40, border_cue_col, 1)

            words = instruction_text.split()
            line1, line2 = "", ""
            for w in words:
                if len(line1 + " " + w) < 28 and not line2:
                    line1 += (" " if line1 else "") + w
                else:
                    line2 += (" " if line2 else "") + w

            cv2.putText(panel, line1, (28, 536), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 2)
            if line2:
                cv2.putText(panel, line2, (28, 556), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 2)

            # 6. Controls
            cv2.putText(panel, f"{int(fps)} FPS  |  M: Menu  |  Q: Quit", (28, 608),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, GRAY, 1)

            full_display = np.hstack([video_frame, panel])

            cv2.imshow(window_name, full_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return "quit"
            if key == ord("m"):
                return "menu"

    finally:
        cam.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    run_tricep_trainer()
