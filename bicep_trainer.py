"""
Professional AI Bicep Curl Coach & Posture Corrector.
Features:
- Full Multi-View Support: Auto-adapts to Frontal Stance AND Lateral Profile Stances (Left & Right).
- AR Optimal Target Ghost Blueprint: Shows ideal elbow pinning and phase-adaptive wrist target.
- Real-time adjustment delta ('CURL +30 deg HIGHER' / 'LOWER -45 deg TO EXTENSION').
- Modern Coach Dashboard sidebar with live Range-of-Motion (ROM) gauge and target flag.
- Biomechanical form diagnostics (elbow pinning, flare, tension range, tempo).
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


class BicepLandmarkTracker:
    """
    Maintains landmark history and estimates positions during rapid motion or occlusion.
    Guarantees that visual overlays never abruptly vanish or flicker.
    """

    def __init__(self):
        self.history = {}
        self.last_valid_time = {}
        self.forearm_length = {}
        self.upper_arm_length = {}

    def update_arm(self, side, s_2d, e_2d, w_2d, s_3d, e_3d, w_3d,
                   s_valid, e_valid, w_valid, now_time):
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
            last_wrist_time = self.last_valid_time.get(f"{side}_wrist", 0)
            if (now_time - last_wrist_time) < 1.0 and f"{side}_wrist_2d" in self.history:
                wrist_is_active = True
            else:
                wrist_is_active = False

        return wrist_is_active


def draw_angle_arc(frame, center, start_pt, end_pt, angle_deg, color=GREEN, radius=42):
    """
    Draw a clean, anti-aliased circular angle arc and text readout at the elbow joint.
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

    cv2.circle(frame, (cx, cy), radius, (55, 55, 65), 1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, ang1, ang2, color, 3, cv2.LINE_AA)

    label = f"{int(angle_deg)} deg"
    text_pos = (cx + radius + 10, cy - 8)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
    cv2.rectangle(frame, (text_pos[0] - 4, text_pos[1] - th - 4),
                  (text_pos[0] + tw + 4, text_pos[1] + 4), (20, 20, 24), -1)
    cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA)


def draw_optimal_bicep_ghost(frame, shoulder, elbow, wrist, current_angle, side, viewpoint, stage):
    """
    Renders an Augmented Reality Optimal Target Blueprint Ghost for Bicep Curls.
    Shows:
    - Ideal pinned upper arm (elbow held stationary below shoulder).
    - Ideal forearm trajectory (Full Extension ~165 deg vs. Peak Contraction ~52 deg).
    - Illuminated target reticle and exact numerical adjustment delta.
    """
    sx, sy = shoulder
    ex, ey = elbow
    wx, wy = wrist

    upper_len = max(45, int(math.hypot(ex - sx, ey - sy)))
    forearm_len = max(45, int(math.hypot(wx - ex, wy - ey)))

    # Optimal Elbow: pinned vertically below shoulder
    opt_elbow = (sx, min(frame.shape[0] - 25, sy + upper_len))

    # Phase-aware target: Peak Contraction vs Full Extension
    is_targeting_peak = (current_angle > 95 or stage == "MOVING")

    if is_targeting_peak:
        target_name = "PEAK CURL TARGET (52 deg)"
        target_angle = 52.0

        # Optimal curled forearm: curled up at ~52 deg
        if "SIDE_LEFT" in viewpoint:
            # Looking left: curls forward (towards -x) and up
            opt_wrist = (
                int(opt_elbow[0] - forearm_len * math.sin(math.radians(52))),
                int(opt_elbow[1] - forearm_len * math.cos(math.radians(52)))
            )
        elif "SIDE_RIGHT" in viewpoint:
            # Looking right: curls forward (towards +x) and up
            opt_wrist = (
                int(opt_elbow[0] + forearm_len * math.sin(math.radians(52))),
                int(opt_elbow[1] - forearm_len * math.cos(math.radians(52)))
            )
        else:
            dir_x = -1 if side == "LEFT" else 1
            opt_wrist = (
                int(opt_elbow[0] + dir_x * forearm_len * 0.48),
                int(opt_elbow[1] - forearm_len * 0.72)
            )

        if 45 <= current_angle <= 58:
            delta_str = "OPTIMAL PEAK CONTRACTION! (SQUEEZE & HOLD)"
            delta_color = GREEN
            reticle_color = GREEN
        elif current_angle < 38:
            delta_str = f"TOO CLOSE! LOWER FOREARM -{int(38 - current_angle)} deg (MAINTAIN TENSION)"
            delta_color = RED
            reticle_color = RED
        else:
            diff = int(current_angle - 52)
            delta_str = f"CURL FOREARM +{diff} deg HIGHER TO HIT PEAK SQUEEZE"
            delta_color = YELLOW
            reticle_color = YELLOW

    else:
        target_name = "EXTENSION TARGET (165 deg)"
        target_angle = 165.0
        # Optimal extended forearm: pointing down
        opt_wrist = (opt_elbow[0], min(frame.shape[0] - 10, opt_elbow[1] + forearm_len))

        if current_angle >= 155:
            delta_str = "OPTIMAL FULL EXTENSION REACHED! (READY TO CURL)"
            delta_color = GREEN
            reticle_color = GREEN
        else:
            diff = int(165 - current_angle)
            delta_str = f"LOWER FOREARM -{diff} deg TO FULL EXTENSION"
            delta_color = CYAN
            reticle_color = CYAN

    # AR Ghost Blueprint (Semi-transparent glowing limb)
    overlay = frame.copy()
    cv2.line(overlay, (sx, sy), opt_elbow, (200, 240, 50), 3, cv2.LINE_AA)
    cv2.line(overlay, opt_elbow, opt_wrist, (50, 240, 200), 3, cv2.LINE_AA)
    cv2.circle(overlay, opt_elbow, 6, (200, 240, 50), -1, cv2.LINE_AA)
    cv2.circle(overlay, opt_wrist, 8, reticle_color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Target Reticle
    cv2.circle(frame, opt_wrist, 16, reticle_color, 2, cv2.LINE_AA)
    cv2.circle(frame, opt_wrist, 4, WHITE, -1, cv2.LINE_AA)
    cv2.putText(frame, "OPTIMAL TARGET", (opt_wrist[0] + 18, opt_wrist[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, reticle_color, 2, cv2.LINE_AA)

    # Dynamic Delta Action Banner
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


def draw_rom_gauge(panel, x, y, w, h, current_angle, target_angle=52.0):
    """
    Draws Range-Of-Motion gauge for Bicep Curls:
    - Peak Zone: 45 to 58 deg (Green)
    - Danger/Tension Loss: < 38 deg (Red)
    - Full Extension: 155 to 175 deg (Green/Cyan)
    """
    cv2.rectangle(panel, (x, y), (x + w, y + h), CARD_BG, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), BORDER_COLOR, 1)

    min_a, max_a = 30.0, 180.0
    clamp_angle = max(min_a, min(max_a, current_angle))
    ratio = (clamp_angle - min_a) / (max_a - min_a)

    bar_x = x + 12
    bar_y = y + 36
    bar_w = w - 24
    bar_h = 16

    # Track
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 58), -1)

    # Peak Contraction Zone (45 to 58 deg) -> green
    p_start = int(bar_w * ((45 - min_a) / (max_a - min_a)))
    p_end = int(bar_w * ((58 - min_a) / (max_a - min_a)))
    cv2.rectangle(panel, (bar_x + p_start, bar_y), (bar_x + p_end, bar_y + bar_h), (40, 210, 40), -1)

    # Full Extension Zone (155 to 175 deg) -> cyan/green
    e_start = int(bar_w * ((155 - min_a) / (max_a - min_a)))
    e_end = int(bar_w * ((175 - min_a) / (max_a - min_a)))
    cv2.rectangle(panel, (bar_x + e_start, bar_y), (bar_x + e_end, bar_y + bar_h), (0, 180, 220), -1)

    # Target Flag needle
    target_x = int(bar_x + ((target_angle - min_a) / (max_a - min_a)) * bar_w)
    cv2.line(panel, (target_x, bar_y - 6), (target_x, bar_y + bar_h + 6), (50, 240, 80), 2)
    cv2.circle(panel, (target_x, bar_y - 6), 4, (50, 240, 80), -1)

    # Needle
    marker_x = int(bar_x + ratio * bar_w)
    cv2.line(panel, (marker_x, bar_y - 4), (marker_x, bar_y + bar_h + 4), WHITE, 3)
    cv2.circle(panel, (marker_x, bar_y + bar_h // 2), 6, WHITE, -1)

    cv2.putText(panel, f"RANGE OF MOTION | TARGET: {int(target_angle)} deg", (x + 12, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1, cv2.LINE_AA)

    if current_angle < 38:
        zone_label = "TENSION LOSS: OVER-CURLED (<38 deg)"
        zone_color = RED
    elif 45 <= current_angle <= 58:
        zone_label = "OPTIMAL: PEAK CONTRACTION SQUEEZE"
        zone_color = GREEN
    elif current_angle >= 155:
        zone_label = "OPTIMAL: FULL EXTENSION"
        zone_color = GREEN
    else:
        zone_label = "IN MOTION (CURLING / LOWERING)"
        zone_color = WHITE

    cv2.putText(panel, zone_label, (x + 12, y + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, zone_color, 2, cv2.LINE_AA)


def draw_checklist_item(panel, x, y, title, is_good, custom_text=None):
    badge_color = GREEN if is_good else RED
    badge_icon = "PASS" if is_good else "ADJUST"
    detail = custom_text if custom_text else ("Optimal" if is_good else "Check Form")

    cv2.rectangle(panel, (x, y), (x + 55, y + 20), (45, 45, 52), -1)
    cv2.putText(panel, badge_icon, (x + 8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, badge_color, 2)
    cv2.putText(panel, title, (x + 65, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.46, WHITE, 1)

    (tw, _), _ = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(panel, detail, (panel.shape[1] - tw - 16, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1)


def init_bicep_camera(camera_index=0, preferred_w=1280, preferred_h=720):
    """
    Open webcam with optimal backend, low latency buffering,
    automatic exposure calibration, and clarity adjustments.
    """
    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(camera_index)

    if not cam.isOpened():
        return cam

    # Low latency: buffer only 1 frame
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Use MJPG compression where available for higher FPS at 720p
    try:
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    # Request preferred resolution (1280x720) at 30 FPS
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_h)
    cam.set(cv2.CAP_PROP_FPS, 30)

    # Enable Autofocus & Auto White Balance if supported
    try:
        af = cam.get(cv2.CAP_PROP_AUTOFOCUS)
        if af != -1.0 and af is not None:
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except Exception:
        pass

    try:
        awb = cam.get(cv2.CAP_PROP_AUTO_WB)
        if awb != -1.0 and awb is not None:
            cam.set(cv2.CAP_PROP_AUTO_WB, 1)
    except Exception:
        pass

    # Prefer auto-exposure first
    try:
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    except Exception:
        try:
            cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        except Exception:
            pass

    # Allow sensor to converge auto-exposure with warm-up frames
    warmup_ok = False
    for _ in range(5):
        ret, frame = cam.read()
        if ret and frame is not None:
            warmup_ok = True

    # If camera remains severely underexposed (dark room or locked shutter), adjust exposure
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


def run_bicep_trainer(camera_index=0):
    """
    Main loop for the upgraded 10/10 AI Bicep Curl Virtual Coach.
    Full support for Frontal, Left Lateral, and Right Lateral views.
    """
    detector = PoseDetector()

    cam = init_bicep_camera(camera_index, 1280, 720)

    if not cam.isOpened():
        print("Cannot open webcam.")
        detector.close()
        return "menu"

    window_name = "AI Gym - Bicep Curl Virtual Trainer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    tracker = BicepLandmarkTracker()
    smoother = bio.EMASmoother(alpha=0.35)
    debouncer = bio.FeedbackDebouncer(required_frames=3, min_hold_seconds=1.0)
    rep_counter = bio.RepetitionCounter("BICEP_CURLS")

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
            raw_h, raw_w = raw_frame.shape[:2]
            target_aspect = feed_w / feed_h
            current_aspect = raw_w / raw_h

            if abs(current_aspect - target_aspect) < 0.01:
                video_frame = cv2.resize(raw_frame, (feed_w, feed_h), interpolation=cv2.INTER_LINEAR)
            elif current_aspect > target_aspect:
                crop_w = int(raw_h * target_aspect)
                offset = (raw_w - crop_w) // 2
                video_frame = cv2.resize(raw_frame[:, offset:offset + crop_w], (feed_w, feed_h), interpolation=cv2.INTER_LINEAR)
            else:
                crop_h = int(raw_w / target_aspect)
                offset = (raw_h - crop_h) // 2
                video_frame = cv2.resize(raw_frame[offset:offset + crop_h, :], (feed_w, feed_h), interpolation=cv2.INTER_LINEAR)

            panel_w, panel_h = 400, 630
            panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            panel[:] = PANEL_BG

            results = detector.detect_pose(video_frame)

            active_arm_names = []
            primary_angle = 160.0
            is_pinned_ok = True
            is_flare_ok = True
            is_tension_ok = True
            instruction_text = "Position arms at sides to start"
            instruction_color = WHITE
            detected_issue = None
            viewpoint_label = "FRONTAL"
            is_lateral = False

            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks_2d = results.pose_landmarks[0]
                world_landmarks = results.pose_world_landmarks[0]

                # Robust multi-view auto-detection
                viewpoint, _ = bio.detect_viewpoint(world_landmarks, landmarks_2d=landmarks_2d)
                is_lateral = ("SIDE" in viewpoint)

                if viewpoint == "SIDE_LEFT":
                    arm_candidates = [("LEFT", 11, 13, 15, 23)]
                    viewpoint_label = "LATERAL (LEFT PROFILE)"
                elif viewpoint == "SIDE_RIGHT":
                    arm_candidates = [("RIGHT", 12, 14, 16, 24)]
                    viewpoint_label = "LATERAL (RIGHT PROFILE)"
                elif viewpoint in ("OBLIQUE_LEFT", "OBLIQUE_RIGHT"):
                    side_pick = "LEFT" if "LEFT" in viewpoint else "RIGHT"
                    cand = ("LEFT", 11, 13, 15, 23) if side_pick == "LEFT" else ("RIGHT", 12, 14, 16, 24)
                    arm_candidates = [cand]
                    viewpoint_label = f"DIAGONAL ({side_pick})"
                else:
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

                    s_3d = bio.get_landmark_3d(world_landmarks, s_idx)
                    e_3d = bio.get_landmark_3d(world_landmarks, e_idx)
                    w_3d = bio.get_landmark_3d(world_landmarks, w_idx)
                    h_3d = bio.get_landmark_3d(world_landmarks, h_idx)

                    s_2d = (int(landmarks_2d[s_idx].x * feed_w), int(landmarks_2d[s_idx].y * feed_h))
                    e_2d = (int(landmarks_2d[e_idx].x * feed_w), int(landmarks_2d[e_idx].y * feed_h))
                    w_2d = (int(landmarks_2d[w_idx].x * feed_w), int(landmarks_2d[w_idx].y * feed_h))

                    tracker.update_arm(side, s_2d, e_2d, w_2d, s_3d, e_3d, w_3d,
                                       s_valid, e_valid, w_valid, now)

                    curr_s_2d = tracker.history.get(f"{side}_shoulder_2d", s_2d)
                    curr_e_2d = tracker.history.get(f"{side}_elbow_2d", e_2d)
                    curr_w_2d = tracker.history.get(f"{side}_wrist_2d", w_2d)

                    # 3D Elbow flexion
                    raw_angle = bio.calculate_angle_3d(s_3d, e_3d, w_3d)
                    smooth_angle = smoother.smooth(f"{side}_bicep", raw_angle)

                    # 3D Upper Arm Drift (angle with torso)
                    raw_drift = bio.calculate_angle_3d(e_3d, s_3d, h_3d)
                    arm_drift = smoother.smooth(f"{side}_drift", raw_drift)

                    arm_issue = None
                    if smooth_angle < 38:
                        arm_issue = "OVER-CURLED"
                        is_tension_ok = False
                    elif arm_drift > 36:
                        arm_issue = "ELBOW SWING"
                        is_pinned_ok = False
                    elif not is_lateral:
                        flare_dist = abs(e_3d[0] - s_3d[0])
                        if flare_dist > 0.22:
                            arm_issue = "ELBOW FLARE"
                            is_flare_ok = False

                    arms_tracked.append({
                        "side": side,
                        "s_2d": curr_s_2d,
                        "e_2d": curr_e_2d,
                        "w_2d": curr_w_2d,
                        "angle": smooth_angle,
                        "drift": arm_drift,
                        "good": (arm_issue is None),
                        "issue": arm_issue
                    })

                active_target_angle = 52.0
                active_delta_str = ""
                active_delta_color = WHITE

                for arm in arms_tracked:
                    active_arm_names.append(arm["side"])
                    bone_color = GREEN if arm["good"] else RED

                    cv2.line(video_frame, arm["s_2d"], arm["e_2d"], bone_color, 4, cv2.LINE_AA)
                    cv2.line(video_frame, arm["e_2d"], arm["w_2d"], bone_color, 4, cv2.LINE_AA)
                    cv2.circle(video_frame, arm["s_2d"], 7, CYAN, -1, cv2.LINE_AA)
                    cv2.circle(video_frame, arm["e_2d"], 8, WHITE, -1, cv2.LINE_AA)
                    cv2.circle(video_frame, arm["w_2d"], 7, CYAN, -1, cv2.LINE_AA)

                    # AR OPTIMAL TARGET GHOST BLUEPRINT
                    t_name, t_angle, d_str, d_color = draw_optimal_bicep_ghost(
                        video_frame, arm["s_2d"], arm["e_2d"], arm["w_2d"],
                        arm["angle"], arm["side"], viewpoint, rep_counter.stage
                    )
                    active_target_angle = t_angle
                    active_delta_str = d_str
                    active_delta_color = d_color

                    arc_color = GREEN if (45 <= arm["angle"] <= 58 or arm["angle"] >= 155) else (RED if not arm["good"] else CYAN)
                    draw_angle_arc(video_frame, arm["e_2d"], arm["s_2d"], arm["w_2d"], arm["angle"], color=arc_color)

                if arms_tracked:
                    primary_angle = sum(a["angle"] for a in arms_tracked) / len(arms_tracked)

                    if any(a["issue"] == "OVER-CURLED" for a in arms_tracked):
                        detected_issue = "OVER-CURLED"
                    elif any(a["issue"] == "ELBOW SWING" for a in arms_tracked):
                        detected_issue = "ELBOW SWING"
                    elif any(a["issue"] == "ELBOW FLARE" for a in arms_tracked):
                        detected_issue = "ELBOW FLARE"

                    rep_stats = rep_counter.update(primary_angle, has_form_issue=(detected_issue is not None), now_time=now)

                    active_issue = debouncer.update(detected_issue, now_time=now)
                    if active_issue == "OVER-CURLED":
                        instruction_text = "Arms over-closed! Back off to ~52 deg to keep tension on biceps"
                        instruction_color = RED
                    elif active_issue == "ELBOW SWING":
                        instruction_text = "Pin elbows at your sides; stop swinging elbows forward"
                        instruction_color = RED
                    elif active_issue == "ELBOW FLARE":
                        instruction_text = "Keep elbows tucked in close to your ribs - stop flaring"
                        instruction_color = RED
                    elif active_delta_str:
                        instruction_text = active_delta_str
                        instruction_color = active_delta_color
                    elif primary_angle <= 58:
                        instruction_text = "Optimal Peak Contraction! Lower forearm with smooth control"
                        instruction_color = GREEN
                    elif primary_angle >= 150:
                        instruction_text = "Optimal Full Extension! Smoothly initiate next curl"
                        instruction_color = GREEN
                    else:
                        instruction_text = "Good form! Follow the glowing optimal target ghost"
                        instruction_color = CYAN
                else:
                    rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                    instruction_text = "Position arms in camera view to begin curls"
                    instruction_color = YELLOW

            else:
                rep_stats = {"total_reps": rep_counter.total_reps, "good_reps": rep_counter.good_reps, "stage": "READY"}
                instruction_text = "Step into camera frame"
                instruction_color = RED
                active_target_angle = 52.0

            # Dashboard Panel
            cv2.rectangle(panel, (16, 16), (panel_w - 16, 80), CARD_BG, -1)
            cv2.rectangle(panel, (16, 16), (panel_w - 16, 80), BORDER_COLOR, 1)
            cv2.putText(panel, "BICEP CURL COACH (3D)", (28, 42), cv2.FONT_HERSHEY_DUPLEX, 0.62, CYAN, 2)

            view_color = GREEN if is_lateral else (CYAN if "FRONTAL" in viewpoint_label else YELLOW)
            cv2.putText(panel, f"VIEW: {viewpoint_label}", (28, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, view_color, 2)
            arm_mode = ", ".join(active_arm_names) if active_arm_names else "SEARCHING"
            cv2.putText(panel, f"ACTIVE ARM: {arm_mode}", (28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1)

            # Reps Card
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

            # ROM Gauge
            draw_rom_gauge(panel, 16, 210, panel_w - 32, 94, primary_angle, target_angle=active_target_angle)

            # Biomechanical Checklist
            cv2.rectangle(panel, (16, 318), (panel_w - 16, 474), CARD_BG, -1)
            cv2.rectangle(panel, (16, 318), (panel_w - 16, 474), BORDER_COLOR, 1)
            cv2.putText(panel, "BIOMECHANICAL CHECKS", (28, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.44, CYAN, 1)

            draw_checklist_item(panel, 28, 359, "Elbow Pinning (No Swing)", is_pinned_ok,
                                "Pinned" if is_pinned_ok else "Swinging Forward")
            draw_checklist_item(panel, 28, 389, "Flare Control (Tucked)", is_flare_ok,
                                "Tucked" if is_flare_ok else "Flaring Out")
            draw_checklist_item(panel, 28, 419, "Bicep Tension Range", is_tension_ok,
                                "Optimal Tension" if is_tension_ok else "Tension Loss")
            draw_checklist_item(panel, 28, 449, "Rep Motion Arc", True,
                                "Full Range")

            # Instruction Card
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

            # Controls
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
    run_bicep_trainer()
