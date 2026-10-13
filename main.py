
import cv2
import math
import numpy as np
from pose_detector import PoseDetector

# =========================
# SETTINGS
# =========================

WINDOW_NAME = "AI Gym - Posture Corrector"
CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLUE = (255, 160, 50)
BLACK = (25, 25, 25)
DARK_GRAY = (45, 45, 45)

# MediaPipe pose landmark indices
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


# =========================
# HELPER FUNCTIONS
# =========================

def point(landmarks, index, width, height):
    lm = landmarks[index]
    return int(lm.x * width), int(lm.y * height)


def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    denominator = math.hypot(*ba) * math.hypot(*bc)

    if denominator == 0:
        return 0

    cosine = (ba[0] * bc[0] + ba[1] * bc[1]) / denominator
    cosine = max(-1, min(1, cosine))

    return math.degrees(math.acos(cosine))


def visible(landmark):
    value = getattr(landmark, "visibility", 0)
    return value is not None and value >= 0.5


def draw_dotted_line(frame, start, end, color=YELLOW):
    """Draw a dotted line between two points."""
    x1, y1 = start
    x2, y2 = end

    distance = math.hypot(x2 - x1, y2 - y1)

    if distance < 1:
        return

    dx = (x2 - x1) / distance
    dy = (y2 - y1) / distance

    dash_length = 5
    gap_length = 7
    step = dash_length + gap_length

    for position in range(0, int(distance), step):
        finish = min(position + dash_length, distance)

        p1 = (
            int(x1 + dx * position),
            int(y1 + dy * position)
        )

        p2 = (
            int(x1 + dx * finish),
            int(y1 + dy * finish)
        )

        cv2.line(frame, p1, p2, color, 2)


def draw_arm_guide(frame, shoulder, elbow, wrist, side):
    """
    Draw a target guide for each arm in bicep curls.
    The curled wrist target is mirrored between arms.
    """
    height, width = frame.shape[:2]

    upper_length = math.hypot(
        elbow[0] - shoulder[0],
        elbow[1] - shoulder[1]
    )

    lower_length = math.hypot(
        wrist[0] - elbow[0],
        wrist[1] - elbow[1]
    )

    if upper_length < 5 or lower_length < 5:
        return

    # Approximate upper-arm target below the shoulder.
    target_elbow = (
        shoulder[0],
        min(height - 1, int(shoulder[1] + upper_length))
    )

    # Approximate lowered-arm wrist target.
    extended_wrist = (
        target_elbow[0],
        min(height - 1, int(target_elbow[1] + lower_length))
    )

    # Mirror the curl target for the two screen-side arms.
    direction = -1 if side == "LEFT" else 1

    curled_wrist = (
        max(
            0,
            min(
                width - 1,
                int(target_elbow[0] + direction * lower_length * 0.7)
            )
        ),
        max(
            0,
            int(target_elbow[1] - lower_length * 0.7)
        )
    )

    draw_dotted_line(frame, shoulder, target_elbow)
    draw_dotted_line(frame, target_elbow, extended_wrist)
    draw_dotted_line(frame, target_elbow, curled_wrist)


def draw_tricep_guide(frame, shoulder, elbow, wrist, side):
    """
    Draw target guide for overhead tricep extensions.
    Guide lines represent stationary vertical upper arm and full extension path.
    """
    height, width = frame.shape[:2]

    upper_length = math.hypot(
        elbow[0] - shoulder[0],
        elbow[1] - shoulder[1]
    )
    lower_length = math.hypot(
        wrist[0] - elbow[0],
        wrist[1] - elbow[1]
    )

    if upper_length < 5 or lower_length < 5:
        return

    # Stationary vertical upper arm target pointing overhead
    target_elbow = (
        shoulder[0],
        max(0, int(shoulder[1] - upper_length))
    )

    # Full overhead extension target (straight up)
    target_extended_wrist = (
        target_elbow[0],
        max(0, int(target_elbow[1] - lower_length))
    )

    # Flexed target behind head (elbow bent at ~90 degrees)
    direction = 1 if side == "LEFT" else -1
    target_flexed_wrist = (
        max(0, min(width - 1, int(target_elbow[0] + direction * lower_length * 0.5))),
        min(height - 1, int(target_elbow[1] + lower_length * 0.7))
    )

    draw_dotted_line(frame, shoulder, target_elbow)
    draw_dotted_line(frame, target_elbow, target_extended_wrist)
    draw_dotted_line(frame, target_elbow, target_flexed_wrist)


def draw_squat_guide(frame, hip, knee, ankle, side):
    """
    Draw target guide for squats showing ideal parallel thigh level at knee height.
    """
    height, width = frame.shape[:2]

    thigh_len = math.hypot(knee[0] - hip[0], knee[1] - hip[1])
    shin_len = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])

    if thigh_len < 5 or shin_len < 5:
        return

    direction = -1 if side == "LEFT" else 1

    # Ideal parallel squat hip target: horizontally aligned with knee
    target_hip = (
        max(0, min(width - 1, int(knee[0] + direction * thigh_len * 0.85))),
        int(knee[1])
    )

    draw_dotted_line(frame, ankle, knee)
    draw_dotted_line(frame, knee, target_hip)


def draw_arm(frame, shoulder, elbow, wrist, color):
    cv2.line(frame, shoulder, elbow, color, 4)
    cv2.line(frame, elbow, wrist, color, 4)

    for p in (shoulder, elbow, wrist):
        cv2.circle(frame, p, 5, color, -1)


def draw_leg(frame, hip, knee, ankle, color):
    cv2.line(frame, hip, knee, color, 4)
    cv2.line(frame, knee, ankle, color, 4)

    for p in (hip, knee, ankle):
        cv2.circle(frame, p, 5, color, -1)


def draw_torso(frame, shoulder, hip, color):
    cv2.line(frame, shoulder, hip, color, 3)
    cv2.circle(frame, shoulder, 5, color, -1)
    cv2.circle(frame, hip, 5, color, -1)


def draw_top_bar(frame, title):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 42), BLACK, -1)

    cv2.putText(
        frame, title,
        (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, BLUE, 2
    )

    cv2.putText(
        frame, "M: MENU | Q: QUIT",
        (width - 180, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1
    )


def draw_feedback_banner(frame, message, color):
    height, width = frame.shape[:2]
    banner_y = height - 52
    cv2.rectangle(frame, (8, banner_y - 20), (width - 8, banner_y + 10), DARK_GRAY, -1)
    cv2.putText(
        frame, f"FEEDBACK: {message}",
        (16, banner_y), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, color, 2
    )


def draw_legend(frame):
    height, width = frame.shape[:2]
    y = height - 12
    cv2.putText(frame, "RED: Adjust", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1)
    cv2.putText(frame, "GREEN: Good", (135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1)
    cv2.putText(frame, "YELLOW DOTS: Guide", (260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1)


def get_camera(index=CAMERA_INDEX):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(index)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


# =========================
# EXERCISE MENU
# =========================

def draw_menu():
    frame = np.zeros((540, 800, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    cv2.putText(
        frame, "AI GYM - POSTURE CORRECTOR",
        (65, 70), cv2.FONT_HERSHEY_SIMPLEX,
        1, WHITE, 2
    )

    cv2.putText(
        frame, "Choose an exercise",
        (65, 125), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, BLUE, 2
    )

    items = [
        ("1. Bicep Curls", GREEN),
        ("2. Squats", GREEN),
        ("3. Tricep Extensions", GREEN),
        ("4. Push-ups - Coming Soon", WHITE),
    ]

    y = 200

    for label, color in items:
        cv2.rectangle(
            frame, (65, y - 32), (735, y + 18),
            (55, 55, 55), -1
        )

        cv2.putText(
            frame, label, (90, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
        )

        y += 75

    cv2.putText(
        frame, "Press Q to quit",
        (65, 500), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, WHITE, 1
    )

    cv2.imshow(WINDOW_NAME, frame)


# =========================
# 1. BICEP CURLS
# =========================

def run_bicep_curls():
    detector = PoseDetector()
    camera = get_camera(CAMERA_INDEX)

    if not camera.isOpened():
        print("Cannot open webcam. Check CAMERA_INDEX.")
        detector.detector.close()
        return "menu"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 600)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            height, width = frame.shape[:2]
            results = detector.detect_pose(frame)

            feedback_msg = "Keep upper arms still at your sides"
            feedback_color = WHITE

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                arm_results = []

                # Process both arms independently.
                for name, s_idx, e_idx, w_idx, h_idx in ARMS:
                    if not all(
                        visible(landmarks[i])
                        for i in (s_idx, e_idx, w_idx)
                    ):
                        continue

                    shoulder = point(landmarks, s_idx, width, height)
                    elbow = point(landmarks, e_idx, width, height)
                    wrist = point(landmarks, w_idx, width, height)

                    angle = calculate_angle(shoulder, elbow, wrist)
                    upper_len = max(15, math.hypot(elbow[0] - shoulder[0], elbow[1] - shoulder[1]))

                    # Error 1: Closing biceps too much (loss of tension, wrist touching shoulder)
                    # Normal peak squeeze is ~45-60 deg. Less than 38 deg is over-curled.
                    is_over_curled = (angle < 38)

                    # Error 2: Elbow flaring out or swinging horizontally (normalized to upper arm length)
                    horizontal_drift = abs(elbow[0] - shoulder[0]) / upper_len
                    is_flaring = (horizontal_drift > 0.35)

                    # Error 3: Lifting elbows forward/upward (cheating the curl with shoulders)
                    vertical_drop = (elbow[1] - shoulder[1]) / upper_len
                    is_lifted = (vertical_drop < 0.60)

                    # Determine specific issue
                    issue = None
                    if is_over_curled:
                        issue = "TOO CLOSE"
                    elif is_lifted:
                        issue = "ELBOW LIFTED"
                    elif is_flaring:
                        issue = "ELBOW FLARE"

                    good = (issue is None)

                    arm_results.append({
                        "name": name,
                        "shoulder": shoulder,
                        "elbow": elbow,
                        "wrist": wrist,
                        "angle": angle,
                        "good": good,
                        "issue": issue
                    })

                # Guides are drawn first, actual arms on top.
                for arm in arm_results:
                    draw_arm_guide(
                        frame, arm["shoulder"], arm["elbow"], arm["wrist"], arm["name"]
                    )

                for arm in arm_results:
                    color = GREEN if arm["good"] else RED
                    draw_arm(frame, arm["shoulder"], arm["elbow"], arm["wrist"], color)

                    label_y = 65 if arm["name"] == "LEFT" else 95
                    if arm["good"]:
                        status_str = f"{arm['name']}: GOOD ({int(arm['angle'])} deg)"
                    else:
                        status_str = f"{arm['name']}: ADJUST - {arm['issue']} ({int(arm['angle'])} deg)"

                    cv2.putText(
                        frame,
                        status_str,
                        (12, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 2
                    )

                if arm_results:
                    avg_angle = sum(a["angle"] for a in arm_results) / len(arm_results)
                    all_good = all(a["good"] for a in arm_results)

                    # Identify priority feedback message
                    over_curled_any = any(a["issue"] == "TOO CLOSE" for a in arm_results)
                    lifted_any = any(a["issue"] == "ELBOW LIFTED" for a in arm_results)
                    flare_any = any(a["issue"] == "ELBOW FLARE" for a in arm_results)

                    if over_curled_any:
                        feedback_msg = "Over-curled! Don't bring hands all the way to shoulders; maintain tension"
                        feedback_color = RED
                    elif lifted_any:
                        feedback_msg = "Keep elbows pinned at sides; don't lift elbows forward"
                        feedback_color = RED
                    elif flare_any:
                        feedback_msg = "Keep elbows tucked in at your sides; stop flaring out"
                        feedback_color = RED
                    elif avg_angle <= 55:
                        feedback_msg = "Peak contraction! Lower slowly with control"
                        feedback_color = GREEN
                    elif avg_angle >= 145:
                        feedback_msg = "Full extension! Curl upward"
                        feedback_color = GREEN
                    else:
                        feedback_msg = "Good form! Keep smooth, controlled tempo"
                        feedback_color = GREEN
                else:
                    feedback_msg = "Show your arms clearly"
                    feedback_color = RED

            else:
                feedback_msg = "PERSON NOT DETECTED"
                feedback_color = RED

            # HUD
            draw_top_bar(frame, "BICEP CURLS")
            draw_feedback_banner(frame, feedback_msg, feedback_color)
            draw_legend(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return "quit"

            if key == ord("m"):
                return "menu"

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.detector.close()


# =========================
# 2. SQUATS
# =========================

def run_squats():
    detector = PoseDetector()
    camera = get_camera(CAMERA_INDEX)

    if not camera.isOpened():
        print("Cannot open webcam. Check CAMERA_INDEX.")
        detector.detector.close()
        return "menu"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 600)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            height, width = frame.shape[:2]
            results = detector.detect_pose(frame)

            feedback_msg = "Stand in frame with legs visible"
            feedback_color = WHITE

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                leg_results = []

                # Process legs
                for name, h_idx, k_idx, a_idx, s_idx in LEGS:
                    if not all(
                        visible(landmarks[i])
                        for i in (h_idx, k_idx, a_idx)
                    ):
                        continue

                    hip = point(landmarks, h_idx, width, height)
                    knee = point(landmarks, k_idx, width, height)
                    ankle = point(landmarks, a_idx, width, height)

                    knee_angle = calculate_angle(hip, knee, ankle)

                    shoulder = None
                    back_angle = None
                    if visible(landmarks[s_idx]):
                        shoulder = point(landmarks, s_idx, width, height)
                        back_angle = calculate_angle(shoulder, hip, knee)

                    leg_results.append({
                        "name": name,
                        "hip": hip,
                        "knee": knee,
                        "ankle": ankle,
                        "shoulder": shoulder,
                        "knee_angle": knee_angle,
                        "back_angle": back_angle
                    })

                # Draw guides first
                for leg in leg_results:
                    draw_squat_guide(
                        frame, leg["hip"], leg["knee"], leg["ankle"], leg["name"]
                    )

                if leg_results:
                    # Posture checks
                    # 1. Back angle check (avoid excessive forward bending / hunched chest)
                    back_angles = [l["back_angle"] for l in leg_results if l["back_angle"] is not None]
                    back_too_bent = any(ba < 65 for ba in back_angles)

                    # 2. Knee Cave / Valgus check (if facing camera with both legs visible)
                    left_leg = next((l for l in leg_results if l["name"] == "LEFT"), None)
                    right_leg = next((l for l in leg_results if l["name"] == "RIGHT"), None)
                    knee_cave = False
                    if left_leg and right_leg:
                        hip_dist = abs(left_leg["hip"][0] - right_leg["hip"][0])
                        knee_dist = abs(left_leg["knee"][0] - right_leg["knee"][0])
                        ankle_dist = abs(left_leg["ankle"][0] - right_leg["ankle"][0])
                        # If frontal view and knees are noticeably closer together than ankles
                        avg_knee_angle = (left_leg["knee_angle"] + right_leg["knee_angle"]) / 2
                        if hip_dist > 30 and avg_knee_angle < 135:
                            if ankle_dist > 40 and knee_dist < (ankle_dist * 0.72):
                                knee_cave = True

                    # 3. Squat depth check
                    avg_knee_angle = sum(l["knee_angle"] for l in leg_results) / len(leg_results)
                    too_deep = any(l["knee_angle"] < 68 for l in leg_results)

                    # Draw legs and status
                    for i, leg in enumerate(leg_results):
                        leg_issue = None
                        if leg["knee_angle"] < 68:
                            leg_issue = "TOO DEEP"
                        elif leg["back_angle"] is not None and leg["back_angle"] < 65:
                            leg_issue = "CHEST DOWN"
                        elif knee_cave:
                            leg_issue = "KNEE CAVE"

                        leg_good = (leg_issue is None)
                        color = GREEN if leg_good else RED
                        draw_leg(frame, leg["hip"], leg["knee"], leg["ankle"], color)

                        if leg["shoulder"]:
                            draw_torso(frame, leg["shoulder"], leg["hip"], color)

                        label_y = 65 + i * 28
                        if leg_good:
                            status_str = f"{leg['name']}: GOOD (Knee: {int(leg['knee_angle'])} deg)"
                        else:
                            status_str = f"{leg['name']}: ADJUST - {leg_issue} (Knee: {int(leg['knee_angle'])} deg)"

                        cv2.putText(
                            frame,
                            status_str,
                            (12, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, color, 2
                        )

                    # Form feedback cues
                    if back_too_bent:
                        feedback_msg = "Keep chest up and back straight!"
                        feedback_color = RED
                    elif knee_cave:
                        feedback_msg = "Push knees outward - don't let knees cave in!"
                        feedback_color = RED
                    elif too_deep:
                        feedback_msg = "Squatting too deep! Stop around parallel (90 deg)"
                        feedback_color = RED
                    elif avg_knee_angle <= 100:
                        feedback_msg = "Great squat depth! Push through heels to stand up"
                        feedback_color = GREEN
                    elif avg_knee_angle <= 140:
                        feedback_msg = "Squat lower to reach parallel (90 deg)"
                        feedback_color = YELLOW
                    else:
                        feedback_msg = "Standing tall. Initiate squat by pushing hips back"
                        feedback_color = GREEN
                else:
                    feedback_msg = "Stand further back so legs are clearly visible"
                    feedback_color = RED

            else:
                feedback_msg = "PERSON NOT DETECTED"
                feedback_color = RED

            # HUD
            draw_top_bar(frame, "SQUATS")
            draw_feedback_banner(frame, feedback_msg, feedback_color)
            draw_legend(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return "quit"

            if key == ord("m"):
                return "menu"

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.detector.close()


# =========================
# 3. TRICEP EXTENSIONS
# =========================

def run_tricep_extensions():
    detector = PoseDetector()
    camera = get_camera(CAMERA_INDEX)

    if not camera.isOpened():
        print("Cannot open webcam. Check CAMERA_INDEX.")
        detector.detector.close()
        return "menu"

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 600)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(
                frame,
                (FRAME_WIDTH, FRAME_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            height, width = frame.shape[:2]
            results = detector.detect_pose(frame)

            feedback_msg = "Raise arms overhead for tricep extension"
            feedback_color = WHITE

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                arm_results = []

                # Process arms
                for name, s_idx, e_idx, w_idx, h_idx in ARMS:
                    if not all(
                        visible(landmarks[i])
                        for i in (s_idx, e_idx, w_idx)
                    ):
                        continue

                    shoulder = point(landmarks, s_idx, width, height)
                    elbow = point(landmarks, e_idx, width, height)
                    wrist = point(landmarks, w_idx, width, height)

                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    upper_len = max(15, math.hypot(elbow[0] - shoulder[0], elbow[1] - shoulder[1]))

                    # Error 1: Closing triceps too much (hyperflexion of the elbow joint)
                    # Healthy stretch is ~65-85 deg. Angle < 58 deg strains the elbow joint and triceps tendon.
                    is_over_closed = (elbow_angle < 58)

                    # Error 2: Elbow dropping below overhead position
                    # In overhead extension, elbow must stay high above the shoulder:
                    # (shoulder[1] - elbow[1]) / upper_len should be >= 0.40
                    is_elbow_dropped = ((shoulder[1] - elbow[1]) / upper_len < 0.40)

                    # Error 3: Elbow flaring outward away from shoulder line
                    horizontal_drift = abs(elbow[0] - shoulder[0]) / upper_len
                    is_flaring = (horizontal_drift > 0.35)

                    # Determine specific issue
                    issue = None
                    if is_over_closed:
                        issue = "TOO CLOSE"
                    elif is_elbow_dropped:
                        issue = "ELBOWS LOW"
                    elif is_flaring:
                        issue = "ELBOW FLARE"

                    good = (issue is None)

                    arm_results.append({
                        "name": name,
                        "shoulder": shoulder,
                        "elbow": elbow,
                        "wrist": wrist,
                        "angle": elbow_angle,
                        "good": good,
                        "issue": issue
                    })

                # Guides drawn first
                for arm in arm_results:
                    draw_tricep_guide(
                        frame, arm["shoulder"], arm["elbow"], arm["wrist"], arm["name"]
                    )

                if arm_results:
                    for i, arm in enumerate(arm_results):
                        color = GREEN if arm["good"] else RED
                        draw_arm(frame, arm["shoulder"], arm["elbow"], arm["wrist"], color)

                        label_y = 65 + i * 28
                        if arm["good"]:
                            status_str = f"{arm['name']}: GOOD ({int(arm['angle'])} deg)"
                        else:
                            status_str = f"{arm['name']}: ADJUST - {arm['issue']} ({int(arm['angle'])} deg)"

                        cv2.putText(
                            frame,
                            status_str,
                            (12, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, color, 2
                        )

                    avg_angle = sum(a["angle"] for a in arm_results) / len(arm_results)
                    all_good = all(a["good"] for a in arm_results)

                    # Posture feedback cues
                    over_closed_any = any(a["issue"] == "TOO CLOSE" for a in arm_results)
                    dropped_any = any(a["issue"] == "ELBOWS LOW" for a in arm_results)
                    flare_any = any(a["issue"] == "ELBOW FLARE" for a in arm_results)

                    if over_closed_any:
                        feedback_msg = "Arms closed too much! Stop at ~70 deg; don't over-flex (protect elbow joints)"
                        feedback_color = RED
                    elif dropped_any:
                        feedback_msg = "Raise elbows up overhead! Keep upper arms pointing straight up"
                        feedback_color = RED
                    elif flare_any:
                        feedback_msg = "Keep elbows tucked close to head - stop flaring outward"
                        feedback_color = RED
                    elif avg_angle <= 85:
                        feedback_msg = "Deep stretch achieved! Now extend overhead"
                        feedback_color = GREEN
                    elif avg_angle >= 145:
                        feedback_msg = "Full extension! Lower forearm smoothly behind head"
                        feedback_color = GREEN
                    else:
                        feedback_msg = "Good elbow control! Keep upper arms stationary"
                        feedback_color = GREEN

                else:
                    feedback_msg = "Raise your arms overhead so they are clearly visible"
                    feedback_color = RED

            else:
                feedback_msg = "PERSON NOT DETECTED"
                feedback_color = RED

            # HUD
            draw_top_bar(frame, "TRICEP EXTENSIONS")
            draw_feedback_banner(frame, feedback_msg, feedback_color)
            draw_legend(frame)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                return "quit"

            if key == ord("m"):
                return "menu"

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.detector.close()


# =========================
# MAIN
# =========================

def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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
            print("Push-ups are coming soon.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
