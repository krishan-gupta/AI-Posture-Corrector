"""
Biomechanics and Geometric Math Module for AI Posture Corrector.
Provides:
- Camera-invariant 3D joint angle and inclination math using pose_world_landmarks.
- Viewpoint / orientation auto-detection (Frontal, Side Left, Side Right, Oblique).
- Temporal smoothing (EMA) for flutter-free angle tracking.
- Feedback debouncing with hysteresis.
- Biomechanical repetition counting state machine.
"""

import math
import time
import numpy as np


# ==============================================================================
# 1. 3D GEOMETRIC & VECTOR MATHEMATICS
# ==============================================================================

def get_landmark_3d(landmarks, index):
    """Extract (x, y, z) numpy vector in meters from world_landmarks."""
    lm = landmarks[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def is_landmark_valid(landmarks, index, min_confidence=0.5):
    """Check both visibility and presence of a landmark."""
    lm = landmarks[index]
    vis = getattr(lm, "visibility", 1.0)
    pres = getattr(lm, "presence", 1.0)
    if vis is None:
        vis = 1.0
    if pres is None:
        pres = 1.0
    return (vis >= min_confidence) and (pres >= min_confidence)


def calculate_angle_3d(a, b, c):
    """
    Calculate the 3D angle at vertex b between vectors (a - b) and (c - b).
    Inputs:
        a, b, c: (x, y, z) 3D coordinates in meters.
    Returns:
        Angle in degrees [0.0, 180.0].
    Mathematically invariant to camera tilt, distance, and perspective distortion.
    """
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_inclination_from_vertical_3d(origin, extremity):
    """
    Calculate the 3D inclination angle of a segment (e.g., torso) from the vertical axis.
    In MediaPipe world coordinates:
      -y points vertically upward (gravity direction is +y).
    Returns:
      Angle in degrees [0.0, 180.0], where 0 deg is perfectly upright vertical.
    """
    # Vector pointing from bottom (origin) to top (extremity)
    vec = extremity - origin
    norm = np.linalg.norm(vec)
    if norm < 1e-5:
        return 0.0

    # True upward unit vector
    up_vec = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cosine = np.dot(vec, up_vec) / norm
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_femur_angle_to_horizontal_3d(hip, knee):
    """
    Calculate the angle of the thigh segment (hip to knee) relative to the horizontal floor.
    0 deg = thigh is exactly parallel to the ground (ideal squat depth).
    Positive = knee is lower than hip (standing/descending).
    Negative = hip is lower than knee (deep squat).
    """
    dy = knee[1] - hip[1]  # positive if knee is lower than hip in screen/world
    horizontal_dist = math.hypot(knee[0] - hip[0], knee[2] - hip[2])
    if horizontal_dist < 1e-5:
        return 90.0 if dy > 0 else -90.0
    return float(np.degrees(np.arctan2(dy, horizontal_dist)))


# ==============================================================================
# 2. VIEWPOINT & ORIENTATION AUTO-DETECTION
# ==============================================================================

def detect_viewpoint(world_landmarks, left_shoulder_idx=11, right_shoulder_idx=12,
                     left_hip_idx=23, right_hip_idx=24, landmarks_2d=None):
    """
    Automatically detect user body orientation relative to the camera:
    - 'FRONTAL': Facing camera head-on
    - 'SIDE_LEFT': User standing laterally with left side facing camera
    - 'SIDE_RIGHT': User standing laterally with right side facing camera
    - 'OBLIQUE_LEFT': 3/4 diagonal orientation (left side closer)
    - 'OBLIQUE_RIGHT': 3/4 diagonal orientation (right side closer)
    """
    ls = get_landmark_3d(world_landmarks, left_shoulder_idx)
    rs = get_landmark_3d(world_landmarks, right_shoulder_idx)
    lh = get_landmark_3d(world_landmarks, left_hip_idx)
    rh = get_landmark_3d(world_landmarks, right_hip_idx)

    # Shoulder and hip offsets
    shoulder_dx = abs(ls[0] - rs[0])
    shoulder_dz = abs(ls[2] - rs[2])

    hip_dx = abs(lh[0] - rh[0])
    hip_dz = abs(lh[2] - rh[2])

    avg_dx = (shoulder_dx + hip_dx) / 2.0
    avg_dz = (shoulder_dz + hip_dz) / 2.0

    # 2D screen horizontal shoulder separation (if provided)
    dx_2d = None
    if landmarks_2d is not None:
        dx_2d = abs(landmarks_2d[left_shoulder_idx].x - landmarks_2d[right_shoulder_idx].x)

    # z_diff: negative means left shoulder is closer to camera (left lateral view)
    z_diff = ls[2] - rs[2]

    # When standing laterally:
    # 1. 2D shoulder horizontal distance compresses (< 0.11 of screen width)
    # 2. Or 3D depth separation dz exceeds horizontal width dx
    is_lateral = False
    if dx_2d is not None and dx_2d < 0.11:
        is_lateral = True
    elif avg_dz > max(0.04, avg_dx * 0.85):
        is_lateral = True

    if is_lateral:
        view = "SIDE_LEFT" if z_diff < 0 else "SIDE_RIGHT"
        ratio = 2.0
    elif (dx_2d is not None and dx_2d < 0.18) or (avg_dz > max(0.04, avg_dx * 0.45)):
        view = "OBLIQUE_LEFT" if z_diff < 0 else "OBLIQUE_RIGHT"
        ratio = 1.0
    else:
        view = "FRONTAL"
        ratio = 0.0

    return view, ratio



# ==============================================================================
# 3. TEMPORAL SMOOTHING (EXPONENTIAL MOVING AVERAGE)
# ==============================================================================

class EMASmoother:
    """Smooths fast-changing landmark angles to eliminate single-frame jitter."""

    def __init__(self, alpha=0.38):
        self.alpha = alpha
        self.values = {}

    def smooth(self, key, value):
        if value is None:
            return None
        if key not in self.values or self.values[key] is None:
            self.values[key] = float(value)
        else:
            self.values[key] = self.alpha * float(value) + (1.0 - self.alpha) * self.values[key]
        return self.values[key]

    def reset(self):
        self.values.clear()


# ==============================================================================
# 4. FEEDBACK DEBOUNCER & HYSTERESIS
# ==============================================================================

class FeedbackDebouncer:
    """
    Prevents screen flicker.
    Requires an adjustment issue to persist across N frames before triggering an alert,
    and holds advice steady for a minimum duration so the user can easily read it.
    """

    def __init__(self, required_frames=4, min_hold_seconds=1.2):
        self.required_frames = required_frames
        self.min_hold_seconds = min_hold_seconds
        self.active_issue = None
        self.candidate_issue = None
        self.candidate_count = 0
        self.last_switch_time = 0.0

    def update(self, detected_issue, now_time=None):
        if now_time is None:
            now_time = time.time()

        if detected_issue == self.candidate_issue:
            self.candidate_count += 1
        else:
            self.candidate_issue = detected_issue
            self.candidate_count = 1

        time_since_switch = now_time - self.last_switch_time

        # If candidate issue has persisted long enough, allow switch
        if self.candidate_count >= self.required_frames:
            if self.candidate_issue != self.active_issue and (time_since_switch >= self.min_hold_seconds or self.active_issue is None):
                self.active_issue = self.candidate_issue
                self.last_switch_time = now_time

        # If everything is good for long enough, return to None (no issue)
        if detected_issue is None and self.candidate_count >= (self.required_frames + 2):
            if time_since_switch >= self.min_hold_seconds:
                self.active_issue = None
                self.last_switch_time = now_time

        return self.active_issue


# ==============================================================================
# 5. REPETITION COUNTER STATE MACHINE
# ==============================================================================

class RepetitionCounter:
    """
    Tracks workout repetitions and repetition stages:
    - Counts full reps
    - Distinguishes Good Form reps from Flagged reps
    - Tracks repetition phase ('READY', 'INFLECTION', 'RETURNING')
    """

    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.total_reps = 0
        self.good_reps = 0
        self.stage = "READY"
        self.rep_had_issue = False
        self.last_rep_time = 0.0

    def reset(self):
        self.total_reps = 0
        self.good_reps = 0
        self.stage = "READY"
        self.rep_had_issue = False

    def update(self, primary_angle, has_form_issue, now_time=None):
        """
        Update state machine based on exercise-specific angle thresholds.
        """
        if now_time is None:
            now_time = time.time()

        if has_form_issue:
            self.rep_had_issue = True

        rep_just_completed = False

        if self.exercise_name == "BICEP_CURLS":
            # Extension: > 140 deg; Full Curl Peak: < 55 deg
            if primary_angle > 140:
                if self.stage == "INFLECTION":
                    # Transitioned from top back down to bottom -> Completed Rep!
                    if (now_time - self.last_rep_time) > 0.8:
                        self.total_reps += 1
                        if not self.rep_had_issue:
                            self.good_reps += 1
                        self.last_rep_time = now_time
                        rep_just_completed = True
                    self.rep_had_issue = False
                self.stage = "READY"
            elif primary_angle < 58:
                if self.stage == "READY" or self.stage == "MOVING":
                    self.stage = "INFLECTION"
            elif primary_angle <= 135:
                if self.stage == "READY":
                    self.stage = "MOVING"

        elif self.exercise_name == "SQUATS":
            # Standing: > 145 deg; Squat Depth: <= 100 deg
            if primary_angle > 148:
                if self.stage == "INFLECTION":
                    if (now_time - self.last_rep_time) > 0.9:
                        self.total_reps += 1
                        if not self.rep_had_issue:
                            self.good_reps += 1
                        self.last_rep_time = now_time
                        rep_just_completed = True
                    self.rep_had_issue = False
                self.stage = "READY"
            elif primary_angle <= 102:
                if self.stage == "READY" or self.stage == "MOVING":
                    self.stage = "INFLECTION"
            elif primary_angle <= 140:
                if self.stage == "READY":
                    self.stage = "MOVING"

        elif self.exercise_name == "TRICEP_EXTENSIONS":
            # Full Overhead Extension: > 145 deg; Deep Flexion/Stretch: <= 85 deg
            if primary_angle > 145:
                if self.stage == "INFLECTION":
                    if (now_time - self.last_rep_time) > 0.8:
                        self.total_reps += 1
                        if not self.rep_had_issue:
                            self.good_reps += 1
                        self.last_rep_time = now_time
                        rep_just_completed = True
                    self.rep_had_issue = False
                self.stage = "READY"
            elif primary_angle <= 85:
                if self.stage == "READY" or self.stage == "MOVING":
                    self.stage = "INFLECTION"
            elif primary_angle <= 135:
                if self.stage == "READY":
                    self.stage = "MOVING"

        return {
            "total_reps": self.total_reps,
            "good_reps": self.good_reps,
            "stage": self.stage,
            "completed": rep_just_completed
        }
