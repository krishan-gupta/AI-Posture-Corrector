import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseDetector:

    def __init__(self):

        # Path to the AI pose model
        model_path = "pose_landmarker.task"

        # Configure the model
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create pose detector
        self.detector = vision.PoseLandmarker.create_from_options(
            options
        )

        # Frame timestamp
        self.timestamp = 0


    def detect_pose(self, frame):

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Increase timestamp
        self.timestamp += 1

        # Detect pose
        results = self.detector.detect_for_video(
            mp_image,
            self.timestamp
        )

        return results