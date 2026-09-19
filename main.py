import cv2
from pose_detector import PoseDetector


# Create pose detector
detector = PoseDetector()

# Open webcam
camera = cv2.VideoCapture(0)


# Important body landmarks
BODY_POINTS = {
    0: "Face",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle"
}


# Connections between joints
CONNECTIONS = [
    (11, 12),       # Shoulders

    (11, 13),       # Left shoulder → elbow
    (13, 15),       # Left elbow → wrist

    (12, 14),       # Right shoulder → elbow
    (14, 16),       # Right elbow → wrist

    (11, 23),       # Left shoulder → hip
    (12, 24),       # Right shoulder → hip

    (23, 24),       # Hips

    (23, 25),       # Left hip → knee
    (25, 27),       # Left knee → ankle

    (24, 26),       # Right hip → knee
    (26, 28)        # Right knee → ankle
]


while True:

    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Detect pose
    results = detector.detect_pose(frame)

    if results.pose_landmarks:

        # Get first detected person
        landmarks = results.pose_landmarks[0]

        # --------------------------------
        # Draw skeleton connections
        # --------------------------------

        for start, end in CONNECTIONS:

            start_point = landmarks[start]
            end_point = landmarks[end]

            x1 = int(start_point.x * frame.shape[1])
            y1 = int(start_point.y * frame.shape[0])

            x2 = int(end_point.x * frame.shape[1])
            y2 = int(end_point.y * frame.shape[0])

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # --------------------------------
        # Draw only selected body points
        # --------------------------------

        for index, name in BODY_POINTS.items():

            landmark = landmarks[index]

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Green joint point
            cv2.circle(
                frame,
                (x, y),
                7,
                (0, 255, 0),
                -1
            )


        # Person detected
        cv2.putText(
            frame,
            "PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # Show camera
    cv2.imshow(
        "AI Gym - Body Joint Detection",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()