import cv2
import mediapipe as mp
import ctypes

MODEL_PATH = "../models/pose_landmarker_full.task"

WINDOW_NAME = "Ap Chagi Pose Preview"

MAX_DISPLAY_WIDTH = 900
MAX_DISPLAY_HEIGHT = 700

VISIBILITY_THRESHOLD = 0.5
LANDMARK_RADIUS = 4
LANDMARK_COLOR = (0, 255, 0)


def open_video(video_path):
    """
    Open the video file and create a VideoCapture object.

    Returns:
        video_capture: Object used to read frames from the video.
        fps: Number of frames per second in the video.
    """

    video_capture = cv2.VideoCapture(str(video_path))

    if not video_capture.isOpened():
        raise RuntimeError("Video could not be opened.")

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    return video_capture, fps


def get_display_size(video_capture, max_width, max_height):
    """Calculate proportional display dimensions."""
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_width <= 0 or video_height <= 0:
        raise RuntimeError("Video dimensions could not be read.")

    print(f"Video dimensions: {video_width} x {video_height}")

    scale_factor = min(
        1.0,
        max_width / video_width,
        max_height / video_height
    )

    display_width = int(video_width * scale_factor)
    display_height = int(video_height * scale_factor)

    return display_width, display_height


def setup_window(window_name, window_width, window_height):
    """Create, resize, and center the OpenCV window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    window_x = (screen_width - window_width) // 2
    window_y = (screen_height - window_height) // 2

    cv2.moveWindow(window_name, window_x, window_y)


def create_pose_landmarker(model_path):
    """
    Create the MediaPipe Pose Landmarker.

    Args:
        model_path: Path to the MediaPipe pose landmarker model file.
    Returns:
        A configured PoseLandmarker object used to detect body landmarks
    in video frames.
    """

    base_options = mp.tasks.BaseOptions
    pose_landmarker = mp.tasks.vision.PoseLandmarker
    pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode

    options = pose_landmarker_options(
        base_options=base_options(model_asset_path=str(model_path)),
        running_mode=running_mode.VIDEO,
        num_poses=1
    )

    return pose_landmarker.create_from_options(options)


def draw_landmarks(frame, landmarks):
    """Draw visible pose landmarks on a frame."""
    frame_height, frame_width = frame.shape[:2]

    for landmark in landmarks:
        if landmark.visibility > VISIBILITY_THRESHOLD:
            landmark_x = int(landmark.x * frame_width)
            landmark_y = int(landmark.y * frame_height)

            cv2.circle(
                frame,
                (landmark_x, landmark_y),
                LANDMARK_RADIUS,
                LANDMARK_COLOR,
                -1
            )


def resize_frame(frame, display_width, display_height):
    """Resize the frame for display."""
    return cv2.resize(
        frame,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )


def process_video(video_path):
    video_capture, fps = open_video(video_path)

    display_width, display_height = get_display_size(
        video_capture,
        MAX_DISPLAY_WIDTH,
        MAX_DISPLAY_HEIGHT
    )

    setup_window(WINDOW_NAME, display_width, display_height)

    try:
        with create_pose_landmarker(MODEL_PATH) as landmarker:
            frame_index = 0

            while True:
                success, frame = video_capture.read()

                if not success:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(frame_index * 1000 / fps)

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.pose_landmarks:
                    draw_landmarks(
                        frame,
                        result.pose_landmarks[0]
                    )

                display_frame = resize_frame(
                    frame,
                    display_width,
                    display_height
                )

                cv2.imshow(WINDOW_NAME, display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_index += 1

    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_video_path = (
        "../dataset/ap_chagi/correct/ap_chagi_p001_c001.mp4"
    )

    process_video(
        test_video_path
    )