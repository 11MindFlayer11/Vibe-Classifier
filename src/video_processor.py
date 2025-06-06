import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple


class VideoProcessor:
    def __init__(self, video_path: str):
        """Initialize the video processor with a video path.

        Args:
            video_path (str): Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def extract_keyframes(
        self, interval: float = 0.5
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Extract keyframes from the video at specified intervals.

        Args:
            interval (float): Time interval between frames in seconds

        Yields:
            Tuple[int, np.ndarray]: Frame number and the frame image
        """
        frame_interval = int(self.fps * interval)
        frame_num = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                yield frame_num, frame

            frame_num += 1

    def get_frame_at_time(self, time_sec: float) -> Tuple[int, np.ndarray]:
        """Get a specific frame at the given timestamp.

        Args:
            time_sec (float): Time in seconds

        Returns:
            Tuple[int, np.ndarray]: Frame number and the frame image
        """
        frame_num = int(time_sec * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Could not read frame at time {time_sec}s")

        return frame_num, frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
