import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import json


def ensure_directory(path: str) -> Path:
    """Ensure a directory exists, create if it doesn't.

    Args:
        path (str): Directory path

    Returns:
        Path: Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, filepath: str) -> None:
    """Save data to a JSON file.

    Args:
        data (dict): Data to save
        filepath (str): Output file path
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    """Load data from a JSON file.

    Args:
        filepath (str): Input file path

    Returns:
        dict: Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_size: int = 1024,
) -> np.ndarray:
    """Resize an image while maintaining aspect ratio.

    Args:
        image (np.ndarray): Input image
        target_size (Tuple[int, int], optional): Target width and height
        max_size (int): Maximum dimension size

    Returns:
        np.ndarray: Resized image
    """
    if target_size is not None:
        return cv2.resize(image, target_size)

    height, width = image.shape[:2]
    if height <= max_size and width <= max_size:
        return image

    scale = max_size / max(height, width)
    new_height = int(height * scale)
    new_width = int(width * scale)

    return cv2.resize(image, (new_width, new_height))


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.5,
    thickness: int = 1,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Draw text with background on an image.

    Args:
        image (np.ndarray): Input image
        text (str): Text to draw
        position (Tuple[int, int]): Position (x, y) to draw text
        font_scale (float): Font scale
        thickness (int): Text thickness
        font_face (int): Font face
        text_color (Tuple[int, int, int]): Text color in BGR
        bg_color (Tuple[int, int, int]): Background color in BGR

    Returns:
        np.ndarray: Image with text
    """
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )

    # Calculate background rectangle
    x, y = position
    cv2.rectangle(
        image,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        -1,
    )

    # Draw text
    cv2.putText(image, text, position, font_face, font_scale, text_color, thickness)

    return image
