from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import torch
import cv2


class FashionDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize the fashion detector with a YOLOv8 model.

        Args:
            model_path (str): Path to the YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fashion item classes we're interested in
        self.fashion_classes = {
            "person": 0,
            "tie": 27,
            "backpack": 24,
            "umbrella": 25,
            "handbag": 26,
            "suitcase": 28,
        }

    def detect_items(
        self, frame: np.ndarray, conf_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """Detect fashion items in a frame.

        Args:
            frame (np.ndarray): Input frame
            conf_threshold (float): Confidence threshold for detections

        Returns:
            List[Dict[str, Any]]: List of detected items with their details
        """
        results = self.model(frame, conf=conf_threshold)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()  # Convert to numpy array

            # Get class name from the model's names dictionary
            class_name = results.names[cls_id]

            # Only include fashion-related items
            if class_name in self.fashion_classes:
                detection = {
                    "class": class_name,
                    "confidence": conf,
                    "bbox": {
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    },
                }
                detections.append(detection)

        return detections

    def crop_detection(self, frame: np.ndarray, bbox: Dict[str, float]) -> np.ndarray:
        """Crop a detected object from the frame.

        Args:
            frame (np.ndarray): Input frame
            bbox (Dict[str, float]): Bounding box coordinates

        Returns:
            np.ndarray: Cropped image of the detected object
        """
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        return frame[y1:y2, x1:x2]

    @staticmethod
    def draw_detections(
        frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on the frame.

        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict[str, Any]]): List of detections

        Returns:
            np.ndarray: Frame with drawn detections
        """
        frame_copy = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])

            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return frame_copy
