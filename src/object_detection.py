from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import torch
import cv2
from huggingface_hub import hf_hub_download

deep_fashion = hf_hub_download(
    repo_id="Bingsu/adetailer", filename="deepfashion2_yolov8s-seg.pt"
)


# We need an images_df with image_urls, ids and prod-type(classified as top or bottom or other)
class FashionDetector:
    def __init__(self, model_path: str = deep_fashion):
        """Initialize the fashion detector with a YOLOv8 model.

        Args:
            model_path (str): Path to the YOLOv8 model weights
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

        # Fashion item classes we're interested in
        self.fashion_classes = {
            "short_sleeved_shirt": 0,
            "long_sleeved_shirt": 1,
            "short_sleeved_outwear": 2,
            "long_sleeved_outwear": 3,
            "vest": 4,
            "sling": 5,
            "shorts": 6,
            "trousers": 7,
            "skirt": 8,
            "short_sleeved_dress": 9,
            "long_sleeved_dress": 10,
            "vest_dress": 11,
            "sling_dress": 12,
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


# detector = FashionDetector()
# # Read image as ndarray (shape: [H, W, C], BGR format)
# img = cv2.imread(
#     r"D:\AI Hackathon-20250604T183910Z-1-001\AI Hackathon\videos\2025-06-02_11-31-19_UTC.jpg"
# )

# Optional: Convert BGR to RGB
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_rgb = img

# detections = detector.detect_items(frame=img_rgb, conf_threshold=0.3)
# new_frame = detector.draw_detections(detections=detections, frame=img_rgb)
# cv2.imwrite(r"C:\Users\SHIV\Desktop\results.jpg", new_frame)

# crop_obj = detector.crop_detection(frame=img_rgb, bbox=detections[0]["bbox"])
# cv2.imwrite(r"C:\Users\SHIV\Desktop\crop.jpg", crop_obj)

# import requests

# response = requests.get(
#     r"https://cdn.shopify.com/s/files/1/0785/1674/8585/files/15thJuneVirgio-0769_1600x.jpg?v=1719118386"
# )
# image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
# img_rgb = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# detections = detector.detect_items(frame=img_rgb, conf_threshold=0.4)
# crop_obj = detector.crop_detection(frame=img_rgb, bbox=detections[0]["bbox"])
