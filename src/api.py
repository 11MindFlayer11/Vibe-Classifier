from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import requests
from io import BytesIO
import uuid
import os
from urllib.parse import unquote, quote
import re
from PIL import Image

from vibe_llm import classify_video_vibe
from object_detection import FashionDetector
from product_matching import ProductMatcher
from generate_embeddings import EmbeddingMaker


app = FastAPI()

# Initialize our components
detector = FashionDetector()
product_matcher = ProductMatcher(
    r"C:\Users\SHIV\Desktop\Flickd Hackathon\data\catalog\images+id+prodtype.csv"
)
embedding_maker = EmbeddingMaker()


class VideoRequest(BaseModel):
    video_url: str  # Can be URL or local path
    caption: Optional[str] = None  # Can be direct text or path to .txt file

    @field_validator("video_url")
    @classmethod
    def normalize_path(cls, v: str) -> str:
        if not v:
            raise ValueError("Video URL/path cannot be empty")
        # Handle both URLs and local paths
        try:
            if is_url(v):
                return sanitize_url(v)
            return unquote(v.replace("\\", "/"))
        except Exception as e:
            raise ValueError(f"Invalid video URL/path: {str(e)}")

    class Config:
        json_encoders = {
            # Add custom encoders if needed
        }
        schema_extra = {
            "example": {
                "video_url": "https://example.com/videos/sample.mp4",
                "caption": "Summer vibes with my favorite dress!",
            }
        }


class ProductMatch(BaseModel):
    type: str
    imageurl: str  # Changed from color to imageurl
    matched_product_id: str
    match_type: str
    confidence: float


class CategoryMatches(BaseModel):
    category: str  # "top" or "bottom"
    matches: List[ProductMatch]


class VideoAnalysisResponse(BaseModel):
    video_id: str
    vibes: List[str]
    products: List[Dict[str, List[ProductMatch]]]


def is_url(path: str) -> bool:
    """Check if the path is a URL."""
    # More robust URL checking
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return bool(url_pattern.match(path))


def sanitize_url(url: str) -> str:
    """Sanitize and properly encode URL."""
    # First decode the URL in case it's already encoded
    decoded_url = unquote(url)
    # Then properly encode it
    # Split URL into parts to preserve the structure
    if "://" in decoded_url:
        scheme, rest = decoded_url.split("://", 1)
        # Encode path parts while preserving slashes
        encoded_parts = [quote(part) for part in rest.split("/")]
        encoded_rest = "/".join(encoded_parts)
        return f"{scheme}://{encoded_rest}"
    return quote(decoded_url)


def get_video_path(video_source: str) -> str:
    """Get video path, handling both URLs and local files."""
    try:
        if is_url(video_source):
            # Sanitize and encode URL
            sanitized_url = sanitize_url(video_source)

            # Download from URL with proper error handling
            try:
                response = requests.get(sanitized_url, timeout=10)
                response.raise_for_status()  # Raise error for bad status codes
            except requests.RequestException as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to download video: {str(e)}"
                )

            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)

            # Save video to temp file with safe filename
            temp_path = f"temp/{uuid.uuid4()}.mp4"
            with open(temp_path, "wb") as f:
                f.write(response.content)
            return temp_path
        else:
            # Handle local path
            # Decode any percent-encoded characters in the path
            local_path = unquote(video_source.replace("/", "\\"))
            if not os.path.exists(local_path):
                raise HTTPException(
                    status_code=400, detail=f"Video file not found at {local_path}"
                )
            return local_path
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=422, detail=f"Error processing video source: {str(e)}"
        )


def get_caption_text(caption_source: str) -> str:
    """Get caption text, handling both direct text and files."""
    if not caption_source:
        return ""

    if caption_source.endswith(".txt"):
        # Convert forward slashes back to backslashes for Windows
        local_path = caption_source.replace("/", "\\")
        if not os.path.exists(local_path):
            raise HTTPException(
                status_code=400, detail=f"Caption file not found at {local_path}"
            )
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error reading caption file: {str(e)}"
            )
    else:
        # It's direct text
        return caption_source


def extract_frames(video_path: str, interval: int = 30) -> List[np.ndarray]:
    """Extract frames from video at given interval."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        for _ in range(interval):
            ret, _ = cap.read()
            if not ret:
                break
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def process_frame(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Process a single frame to detect and match products."""
    # Detect fashion items
    detections = detector.detect_items(frame=frame, conf_threshold=0.3)
    matches = []

    for det in detections:
        # Crop the detected object
        crop_obj = detector.crop_detection(frame=frame, bbox=det["bbox"])

        matches.append(
            {
                "class": det["class"],
                "crop": crop_obj,  # Store the cropped image for later deduplication
                "confidence": det["confidence"],
            }
        )

    return matches


#


@app.post("/analyze_video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoRequest):
    # Get video path and caption text
    video_path = get_video_path(request.video_url)
    caption_text = get_caption_text(request.caption) if request.caption else ""

    should_cleanup = is_url(request.video_url)  # Only cleanup if it was downloaded

    try:
        # Get vibes classification
        vibe_result = classify_video_vibe(video_path, caption_text)
        vibe_data = vibe_result
        vibes_list = [v.strip() for v in vibe_data["vibes"].split(",")]

        # Extract frames and detect products
        frames = extract_frames(video_path)
        all_detections = []

        for frame in frames:
            frame_detections = process_frame(frame)
            all_detections.extend(frame_detections)

        # Deduplicate products within the video using visual similarity
        unique_products = []
        for detection in all_detections:
            is_duplicate = False
            current_embedding = embedding_maker.get_embedding(
                Image.fromarray(detection["crop"])
            )

            # Compare with already found unique products
            for existing_prod in unique_products:
                existing_embedding = embedding_maker.get_embedding(
                    Image.fromarray(existing_prod["crop"])
                )

                # Calculate cosine similarity
                similarity = np.dot(current_embedding, existing_embedding) / (
                    np.linalg.norm(current_embedding)
                    * np.linalg.norm(existing_embedding)
                )

                # If similarity is high and same class, consider it a duplicate
                if similarity > 0.85 and detection["class"] == existing_prod["class"]:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if detection["confidence"] > existing_prod["confidence"]:
                        existing_prod.update(detection)
                    break

            if not is_duplicate:
                unique_products.append(detection)

        # Now match unique products with catalog
        products_list = []
        product_counter = 1  # Separate counter for valid matches only

        for unique_prod in unique_products:
            # Match with catalog - get top 3 matches
            product_matches = product_matcher.match_product(
                unique_prod["crop"], top_k=3
            )

            # Filter out matches with "no_match" match_type
            valid_matches = [
                {
                    "type": unique_prod["class"],
                    "imageurl": match["image_url"],
                    "matched_product_id": match["product_id"],
                    "match_type": match["match_type"],
                    "confidence": float(match["similarity"]),
                }
                for match in product_matches
                if match["match_type"] != "no_match"
            ]

            # Only add to response if there are valid matches
            if valid_matches:
                products_list.append({f"product_{product_counter}": valid_matches})
                product_counter += 1

        # Create response
        response = VideoAnalysisResponse(
            video_id=str(uuid.uuid4()),
            vibes=vibes_list,
            products=products_list,
        )

        return response

    finally:
        # Cleanup only if we  the file
        if should_cleanup and os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
