from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import tempfile
import os
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.video_processor import VideoProcessor
from src.object_detection import FashionDetector
from src.product_matching import ProductMatcher
from src.vibe_classifier import VibeClassifier

app = FastAPI(title="Flickd Fashion Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
detector = FashionDetector()
matcher = ProductMatcher("data/catalog/products.csv")
vibe_classifier = VibeClassifier()


class VideoAnalysisResponse(BaseModel):
    video_id: str
    vibes: List[str]
    products: List[Dict[str, Any]]


@app.post("/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(
    video: UploadFile = File(...),
    caption: Optional[str] = None,
    hashtags: Optional[str] = None,
):
    """Analyze a fashion video and return detected products and vibes.

    Args:
        video (UploadFile): The video file to analyze
        caption (str, optional): Video caption
        hashtags (str, optional): Video hashtags

    Returns:
        VideoAnalysisResponse: Analysis results
    """
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        video_path = temp_video.name

    try:
        # Process video frames
        detected_products = []
        with VideoProcessor(video_path) as processor:
            for frame_num, frame in processor.extract_keyframes(interval=0.5):
                # Detect fashion items
                detections = detector.detect_items(frame)

                for det in detections:
                    # Crop detected object
                    cropped_img = detector.crop_detection(frame, det["bbox"])

                    # Match product
                    matches = matcher.match_product(cropped_img, top_k=1)
                    if matches:
                        best_match = matches[0]

                        # Add to results if confidence is high enough
                        if best_match["match_type"] != "no_match":
                            product_info = {
                                "type": det["class"],
                                "frame_number": frame_num,
                                "confidence": det["confidence"],
                                "match_type": best_match["match_type"],
                                "matched_product_id": best_match["product_id"],
                                "similarity": best_match["similarity"],
                            }
                            detected_products.append(product_info)

        # Analyze vibes from caption and hashtags
        text_content = " ".join(filter(None, [caption, hashtags]))
        vibes = vibe_classifier.get_final_vibes(text_content) if text_content else []

        # Prepare response
        response = {
            "video_id": video.filename,
            "vibes": vibes,
            "products": detected_products,
        }

        return response

    finally:
        # Clean up temporary file
        os.unlink(video_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
