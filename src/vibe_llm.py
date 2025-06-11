import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
from openai import OpenAI
from faster_whisper import WhisperModel
import os
import logging

# Configure logging for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler (or file handler as needed)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --------- Custom OpenAI-compatible API setup ---------
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)


def transcribe_audio(video_path: str) -> str:
    model = WhisperModel("base", compute_type="int8", device="cpu")
    segments, _ = model.transcribe(video_path)
    return " ".join([seg.text for seg in segments])


# --------- 2. Extract middle keyframe ---------
def extract_middle_keyframe(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Failed to extract frame.")
    return frame


# --------- 3. Convert image to base64 ---------
def encode_image_to_base64(image: np.ndarray) -> str:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_caption_text(caption_source: str) -> str:
    """Get caption text, handling both direct text and file paths."""
    if caption_source.endswith(".txt"):
        # Convert forward slashes back to backslashes for Windows
        local_path = caption_source.replace("/", "\\")
        if not os.path.exists(local_path):
            logger.error(f"Caption file not found at {local_path}")
            raise FileNotFoundError(f"Caption file not found at {local_path}")
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.exception(f"Error reading caption file: {e}")
            raise IOError(f"Error reading caption file: {e}")
    else:
        # It's direct text
        return caption_source


# --------- 4. Send everything to GPT model ---------
def classify_vibes_llm(caption: str, transcript: str, image_base64: str) -> dict:
    vibe_list = [
        "Coquette",
        "Clean Girl",
        "Cottagecore",
        "Streetcore",
        "Y2K",
        "Boho",
        "Party Glam",
    ]

    prompt = f"""
You are a fashion stylist and cultural trend analyst.

Your task is to classify the fashion *vibe* of a short video post using:
1. The caption text
2. The audio transcript
3. A visual frame from the video

Choose **1 to 3** vibes from this exact list (no additions or variations):
{", ".join(vibe_list)}

Caption: {caption}
Transcript: {transcript}

Respond with ONLY a comma-separated list of 1-3 items from the provided list.
Example: "Coquette, Clean Girl"
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        top_p=1.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
    )

    # Post-process the response
    raw_response = response.choices[0].message.content.strip()

    # Clean and validate the response
    vibes = [vibe.strip().title() for vibe in raw_response.split(",")]
    valid_vibes = [vibe for vibe in vibes if vibe in vibe_list]

    return {"vibes": ", ".join(valid_vibes)}


# --------- 5. Full pipeline ---------
def classify_video_vibe(video_path: str, caption: str) -> dict:
    print("Step 1: Transcribing audio...")
    transcript = transcribe_audio(video_path)

    print("Step 2: Extracting keyframe...")
    keyframe = extract_middle_keyframe(video_path)
    image_base64 = encode_image_to_base64(keyframe)

    print("Step 3: Sending data to LLM...")
    caption = get_caption_text(caption)
    result = classify_vibes_llm(caption, transcript, image_base64)

    return result
