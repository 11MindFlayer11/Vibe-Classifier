# Vibe Classifier

This project implements a smart tagging and vibe classification engine for fashion videos. It processes short-form videos to detect fashion items, match them with a product catalog, and classify the overall fashion vibe.

## Features

1. **Object Detection**
   - Uses YOLOv8 to detect fashion items in video frames
   - Identifies tops, bottoms, dresses, jackets, and accessories
   - Provides bounding boxes and confidence scores

2. **Product Matching**
   - Utilizes CLIP embeddings for product similarity matching
   - Fast similarity search using FAISS
   - Matches detected items against a product catalog

3. **Vibe Classification**
   - NLP-based classification of fashion vibes
   - Supports multiple vibes: Coquette, Clean Girl, Cottagecore, etc.
   - Uses caption and hashtag analysis

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place videos in the `data/videos` directory
   - Ensure product catalog CSV is in `data/catalog`

## Project Structure

```
.
├── api/
│   └── api.py                 # FastAPI endpoints
├── data/
│   ├── catalog/              # Product catalog data
│   ├── index/               # FAISS index files
│   ├── output/              # Generated results
│   └── videos/              # Input videos
├── frontend/
│   └── video_analysis_index.html  # Web interface
├── src/
│   ├── __pycache__/         # Python cache files
│   ├── generate_embeddings.py     # CLIP embedding generation
│   ├── generate_embeddings_pipeline.py  # Embedding pipeline
│   ├── object_detection.py   # YOLOv8 implementation
│   ├── product_matching.py   # CLIP + FAISS matching
│   ├── utils.py             # Helper functions
│   ├── video_processor.py    # Video frame extraction
│   ├── vibe_llm.py          # LLM-based vibe classification
│   └── yolov8n.pt           # YOLOv8 model weights
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Usage

1. Start the API server:
```bash
uvicorn api.main:app --reload
```

2. Send requests to process videos:
```bash
POST /process-video
```

## API Response Format

```json
{
    "video_id": "abc123",
    "vibes": ["Coquette", "Evening"],
    "products": [
        {
            "type": "dress",
            "color": "black",
            "match_type": "similar",
            "matched_product_id": "prod_456",
            "confidence": 0.84
        }
    ]
}
``` 
