# Vibe Classifier

This project implements a smart tagging and vibe classification engine for fashion videos. It processes short-form videos to detect fashion items, match them with a product catalog, and classify the overall fashion vibe using advanced AI models.

## Features

1. **Object Detection**
   - Uses YOLOv8 to detect fashion items in video frames
   - Identifies multiple fashion categories:
     - Tops: short/long-sleeved shirts, outwear, vests, slings
     - Bottoms: shorts, trousers, skirts
     - Dresses: short/long-sleeved, vest, sling dresses
   - Provides bounding boxes and confidence scores
   - Tunable confidence threshold (default: 0.25)

2. **Product Matching**
   - Utilizes CLIP embeddings for product similarity matching
   - Fast similarity search using FAISS index
   - Matches detected items against a product catalog
   - Configurable match types:
     - Exact match: similarity > 0.9
     - Similar match: similarity > 0.75
     - No match: similarity <= 0.75
   - Customizable top-k matches (default: 5)

3. **Vibe Classification**
   - LLM-based classification of fashion vibes
   - Supports multiple vibes:
     - Coquette
     - Clean Girl
     - Cottagecore
     - Streetcore
     - Y2K
     - Boho
     - Party Glam
   - Uses multimodal analysis:
     - Video frames
     - Audio transcript (using Whisper)
     - Caption text
   - Returns 1-3 most relevant vibes

4. **Embedding Generation**
   - Customizable CLIP embedding generation
   - Support for different product types:
     - Tops
     - Bottoms
     - Co-ords (combined top and bottom)
     - Other items
   - Memory-efficient processing with GPU support
   - Automatic fallback to CPU if GPU memory is insufficient

## Customization Options

1. **Object Detection**
   ```python
   # Adjust confidence threshold
   detector.detect_items(frame=frame, conf_threshold=0.3)  # Default: 0.25
   ```

2. **Product Matching**
   ```python
   # Customize number of matches and similarity thresholds
   matcher.match_product(img=img, top_k=3)  # Default: 5
   # Modify match type thresholds in _get_match_type method
   ```

3. **Embedding Generation**
   ```python
   # Customize embedding generation
   embedding_maker = EmbeddingMaker()
   # Add custom product types
   embedding_maker.TOP_CLASSES.add("new_top_type")
   embedding_maker.BOTTOM_CLASSES.add("new_bottom_type")
   ```

4. **Video Processing**
   ```python
   # Adjust frame extraction interval
   video_processor.extract_keyframes(interval=0.5)  # Default: 0.5 seconds
   ```

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
   - Generated embeddings will be stored in `data/index`

## Project Structure

```
.
├── api/
│   └── api.py                 # FastAPI endpoints
├── data/
│   ├── catalog/              # Product catalog data
│   ├── index/               # FAISS index files
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
├── environment/             # Used Anaconda-Environment File if any problem arises while set-up
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Usage

1. Start the API server:
```bash
uvicorn api.api:app --reload
```

2. Send requests to process videos:
```bash
POST /analyze_video
{
    "video_url": "path/to/video.mp4",
    "caption": "Optional caption text"
}
```

## API Response Format

```json
{
    "video_id": "uuid",
    "vibes": ["Coquette", "Clean Girl"],
    "products": [
        {
            "product_1": {
                "matches": [
                    {
                        "type": "short_sleeved_shirt",
                        "imageurl": "https://example.com/image.jpg",
                        "matched_product_id": "123",
                        "match_type": "similar",
                        "confidence": 0.85
                    }
                ],
                "detected_object": "base64_encoded_image"
            }
        }
    ]
}
```

## Training Custom Embeddings

1. Prepare your catalog data:
   - CSV format with columns: id, image_url, prod (product type)
   - Product types: "top", "bottom", "Co-ord", "other"

2. Generate embeddings:
```bash
python src/generate_embeddings_pipeline.py
```

3. The pipeline will:
   - Process each product image
   - Generate CLIP embeddings
   - Create FAISS index
   - Save embeddings and index files

## Performance Considerations

- GPU acceleration for CLIP embeddings (automatic fallback to CPU)
- FAISS for fast similarity search
- Memory-efficient video processing
- Configurable batch sizes and processing intervals 
