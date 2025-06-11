import pandas as pd
import requests
import numpy as np
import torch
import faiss
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from object_detection import FashionDetector  # Custom detector
import os
from contextlib import nullcontext

# Set PyTorch memory management configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.max_split_size_mb = 512  # Limit memory splits


class EmbeddingMaker:
    TOP_CLASSES = {
        "short_sleeved_shirt",
        "long_sleeved_shirt",
        "short_sleeved_outwear",
        "long_sleeved_outwear",
        "vest",
        "sling",
        "short_sleeved_dress",
        "long_sleeved_dress",
        "vest_dress",
        "sling_dress",
    }

    BOTTOM_CLASSES = {"shorts", "trousers", "skirt"}

    def __init__(self):
        try:
            # Try to use GPU with memory-efficient settings
            self.device = torch.device("cuda")
            # Load model with memory-efficient settings
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                self.clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    torch_dtype=torch.float16,  # Use half precision
                    low_cpu_mem_usage=True,
                ).to(self.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            print("GPU memory insufficient, falling back to CPU")
            self.device = torch.device("cpu")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                low_cpu_mem_usage=True,
            ).to(self.device)

        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.fashion_detector = FashionDetector()

    def fetch_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
        except:
            return None

    def get_embedding(self, pil_img: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(images=pil_img, return_tensors="pt").to(
            self.device
        )

        # Use context managers to optimize memory usage
        with (
            torch.no_grad(),
            torch.cuda.amp.autocast() if self.device.type == "cuda" else nullcontext(),
        ):
            outputs = self.clip_model.get_image_features(**inputs)
            # Move to CPU immediately to free GPU memory
            if self.device.type == "cuda":
                outputs = outputs.cpu()

        return outputs.squeeze().numpy()

    def get_best_crop(self, img_pil: Image.Image, prodtype: str) -> Image.Image or None:
        img_cv = np.array(img_pil)[:, :, ::-1]  # RGB to BGR
        detections = self.fashion_detector.detect_items(img_cv)

        filtered = []
        for det in detections:
            cls = det["class"]
            if prodtype == "top" and cls in self.TOP_CLASSES:
                filtered.append(det)
            elif prodtype == "bottom" and cls in self.BOTTOM_CLASSES:
                filtered.append(det)

        if not filtered:
            return None

        best = max(filtered, key=lambda x: x["confidence"])
        cropped = self.fashion_detector.crop_detection(img_cv, best["bbox"])
        return Image.fromarray(cropped[:, :, ::-1])  # BGR to RGB

    def generate_embeddings_from_df(self, images_df: pd.DataFrame) -> dict:
        """
        Generates and returns a dictionary of product_id -> averaged CLIP embeddings.
        """
        id_to_embeddings = {}

        for _, row in tqdm(images_df.iterrows(), total=len(images_df)):
            prod_id = row["id"]
            url = row["image_url"]
            prodtype = row["prod"]

            img = self.fetch_image(url)
            if img is None:
                continue

            if prodtype == "other":
                embedding = self.get_embedding(img)
            else:
                cropped_img = self.get_best_crop(img, prodtype)
                if cropped_img is None:
                    continue
                embedding = self.get_embedding(cropped_img)

            id_to_embeddings.setdefault(prod_id, []).append(embedding)

        # Average embeddings
        final_embeddings = {}
        for prod_id, emb_list in id_to_embeddings.items():
            emb_stack = np.stack(emb_list)
            final_embeddings[prod_id] = emb_stack.mean(axis=0)

        return final_embeddings

    def save_to_faiss(
        self,
        embeddings: dict,
        index_path="product_embeddingsIP.index",
        id_path="product_embedding_idsIP.csv",
    ):
        """
        Saves the embeddings dictionary to FAISS index and CSV.
        """
        embedding_matrix = np.stack(list(embeddings.values())).astype("float32")
        ids = list(embeddings.keys())

        # Normalize
        embedding_matrix = embedding_matrix / np.linalg.norm(
            embedding_matrix, axis=1, keepdims=True
        )

        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        faiss.write_index(index, index_path)
        pd.DataFrame({"id": ids}).to_csv(id_path, index=False)
