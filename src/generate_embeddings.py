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

    def get_embedding(self, pil_img: Image.Image, text: str) -> np.ndarray:
        """Get combined embedding from both image and text in a single forward pass"""
        inputs = self.clip_processor(
            images=pil_img, text=text, return_tensors="pt", padding=True
        ).to(self.device)

        with (
            torch.no_grad(),
            torch.cuda.amp.autocast() if self.device.type == "cuda" else nullcontext(),
        ):
            # Get both image and text features
            outputs = self.clip_model(**inputs)
            image_embedding = outputs.image_embeds.squeeze().cpu().numpy()
            text_embedding = outputs.text_embeds.squeeze().cpu().numpy()
            combined = np.concatenate([image_embedding, text_embedding])

        return combined

    # We use this to get all the detections for a given category
    def get_detections_by_category(self, img_cv: np.ndarray, category: str) -> list:
        """Get detections filtered by category (top/bottom)"""
        detections = self.fashion_detector.detect_items(img_cv)
        filtered = []

        for det in detections:
            cls = det["class"]
            if category == "top" and cls in self.TOP_CLASSES:
                filtered.append(det)
            elif category == "bottom" and cls in self.BOTTOM_CLASSES:
                filtered.append(det)

        return filtered

    def get_best_detection(self, img_cv: np.ndarray, category: str = None) -> tuple:
        """Get the best detection for a given category or overall"""
        if category:
            detections = self.get_detections_by_category(img_cv, category)
        else:
            detections = self.fashion_detector.detect_items(img_cv)

        if not detections:
            return None, None

        best = max(detections, key=lambda x: x["confidence"])
        cropped = self.fashion_detector.crop_detection(img_cv, best["bbox"])
        return Image.fromarray(cropped[:, :, ::-1]), best["class"]  # BGR to RGB

    def process_image(self, img: Image.Image, prodtype: str) -> tuple:
        """Process image according to product type rules"""
        img_cv = np.array(img)[:, :, ::-1]  # RGB to BGR

        if prodtype == "top":
            cropped_img, detected_class = self.get_best_detection(img_cv, "top")
            text = detected_class
            return cropped_img, text

        elif prodtype == "bottom":
            cropped_img, detected_class = self.get_best_detection(img_cv, "bottom")
            text = detected_class
            return cropped_img, text

        elif prodtype == "Co-ord":
            # Get best top and bottom detections
            top_img, top_class = self.get_best_detection(img_cv, "top")
            bottom_img, bottom_class = self.get_best_detection(img_cv, "bottom")

            if top_img and bottom_img:
                # Get embeddings for both top and bottom
                top_embedding = self.get_embedding(top_img, top_class)
                bottom_embedding = self.get_embedding(bottom_img, bottom_class)

                # Average the embeddings without normalization
                combined_embedding = (top_embedding + bottom_embedding) / 2

                # Return the original image and combined text
                text = f"{top_class} {bottom_class}"
                return img, text, combined_embedding
            return None, None, None

        else:  # other or any other type
            cropped_img, detected_class = self.get_best_detection(img_cv)
            text = detected_class
            return cropped_img, text

    def generate_embeddings_from_df(self, images_df: pd.DataFrame) -> dict:
        """
        Generates and returns a dictionary of product_id -> averaged CLIP embeddings.
        Now includes both image and text embeddings in a single forward pass.
        """
        id_to_embeddings = {}

        for _, row in tqdm(images_df.iterrows(), total=len(images_df)):
            prod_id = row["id"]
            url = row["image_url"]
            prodtype = row["prod"]

            img = self.fetch_image(url)
            if img is None:
                continue

            if prodtype == "Co-ord":
                processed_img, text, combined_embedding = self.process_image(
                    img, prodtype
                )
                if processed_img is None:
                    continue
                id_to_embeddings.setdefault(prod_id, []).append(combined_embedding)
            else:
                processed_img, text = self.process_image(img, prodtype)
                if processed_img is None:
                    continue
                embedding = self.get_embedding(processed_img, text)
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
        Normalizes embeddings only at the final step.
        """
        embedding_matrix = np.stack(list(embeddings.values())).astype("float32")
        ids = list(embeddings.keys())

        # Normalize only at the final step
        embedding_matrix = embedding_matrix / np.linalg.norm(
            embedding_matrix, axis=1, keepdims=True
        )

        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        faiss.write_index(index, index_path)
        pd.DataFrame({"id": ids}).to_csv(id_path, index=False)
