import torch
import clip
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Any, Tuple
from pathlib import Path
import requests
from io import BytesIO


class ProductMatcher:
    def __init__(self, catalog_path: str):
        """Initialize the product matcher with a catalog.

        Args:
            catalog_path (str): Path to the product catalog CSV
        """
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Load product catalog
        self.catalog_df = pd.read_csv(catalog_path)
        self.product_embeddings = None
        self.index = None

        # Initialize FAISS index
        self.build_index()

    def build_index(self):
        """Build FAISS index from product catalog images."""
        embeddings = []

        for url in self.catalog_df["image_url"]:
            try:
                # Download and process image
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img_input = self.preprocess(img).unsqueeze(0).to(self.device)

                # Generate embedding
                with torch.no_grad():
                    embedding = self.model.encode_image(img_input)
                    embedding = embedding.cpu().numpy()

                embeddings.append(embedding[0])

            except Exception as e:
                print(f"Error processing image {url}: {str(e)}")
                embeddings.append(np.zeros(512))  # CLIP embedding dimension

        self.product_embeddings = np.array(embeddings)

        # Build FAISS index
        self.index = faiss.IndexFlatIP(512)  # Use inner product similarity
        self.index.add(self.product_embeddings)

    def match_product(self, img: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Match a detected product image against the catalog.

        Args:
            img (np.ndarray): Detected product image
            top_k (int): Number of top matches to return

        Returns:
            List[Dict[str, Any]]: Top-k matching products with similarity scores
        """
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)

        # Preprocess and generate embedding
        img_input = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_image(img_input)
            query_embedding = query_embedding.cpu().numpy()

        # Search in FAISS index
        similarities, indices = self.index.search(query_embedding, top_k)

        # Format results
        matches = []
        for sim, idx in zip(similarities[0], indices[0]):
            product = self.catalog_df.iloc[idx]
            match_type = self._get_match_type(sim)

            match = {
                "product_id": product["product_id"],
                "similarity": float(sim),
                "match_type": match_type,
                "product_name": product["name"],
                "image_url": product["image_url"],
            }
            matches.append(match)

        return matches

    @staticmethod
    def _get_match_type(similarity: float) -> str:
        """Determine the match type based on similarity score.

        Args:
            similarity (float): Cosine similarity score

        Returns:
            str: Match type (exact, similar, or no_match)
        """
        if similarity > 0.9:
            return "exact"
        elif similarity > 0.75:
            return "similar"
        else:
            return "no_match"
