import faiss
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Any
import generate_embeddings as ge


embedding_maker = ge.EmbeddingMaker()


class ProductMatcher:
    def __init__(self, catalog_path: str):
        """Initialize the product matcher with a catalog.

        Args:
            catalog_path (str): Path to the product catalog CSV
        """
        self.catalog_df = pd.read_csv(catalog_path)
        self.product_embeddings = None
        self.index = faiss.read_index("data/index/product_embeddingsIP_new.index")
        self.index_ids = pd.read_csv("data/index/product_embedding_idsIP_new.csv")[
            "id"
        ].tolist()

    def match_product(
        self, img: np.ndarray, top_k: int = 5, text: str = None
    ) -> List[Dict[str, Any]]:
        """Match a detected product image against the catalog.

        Args:
            img (np.ndarray): Detected product image
            top_k (int): Number of top matches to return
            text (str, optional): Text description of the product

        Returns:
            List[Dict[str, Any]]: Top-k matching products with similarity scores
        """
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)

        # Preprocess and generate embedding
        query_embedding = embedding_maker.get_embedding(
            pil_img=img_pil,
            text=text if text else "",  # Use provided text or empty string
        )

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )

        # Search in FAISS index
        similarities, indices = self.index.search(query_embedding, top_k)

        # Format results
        matches = []
        for sim, idx in zip(similarities[0], indices[0]):
            product_id = self.index_ids[idx]
            product = self.catalog_df[self.catalog_df["id"] == product_id].iloc[0]
            match_type = self._get_match_type(sim)

            match = {
                "product_id": str(product["id"]),
                "similarity": float(sim),
                "match_type": match_type,
                "product_name": product["title"],
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


# prodmatch = ProductMatcher(
#     r"C:\Users\SHIV\Desktop\Flickd Hackathon\data\catalog\images+id+prodtype.csv"
# )
# import object_detection as od
# import cv2

# detector = od.FashionDetector()
# # # Read image as ndarray (shape: [H, W, C], BGR format)
# img = cv2.imread(
#     r"D:\AI Hackathon-20250604T183910Z-1-001\AI Hackathon\videos\2025-05-28_13-40-09_UTC.jpg"
# )

# # Optional: Convert BGR to RGB
# # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_rgb = img

# detections = detector.detect_items(frame=img_rgb, conf_threshold=0.3)
# crop_obj = detector.crop_detection(frame=img_rgb, bbox=detections[0]["bbox"])
# matches = prodmatch.match_product(crop_obj)

# index = faiss.read_index(
#     r"C:\Users\SHIV\Desktop\Flickd Hackathon\data\index\product_embeddingsIP.index"
# )
