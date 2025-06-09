import pandas as pd
import requests
import numpy as np
import torch
import faiss
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from object_detection import FashionDetector  # assumes your earlier class is saved here
import os

print("Working Directory:", os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load data
images_df = pd.read_csv(
    r"C:\Users\SHIV\Desktop\Flickd Hackathon\data\catalog\images.csv"
)  # columns: id, image_url
products_df = pd.read_csv(
    r"C:\Users\SHIV\Desktop\Flickd Hackathon\data\catalog\product_data.csv"
)  # columns: id, prodtype (top, bottom, other)

# Merge product type into image data
images_df = images_df.merge(products_df, on="id", how="left")
images_df.head()

# Define top/bottom categories
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

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
fashion_detector = FashionDetector()


def fetch_image(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except:
        return None


def get_embedding(pil_img):
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.cpu().squeeze().numpy()


def get_best_crop(img_pil, prodtype):
    img_cv = np.array(img_pil)[:, :, ::-1]  # Convert PIL to BGR
    detections = fashion_detector.detect_items(img_cv)

    filtered = []
    for det in detections:
        cls = det["class"]
        if prodtype == "top" and cls in TOP_CLASSES:
            filtered.append(det)
        elif prodtype == "bottom" and cls in BOTTOM_CLASSES:
            filtered.append(det)

    if not filtered:
        return None  # no relevant detection

    # Select highest confidence detection
    best = max(filtered, key=lambda x: x["confidence"])
    return fashion_detector.crop_detection(img_cv, best["bbox"])


# Collect embeddings
id_to_embeddings = {}

for _, row in tqdm(images_df.iterrows(), total=len(images_df)):
    prod_id = row["id"]
    url = row["image_url"]
    prodtype = row["prod"]

    img = fetch_image(url)
    if img is None:
        continue

    if prodtype == "other":
        embedding = get_embedding(img)
    else:
        cropped = get_best_crop(img, prodtype)
        if cropped is None:
            continue
        cropped_pil = Image.fromarray(cropped[:, :, ::-1])  # BGR to RGB
        embedding = get_embedding(cropped_pil)

    if prod_id not in id_to_embeddings:
        id_to_embeddings[prod_id] = []
    id_to_embeddings[prod_id].append(embedding)

# Compute final embeddings (average per product ID)
final_embeddings = {}
for prod_id, emb_list in id_to_embeddings.items():
    emb_stack = np.stack(emb_list)
    final_embeddings[prod_id] = emb_stack.mean(axis=0)

# Save to FAISS index or CSV
embedding_matrix = np.stack(list(final_embeddings.values())).astype("float32")
ids = list(final_embeddings.keys())

embedding_matrix = embedding_matrix / np.linalg.norm(
    embedding_matrix, axis=1, keepdims=True
)

index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

faiss.write_index(index, "product_embeddingsIP.index")
pd.DataFrame({"id": ids}).to_csv("product_embedding_idsIP.csv", index=False)

print("Embedding pipeline complete.")
