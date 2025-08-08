from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
import torch

# CLIP for images
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# SentenceTransformer for text
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

async def get_image_embedding(image_bytes: bytes) -> list[float]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs[0].cpu().numpy().tolist()

async def get_text_embedding(text: str) -> list[float]:
    embedding = text_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
