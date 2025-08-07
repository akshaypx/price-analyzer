from io import BytesIO
from PIL import Image
from clip import clip_processor, clip_model

def get_image_embedding(image_bytes: bytes) -> list[float]:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs[0].detach().numpy().tolist()
