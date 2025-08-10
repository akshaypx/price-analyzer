# app_faiss_single.py
from fastapi import FastAPI, File, Form, UploadFile
from typing import List, Dict, Any
import uuid
import asyncio
import concurrent.futures
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import torch
from serpapi import GoogleSearch
from app.core.config import settings
import requests

app = FastAPI(title="FAISS Shopping Search (All Providers)")

# -------------------------
# Global variables
# -------------------------
_products: List[Dict[str, Any]] = []
_next_id = 1
_text_index = None
_image_index = None
_text_id_map: Dict[int, int] = {}
_image_id_map: Dict[int, int] = {}
_text_count = 0
_image_count = 0

# Thread pool
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Device and models
_device = "cuda" if torch.cuda.is_available() else "cpu"
_text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)

# -------------------------
# Provider search functions
# -------------------------
def _google_shopping(query: str):
    params = {
        "api_key": settings.SERPAI_KEY,
        "engine": "google_shopping",
        "google_domain": "google.com",
        "q": query,
        "hl": "en",
        "location": settings.LOCATION
    }
    return GoogleSearch(params).get_dict()

def _yahoo_shopping(query: str):
    params = {
        "api_key": settings.SERPAI_KEY,
        "engine": "yahoo_shopping",
        "p": query
    }
    return GoogleSearch(params).get_dict()

def _ebay(query: str):
    params = {
        "api_key": settings.SERPAI_KEY,
        "engine": "ebay",
        "_nkw": query
    }
    return GoogleSearch(params).get_dict()

def _walmart(query: str):
    params = {
        "currentUser": "[object Object]",
        "api_key": settings.SERPAI_KEY,
        "engine": "walmart",
        "query": query
    }
    return GoogleSearch(params).get_dict()

def _home_depot(query: str):
    params = {
        "country": "us",
        "api_key": settings.SERPAI_KEY,
        "engine": "home_depot",
        "q": query
    }
    return GoogleSearch(params).get_dict()

def _amazon(query: str):
    params = {
        "engine": "amazon",
        "k": query,
        "amazon_domain": "amazon.com",
        "api_key": settings.SERPAI_KEY
    }
    return GoogleSearch(params).get_dict()

_PROVIDER_MAP = {
    "google_shopping": _google_shopping,
    "yahoo_shopping": _yahoo_shopping,
    "ebay": _ebay,
    "walmart": _walmart,
    "home_depot": _home_depot,
    "amazon": _amazon
}

# -------------------------
# Utility functions
# -------------------------
def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

async def get_text_embedding_async(text: str):
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(_executor, _text_model.encode, text, True, "np")
    return emb.astype(np.float32)

async def get_image_embedding_async(image_bytes: bytes):
    def _proc(img_bytes: bytes):
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        inputs = _clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = _clip_model.get_image_features(**inputs)
        return feat[0].cpu().numpy().astype(np.float32)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _proc, image_bytes)

async def fetch_image_bytes_async(url: str) -> bytes:
    def _fetch(url_in):
        try:
            resp = requests.get(url_in, timeout=5)
            if resp.status_code == 200:
                return resp.content
        except Exception:
            pass
        return b""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _fetch, url)

def reset_memory(text_dim=384, image_dim=512):
    global _products, _next_id, _text_index, _image_index, _text_id_map, _image_id_map, _text_count, _image_count
    _products = []
    _next_id = 1
    _text_index = faiss.IndexFlatIP(text_dim)
    _image_index = faiss.IndexFlatIP(image_dim)
    _text_id_map = {}
    _image_id_map = {}
    _text_count = 0
    _image_count = 0

def _add_to_index(index, id_map, count_var, vecs, prod_ids):
    faiss.normalize_L2(vecs)
    index.add(vecs)
    for i, pid in enumerate(prod_ids):
        id_map[count_var + i] = pid
    return count_var + len(prod_ids)

def _search_index(index, id_map, query_vec):
    if index.ntotal == 0:
        return []
    v = np.array([query_vec]).astype(np.float32)
    faiss.normalize_L2(v)
    distances, indices = index.search(v, index.ntotal)  # search all
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx >= 0:
            results.append((id_map[idx], float(dist)))
    return results

# -------------------------
# Main search route
# -------------------------
@app.post("/api/v1/shopping/search")
async def search_store_and_find(
    query: str = Form(...),
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
    providers: str = Form("")
):
    global _next_id, _text_count, _image_count

    reset_memory()

    if providers.strip():
        enabled_providers = {p.strip(): True for p in providers.split(",") if p.strip()}
    else:
        enabled_providers = {p: True for p in _PROVIDER_MAP.keys()}

    failed_providers = []
    all_results = []

    query_image_bytes = await image.read()

    # Run provider searches concurrently
    loop = asyncio.get_running_loop()
    tasks = []
    for provider, enabled in enabled_providers.items():
        if enabled and provider in _PROVIDER_MAP:
            func = _PROVIDER_MAP[provider]
            tasks.append(loop.run_in_executor(_executor, func, query))

    provider_names = [p for p, e in enabled_providers.items() if e and p in _PROVIDER_MAP]
    raw_results_list = await asyncio.gather(*tasks, return_exceptions=True)

    for provider, raw in zip(provider_names, raw_results_list):
        if isinstance(raw, Exception):
            failed_providers.append(provider)
            continue
        try:
            items = raw.get("shopping_results") or raw.get("organic_results") or []
            for item in items:
                all_results.append({
                    "provider": provider,
                    "title": item.get("title"),
                    "brand": item.get("brand"),
                    "url": item.get("link"),
                    "image_url": item.get("thumbnail"),
                    "product_metadata": item
                })
        except Exception:
            failed_providers.append(provider)

    if not all_results:
        return {"status": "no_results", "failed_providers": failed_providers}

    search_id = str(uuid.uuid4())
    prod_ids = []
    text_vectors = []
    image_vectors = []

    async def process_product(item):
        global _next_id  # FIXED: use global instead of nonlocal
        pid = _next_id
        _next_id += 1
        _products.append({
            "id": pid,
            "search_id": search_id,
            "provider": item["provider"],
            "title": item["title"],
            "brand": item["brand"],
            "url": item["url"],
            "image_url": item["image_url"],
            "product_metadata": item["product_metadata"]
        })
        combined_text = f"{item['title']} {item.get('brand', '')}"
        text_emb = await get_text_embedding_async(combined_text)

        img_bytes = b""
        if item["image_url"]:
            img_bytes = await fetch_image_bytes_async(item["image_url"])
        if not img_bytes:
            img_bytes = query_image_bytes

        image_emb = await get_image_embedding_async(img_bytes)
        return pid, normalize_vector(text_emb), normalize_vector(image_emb)

    processed = await asyncio.gather(*[process_product(it) for it in all_results])
    for pid, text_emb, img_emb in processed:
        prod_ids.append(pid)
        text_vectors.append(text_emb)
        image_vectors.append(img_emb)

    _text_count = _add_to_index(_text_index, _text_id_map, _text_count, np.stack(text_vectors), prod_ids)
    _image_count = _add_to_index(_image_index, _image_id_map, _image_count, np.stack(image_vectors), prod_ids)

    query_emb = await get_text_embedding_async(query)
    img_emb = await get_image_embedding_async(query_image_bytes)
    text_matches = _search_index(_text_index, _text_id_map, normalize_vector(query_emb))
    image_matches = _search_index(_image_index, _image_id_map, normalize_vector(img_emb))

    scores = {}
    for pid, score in text_matches + image_matches:
        scores[pid] = scores.get(pid, 0) + (score + 1) / 2

    results_out = [
        {**next(p for p in _products if p["id"] == pid), "score": sc}
        for pid, sc in scores.items()
        if sc >= threshold
    ]
    results_out.sort(key=lambda x: x["score"], reverse=True)

    return {
        "status": "ok",
        "search_id": search_id,
        "failed_providers": failed_providers,
        "results": results_out
    }
