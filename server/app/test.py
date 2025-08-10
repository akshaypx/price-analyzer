# app_faiss.py
"""
FastAPI + FAISS in-memory example for your shopping aggregator dev workflow.

Endpoints:
- POST /api/v1/shopping/search-upload
    multipart form:
      - query (str), description (str), providers (comma-separated str), image (file)
    runs provider searches (SerpAPI), computes embeddings, stores products in memory and indexes in FAISS.
    returns {"status":"stored","count":N,"search_id": "<uuid>"}

- POST /api/v1/shopping/search-results
    form fields:
      - search_id (optional): filter to a particular search UUID
      - query (optional): text query to search by
      - image (optional): image file to search by image
      - provider (optional): provider filter e.g. "google_shopping"
      - price_min / price_max (optional floats)
      - k (optional int) top-K results
    returns list of product dicts (with id, search_id, provider, title, price, url, image_url, score, product_metadata)

This is in-memory only (FAISS + Python lists). Good for local dev/testing.
"""
from fastapi import FastAPI, APIRouter, Depends, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import uuid
import asyncio
import concurrent.futures
import numpy as np
import faiss
import math
import re
from decimal import Decimal, InvalidOperation
import json

# ---- Embedding libs (blocking) ----
# sentence-transformers for text, transformers CLIP for images
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import torch

# ---- SerpAPI provider functions (synchronous) ----
# keep your existing serpapi code here but we run it in threadpool
from serpapi import GoogleSearch
from app.core.config import settings  # adjust import path if needed

# ---- App init ----
app = FastAPI(title="Dev FAISS shopping backend")
router = APIRouter(prefix="/api/v1/shopping")

# ---- Global in-memory stores ----
# products: list of dicts, each product has integer `id` and uuid `search_id` (UUID obj)
_products: List[Dict[str, Any]] = []
_next_id = 1

# FAISS indexes: we keep two separate indexes (text, image). Vectors normalized (for cos similarity via IP).
_text_dim = 384      # match your sentence-transformers model dim (all-MiniLM-L6-v2 -> 384)
_image_dim = 512     # CLIP ViT-B/32 image embedding size -> 512
_text_index = faiss.IndexFlatIP(_text_dim)
_image_index = faiss.IndexFlatIP(_image_dim)
# map faiss internal index position -> product id
_text_id_map: Dict[int, int] = {}
_image_id_map: Dict[int, int] = {}
_text_count = 0
_image_count = 0

# ThreadPoolExecutor for blocking calls
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

# ---- Load models once (heavy) ----
_device = "cuda" if torch.cuda.is_available() else "cpu"
_text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384 dim
_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)


# ---------- Utility functions ----------
def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


async def get_text_embedding_async(text: str) -> List[float]:
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(_executor, _text_model.encode, text, True, "np")
    return emb.astype(np.float32).tolist()


async def get_image_embedding_async(image_bytes: bytes) -> List[float]:
    def _proc(img_bytes: bytes):
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        inputs = _clip_processor(images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = _clip_model.get_image_features(**inputs)
        arr = feat[0].cpu().numpy().astype(np.float32)
        return arr
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(_executor, _proc, image_bytes)
    return emb.tolist()


def parse_price_to_decimal(val: Any) -> Optional[Decimal]:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val.quantize(Decimal("0.01"))
    if isinstance(val, (int, float)):
        return Decimal(str(val)).quantize(Decimal("0.01"))
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d\.\-]", "", val)
        if not cleaned:
            return None
        try:
            return Decimal(cleaned).quantize(Decimal("0.01"))
        except InvalidOperation:
            return None
    return None


# ---------- SerpAPI wrappers (blocking) ----------
# We adapt your earlier serp_api functions; they are blocking so run in executor.
def _google_shopping_search_sync(query: str) -> dict:
    params = {
        "api_key": settings.SERPAI_KEY,
        "engine": "google_shopping",
        "google_domain": "google.com",
        "q": query,
        "hl": "en",
        "location": settings.LOCATION,
    }
    search = GoogleSearch(params)
    return search.get_dict()


def _yahoo_shopping_search_sync(query: str) -> dict:
    params = {"api_key": settings.SERPAI_KEY, "engine": "yahoo_shopping", "p": query}
    search = GoogleSearch(params)
    return search.get_dict()


def _ebay_search_sync(query: str) -> dict:
    params = {"api_key": settings.SERPAI_KEY, "engine": "ebay", "_nkw": query}
    search = GoogleSearch(params)
    return search.get_dict()


def _walmart_search_sync(query: str) -> dict:
    params = {
        "currentUser": "[object Object]",
        "api_key": settings.SERPAI_KEY,
        "engine": "walmart",
        "query": query,
    }
    search = GoogleSearch(params)
    return search.get_dict()


def _home_depot_search_sync(query: str) -> dict:
    params = {"country": "us", "api_key": settings.SERPAI_KEY, "engine": "home_depot", "q": query}
    search = GoogleSearch(params)
    return search.get_dict()


def _amazon_search_sync(query: str) -> dict:
    params = {"engine": "amazon", "k": query, "amazon_domain": "amazon.com", "api_key": settings.SERPAI_KEY}
    search = GoogleSearch(params)
    return search.get_dict()


_PROVIDER_MAP = {
    "google_shopping": _google_shopping_search_sync,
    "yahoo_shopping": _yahoo_shopping_search_sync,
    "ebay": _ebay_search_sync,
    "walmart": _walmart_search_sync,
    "home_depot": _home_depot_search_sync,
    "amazon": _amazon_search_sync,
}


async def run_provider_searches_async(query: str, providers: List[str]) -> List[dict]:
    """
    Run provider searches (blocking) in threadpool and normalize results to a common shape.
    """
    loop = asyncio.get_running_loop()
    results_all = []
    for provider in providers:
        func = _PROVIDER_MAP.get(provider)
        if not func:
            continue
        try:
            raw = await loop.run_in_executor(_executor, func, query)
            items = raw.get("shopping_results") or raw.get("organic_results") or raw.get("results") or []
            for item in items:
                results_all.append(
                    {
                        "provider": provider,
                        "title": item.get("title") or item.get("name"),
                        "price": item.get("price") or item.get("extracted_price") or item.get("price_raw"),
                        "url": item.get("link") or item.get("product_link"),
                        "image_url": item.get("thumbnail") or item.get("image"),
                        "product_metadata": item,
                    }
                )
        except Exception as e:
            # don't raise — log and continue
            print(f"[provider error] {provider}: {e}")
            continue
    return results_all


# ---------- Index & store helpers ----------
def _add_to_text_index(vecs: np.ndarray, prod_ids: List[int]):
    global _text_count
    n = vecs.shape[0]
    faiss.normalize_L2(vecs)
    _text_index.add(vecs)
    for i in range(n):
        _text_id_map[_text_count + i] = prod_ids[i]
    _text_count += n


def _add_to_image_index(vecs: np.ndarray, prod_ids: List[int]):
    global _image_count
    n = vecs.shape[0]
    faiss.normalize_L2(vecs)
    _image_index.add(vecs)
    for i in range(n):
        _image_id_map[_image_count + i] = prod_ids[i]
    _image_count += n


def _search_faiss(index: faiss.IndexFlatIP, query_vec: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    if index.ntotal == 0:
        return []
    v = np.array([query_vec]).astype(np.float32)
    faiss.normalize_L2(v)
    distances, indices = index.search(v, k)
    out = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        out.append((int(idx), float(dist)))
    return out


# ---------- API endpoints ----------
@router.post("/search-upload")
async def search_and_store(
    query: str = Form(...),
    description: str = Form(...),
    providers: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Run provider searches, compute embeddings for the search, store metadata and index embeddings.
    """
    global _products, _next_id

    # parse providers
    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    if not provider_list:
        raise HTTPException(400, "providers is required (comma-separated)")

    # read image bytes
    image_bytes = await image.read()

    # run provider searches (blocking -> threadpool)
    results = await run_provider_searches_async(query, provider_list)

    # compute search-level embeddings (text + image)
    text_emb = await get_text_embedding_async(f"{query}. {description}")
    image_emb = await get_image_embedding_async(image_bytes)

    # Prepare vectors for FAISS and product entries
    text_vectors = []
    image_vectors = []
    added_prod_ids = []

    for item in results:
        pid = _next_id
        _next_id += 1

        price_dec = parse_price_to_decimal(item.get("price"))

        prod = {
            "id": pid,
            "search_id": str(uuid.uuid4()),  # create a dedicated search_id for each product? No — we want same search_id per call
            # but user asked for grouping by search -> set below to outer search id
            "query": query,
            "description": description,
            "provider": item.get("provider"),
            "title": item.get("title"),
            "price": str(item.get("price")) if item.get("price") is not None else None,
            "price_decimal": price_dec,
            "url": item.get("url"),
            "image_url": item.get("image_url"),
            "product_metadata": item.get("product_metadata"),
        }
        _products.append(prod)
        added_prod_ids.append(pid)
        text_vectors.append(normalize_vector(np.array(text_emb, dtype=np.float32)))
        image_vectors.append(normalize_vector(np.array(image_emb, dtype=np.float32)))

    # All products created in this call should share same search_id (for grouping)
    shared_search_id = uuid.uuid4()
    for prod in _products[-len(added_prod_ids) :]:
        prod["search_id"] = str(shared_search_id)

    # add to FAISS indexes
    if len(text_vectors) > 0:
        tv = np.stack(text_vectors).astype(np.float32)
        _add_to_text_index(tv, added_prod_ids)
    if len(image_vectors) > 0:
        iv = np.stack(image_vectors).astype(np.float32)
        _add_to_image_index(iv, added_prod_ids)

    return {"status": "stored", "count": len(added_prod_ids), "search_id": str(shared_search_id)}


@router.post("/search-results")
async def get_similar_results(
    search_id: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    provider: Optional[str] = Form(None),
    price_min: Optional[float] = Form(None),
    price_max: Optional[float] = Form(None),
    k: int = Form(20),
    text_weight: float = Form(0.7),
    image_weight: float = Form(0.3),
):
    """
    Search by query and/or image. If both provided, we compute both and combine scores.
    `text_weight` and `image_weight` control combination. Candidate merging strategy:
    - run both indexes, union candidate ids, compute combined score (weighted sum of normalized similarities).
    """

    if not query and not image:
        raise HTTPException(400, "Either query or image must be provided for similarity search")

    candidates: Dict[int, float] = {}  # product_id -> combined score accumulator

    # text search
    if query:
        text_emb = await get_text_embedding_async(query)
        text_res = _search_faiss(_text_index, normalize_vector(np.array(text_emb, dtype=np.float32)), k)
        # text_res: list of (faiss_idx, score). map to product ids
        # normalize scores to [0,1] by simple transformation (faiss ip on normalized vecs -> cosine in [-1,1])
        for idx, score in text_res:
            pid = _text_id_map.get(idx)
            if pid is None:
                continue
            # rescale cosine from [-1,1] -> [0,1]
            s = (score + 1.0) / 2.0
            candidates[pid] = candidates.get(pid, 0.0) + text_weight * s

    # image search
    if image:
        image_bytes = await image.read()
        image_emb = await get_image_embedding_async(image_bytes)
        img_res = _search_faiss(_image_index, normalize_vector(np.array(image_emb, dtype=np.float32)), k)
        for idx, score in img_res:
            pid = _image_id_map.get(idx)
            if pid is None:
                continue
            s = (score + 1.0) / 2.0
            candidates[pid] = candidates.get(pid, 0.0) + image_weight * s

    # no candidates -> empty
    if not candidates:
        return []

    # assemble candidate list and apply filters
    # sort by combined score desc
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    results_out = []
    for pid, combined_score in ranked:
        # fetch product
        prod = next((p for p in _products if p["id"] == pid), None)
        if prod is None:
            continue
        # filter by search_id if provided
        if search_id and prod.get("search_id") != search_id:
            continue
        # provider filter
        if provider and prod.get("provider") != provider:
            continue
        # price filter
        pd = prod.get("price_decimal")
        if price_min is not None:
            if pd is None or float(pd) < float(price_min):
                continue
        if price_max is not None:
            if pd is None or float(pd) > float(price_max):
                continue

        out = {
            "id": prod["id"],
            "search_id": prod["search_id"],
            "provider": prod["provider"],
            "title": prod["title"],
            "price": prod["price"],
            "url": prod["url"],
            "image_url": prod["image_url"],
            "product_metadata": prod["product_metadata"],
            "score": combined_score,
        }
        results_out.append(out)
        if len(results_out) >= k:
            break

    return results_out


# small helper route to list stored searches (lightweight)
@router.get("/stored-searches")
async def list_searches():
    grouped = {}
    for p in _products:
        gid = p["search_id"]
        grouped.setdefault(gid, 0)
        grouped[gid] += 1
    return [{"search_id": k, "count": v} for k, v in grouped.items()]


# get products by search_id
@router.get("/by-search/{search_id}")
async def get_by_search(search_id: str, limit: int = 50):
    out = [p for p in _products if p["search_id"] == search_id][:limit]
    return out


app.include_router(router)
