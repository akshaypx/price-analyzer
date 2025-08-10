import uuid
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.services.serp_api import (
    google_shopping_search,
    run_provider_searches,
    yahoo_shopping_search,
    ebay_search,
    walmart_search,
    home_depot_search,
    amazon_search
)
from app.db.crud import store_product_results
from app.db.db import get_db
from app.services.embedder import get_text_embedding, get_image_embedding

router = APIRouter()

@router.post("/google")
async def get_google_shopping_results(query: str):
    return await google_shopping_search(query=query)

@router.post("/yahoo")
async def get_yahoo_shopping_results(query: str):
    return await yahoo_shopping_search(query=query)

@router.post("/ebay")
async def get_ebay_results(query: str):
    return await ebay_search(query=query)

@router.post("/walmart")
async def get_walmart_results(query: str):
    return await walmart_search(query=query)

@router.post("/home-depot")
async def get_home_depot_results(query: str):
    return await home_depot_search(query=query)

@router.post("/amazon")
async def get_amazon_results(query: str):
    return await amazon_search(query=query)


@router.post("/search-upload")
async def search_and_store(
    query: str = Form(...),
    description: str = Form(...),
    providers: str = Form(...),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    search_id = uuid.uuid4()  # Generate UUID object
    image_bytes = await image.read()
    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    results = await run_provider_searches(query, provider_list)

    for r in results:
        r["search_id"] = search_id  # Keep as UUID

    await store_product_results(
        db=db,
        search_id=search_id,  # Pass UUID directly
        query=query,
        description=description,
        image_bytes=image_bytes,
        results=results
    )

    return {"status": "stored", "count": len(results), "search_id": str(search_id)}


@router.post("/search-results")
async def get_similar_results(
    search_id: str = Form(None),  # Accept string in request
    query: str = Form(None),
    image: UploadFile = File(None),
    provider: str = Form(None),
    price_min: float = Form(None),
    price_max: float = Form(None),
    db: AsyncSession = Depends(get_db)
):
    where_clauses = []
    params = {}

    if search_id:
        # Convert string to UUID object for Postgres
        where_clauses.append("search_id = :search_id")
        params["search_id"] = uuid.UUID(search_id)

    if query:
        text_emb = await get_text_embedding(query)
        where_clauses.append("text_embedding <=> :text_emb < 0.5")
        params["text_emb"] = text_emb

    if image:
        image_bytes = await image.read()
        image_emb = await get_image_embedding(image_bytes)
        where_clauses.append("image_embedding <=> :image_emb < 0.5")
        params["image_emb"] = image_emb

    if provider:
        where_clauses.append("provider = :provider")
        params["provider"] = provider

    if price_min is not None:
        where_clauses.append("CAST(price AS FLOAT) >= :price_min")
        params["price_min"] = price_min

    if price_max is not None:
        where_clauses.append("CAST(price AS FLOAT) <= :price_max")
        params["price_max"] = price_max

    where = " AND ".join(where_clauses) if where_clauses else "TRUE"
    sql = f"SELECT * FROM product_results WHERE {where} ORDER BY created_at DESC LIMIT 50"

    result = await db.execute(text(sql), params)
    return [dict(r._mapping) for r in result]
