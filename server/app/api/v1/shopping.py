from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.serp_api import google_shopping_search, run_provider_searches, yahoo_shopping_search, ebay_search, walmart_search, home_depot_search, amazon_search
from app.db.crud import store_product_results
from app.db.db import get_db
from app.schemas.schemas import SearchRequest

router = APIRouter()

@router.post("/google")
async def get_google_shopping_results(query: str):
    """
    Perform google shopping search and return results
    """

    result = await google_shopping_search(query=query)
    return result

@router.post("/yahoo")
async def get_yahoo_shopping_results(query: str):
    """
    Perform yahoo shopping search and return results
    """

    result = await yahoo_shopping_search(query=query)
    return result

@router.post("/ebay")
async def get_ebay_results(query: str):
    """
    Perform ebay search and return results
    """

    result = await ebay_search(query=query)
    return result

@router.post("/walmart")
async def get_walmart_results(query: str):
    """
    Perform walmart search and return results
    """

    result = await walmart_search(query=query)
    return result

@router.post("/home-depot")
async def get_home_depot_results(query: str):
    """
    Perform Home Depot search and return results
    """

    result = await home_depot_search(query=query)
    return result

@router.post("/amazon")
async def get_amazon_results(query: str):
    """
    Perform Amazon search and return results
    """

    result = await amazon_search(query=query)
    return result

@router.post("/search-upload")
async def search_and_store(
    query: str = Form(...),
    description: str = Form(...),
    providers: str = Form(...),  # comma-separated
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    image_bytes = await image.read()
    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    results = await run_provider_searches(query, provider_list)

    await store_product_results(
        db=db,
        query=query,
        description=description,
        image_bytes=image_bytes,
        results=results
    )

    return {"status": "stored", "count": len(results)}