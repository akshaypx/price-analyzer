from fastapi import APIRouter

from app.services.serp_api import google_search, google_shopping_search

router = APIRouter()

@router.post("/google-search")
async def get_google_search_results(query:str):
    """
    Perform google search and result output.
    """
    result = await google_search(query=query)
    return result

@router.post("/google-shopping-search")
async def get_google_shopping_results(query: str):
    """
    Perform google shopping search and return results
    """

    result = await google_shopping_search(query=query)
    return result