from fastapi import APIRouter

from app.services.serp_api import google_search

router = APIRouter()

@router.post("/google")
async def get_google_search_results(query:str):
    """
    Perform google search and result output.
    """
    result = await google_search(query=query)
    return result