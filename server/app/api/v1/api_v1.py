from fastapi import APIRouter

from app.services.serp_api import google_search, google_shopping_search, yahoo_shopping_search, ebay_search, walmart_search, home_depot_search

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

@router.post("/yahoo-shopping-search")
async def get_yahoo_shopping_results(query: str):
    """
    Perform yahoo shopping search and return results
    """

    result = await yahoo_shopping_search(query=query)
    return result

@router.post("/ebay-search")
async def get_ebay_results(query: str):
    """
    Perform ebay search and return results
    """

    result = await ebay_search(query=query)
    return result

@router.post("/walmart-search")
async def get_walmart_results(query: str):
    """
    Perform walmart search and return results
    """

    result = await walmart_search(query=query)
    return result

@router.post("/home-depot-search")
async def get_home_depot_results(query: str):
    """
    Perform Home Depot search and return results
    """

    result = await home_depot_search(query=query)
    return result