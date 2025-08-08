from serpapi import GoogleSearch
from typing import List, Callable

from app.core.config import settings

async def google_search(query:str):
    search = GoogleSearch({
        "q": query, 
        "location": settings.LOCATION,
        "api_key": settings.SERPAI_KEY
    })
    result = search.get_dict()
    return result

async def google_shopping_search(query: str):
    params = {
        "api_key": settings.SERPAI_KEY,
        "engine": "google_shopping",
        "google_domain": "google.com",
        "q": query,
        "hl": "en",
        "location": settings.LOCATION
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

async def yahoo_shopping_search(query:str):
    params = {
    "api_key": settings.SERPAI_KEY,
    "engine": "yahoo_shopping",
    "p": query
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

async def ebay_search(query:str):
    params = {
    "api_key": settings.SERPAI_KEY,
    "engine": "ebay",
    "_nkw": query
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

async def walmart_search(query:str):
    params = {
    "currentUser": "[object Object]",
    "api_key": settings.SERPAI_KEY,
    "engine": "walmart",
    "query": query
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

async def home_depot_search(query:str):
    params = {
    "country": "us",
    "api_key": settings.SERPAI_KEY,
    "engine": "home_depot",
    "q": query
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

async def amazon_search(query:str):
    params = {
    "engine": "amazon",
    "k": query,
    "amazon_domain": "amazon.com",
    "api_key": settings.SERPAI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

PROVIDER_FUNCTION_MAP: dict[str, Callable[[str], dict]] = {
    "google_shopping": google_shopping_search,
    "yahoo_shopping": yahoo_shopping_search,
    "ebay": ebay_search,
    "walmart": walmart_search,
    "home_depot": home_depot_search,
    "amazon": amazon_search,
}

async def run_provider_searches(query: str, providers: List[str]) -> List[dict]:
    all_results = []

    for provider in providers:
        search_func = PROVIDER_FUNCTION_MAP.get(provider)

        if not search_func:
            continue

        try:
            result = await search_func(query)

            items = result.get("shopping_results") or result.get("organic_results") or []
            
            for item in items:
                all_results.append({
                    "query": query,
                    "provider": provider,
                    "title": item.get("title"),
                    "price": item.get("price") or item.get("extracted_price"),
                    "url": item.get("link"),
                    "image_url": item.get("thumbnail") or item.get("image"),
                    "product_metadata": item
                })
        except Exception as e:
            print(f"Error fetching from {provider}: {e}")
            continue

    return all_results