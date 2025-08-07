from serpapi import GoogleSearch

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