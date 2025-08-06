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