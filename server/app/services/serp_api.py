from serpapi import GoogleSearch

from app.core.config import settings

def google_search(query:str):
    search = GoogleSearch({
        "q": query, 
        "location": settings.LOCATION,
        "api_key": settings.SERPAI_KEY
    })
    result = search.get_dict()
    return result