from fastapi import FastAPI

from app.api.v1 import shopping, search

app = FastAPI(
    title="Price Analyzer API",
    version="1.0.0"
)

app.include_router(shopping.router, prefix="/api/v1/shopping")
app.include_router(search.router, prefix="/api/v1/search")

@app.get("/")
def test():
    return {"message": "Price Analyzer API is running."}