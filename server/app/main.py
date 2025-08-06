from fastapi import FastAPI

from app.api.v1 import analyzer

app = FastAPI(
    title="Price Analyzer API",
    version="1.0.0"
)

app.include_router(analyzer.router, prefix="/api/v1")

@app.get("/")
def test():
    return {"message": "Price Analyzer API is running."}