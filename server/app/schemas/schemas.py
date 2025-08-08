from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    description: str
    image_url: HttpUrl
    providers: List[str]

class ProductOut(BaseModel):
    id: int
    title: str
    price: Optional[str]
    url: HttpUrl
    image_url: HttpUrl
    provider: str
    product_metadata: dict

    class Config:
        orm_mode = True
