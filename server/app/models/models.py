from sqlalchemy import Column, String, DateTime, Integer, JSON
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.db import Base

class ProductResult(Base):
    __tablename__ = "product_results"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String)
    description = Column(String)
    provider = Column(String)
    title = Column(String)
    price = Column(String)
    url = Column(String)
    image_url = Column(String)
    text_embedding = Column(Vector(1536))
    image_embedding = Column(Vector(512))
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
