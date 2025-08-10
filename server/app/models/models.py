# app/models.py (excerpt)
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID, NUMERIC
from sqlalchemy.sql import func
import uuid
from pgvector.sqlalchemy import Vector
from app.db import Base

class ProductResult(Base):
    __tablename__ = "product_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(UUID(as_uuid=True), nullable=False, index=True, default=uuid.uuid4)
    query = Column(String)
    description = Column(String)
    provider = Column(String)
    title = Column(String)
    price = Column(NUMERIC(12, 2), nullable=True)           # numeric column
    url = Column(String)
    image_url = Column(String)
    text_embedding = Column(Vector(1536))   # if using pgvector
    image_embedding = Column(Vector(512))
    product_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
