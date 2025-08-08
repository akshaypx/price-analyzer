from sqlalchemy.ext.asyncio import AsyncSession
from app.models import ProductResult
from sqlalchemy import insert
from app.services.embedder import get_text_embedding, get_image_embedding
from typing import List

async def store_product_results(
    db: AsyncSession,
    query: str,
    description: str,
    image_bytes: bytes,
    results: List[dict]
):
    text_emb = await get_text_embedding(f"{query}. {description}")
    image_emb = await get_image_embedding(image_bytes)

    for item in results:
        stmt = insert(ProductResult).values(
            query=query,
            description=description,
            provider=item["provider"],
            title=item["title"],
            price=item["price"],
            url=item["url"],
            image_url=item["image_url"],
            text_embedding=text_emb,
            image_embedding=image_emb,
            product_metadata=item["product_metadata"]
        )
        await db.execute(stmt)

    await db.commit()
