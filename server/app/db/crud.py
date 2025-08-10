from sqlalchemy.ext.asyncio import AsyncSession
from app.models import ProductResult
from sqlalchemy import insert
from app.services.embedder import get_text_embedding, get_image_embedding
from typing import List, Optional
import uuid
import json
import re
from decimal import Decimal, InvalidOperation
import numpy as np
from sqlalchemy import types as sqltypes

async def store_product_results(
    db: AsyncSession,
    query: str,
    description: str,
    image_bytes: bytes,
    results: List[dict],
    search_id: Optional[uuid.UUID] = None
):
    """
    Robust storage: normalizes embeddings, price and metadata to safe python types
    and chooses the correct python value to pass based on the column type.
    Returns the search_id (UUID).
    """

    # normalize/generate search_id as uuid.UUID
    if search_id is None:
        search_id = uuid.uuid4()
    elif isinstance(search_id, str):
        search_id = uuid.UUID(search_id)

    # compute embeddings once per search (you already did this pattern previously).
    text_emb_raw = await get_text_embedding(f"{query}. {description}")
    image_emb_raw = await get_image_embedding(image_bytes)

    def ensure_float_list(x):
        """Return a Python list[float] or raise ValueError."""
        if x is None:
            return None
        # if it's already numpy
        if isinstance(x, np.ndarray):
            arr = x.tolist()
        # common case: list/tuple of numerics
        elif isinstance(x, (list, tuple)):
            arr = list(x)
        # if it's a JSON string like "[0.1, 0.2]"
        elif isinstance(x, str):
            try:
                arr = json.loads(x)
            except Exception:
                # fallback to eval-like parsing
                try:
                    arr = eval(x)  # only trusted internal strings (defensive)
                except Exception as e:
                    raise ValueError("Can't parse embedding string") from e
        else:
            raise ValueError("Unsupported embedding type")

        # force float for all elements
        return [float(v) for v in arr]

    # normalize embeddings
    try:
        text_embedding = ensure_float_list(text_emb_raw)
    except Exception:
        # If embedding generation failed for some reason, set to None
        text_embedding = None

    try:
        image_embedding = ensure_float_list(image_emb_raw)
    except Exception:
        image_embedding = None

    # Inspect DB column type for price at runtime so we insert the correct python type
    price_col_type = ProductResult.__table__.c.price.type  # type: ignore

    def parse_price_to_decimal(val):
        if val is None:
            return None
        if isinstance(val, Decimal):
            return val.quantize(Decimal("0.01"))
        if isinstance(val, (int, float)):
            return Decimal(str(val)).quantize(Decimal("0.01"))
        if isinstance(val, str):
            # remove currency symbols and commas, preserve minus and dots
            cleaned = re.sub(r"[^\d\.\-]", "", val)
            if not cleaned:
                return None
            try:
                return Decimal(cleaned).quantize(Decimal("0.01"))
            except InvalidOperation:
                return None
        return None

    # iterate results defensively
    for item in results:
        provider = item.get("provider")
        title = item.get("title")
        raw_price = item.get("price")
        url = item.get("url")
        image_url = item.get("image_url")
        metadata = item.get("product_metadata", None)

        # normalize metadata to python dict (if it's a JSON string)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                # keep raw string if JSON parsing fails
                metadata = {"raw": metadata}

        # compute `price_db_val` based on the declared column type
        parsed_price = parse_price_to_decimal(raw_price)
        if isinstance(price_col_type, sqltypes.Numeric):
            # Column expects numeric -> pass Decimal (or None)
            price_db_val = parsed_price
        else:
            # Column expects TEXT/VARCHAR -> pass string
            price_db_val = None if raw_price is None else str(raw_price)

        # final values to insert (ensure embeddings are lists, not JSON strings)
        stmt = insert(ProductResult).values(
            search_id=search_id,
            query=query,
            description=description,
            provider=provider,
            title=title,
            price=price_db_val,
            url=url,
            image_url=image_url,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            product_metadata=metadata,
        )

        # execute per-item (you can batch this if you want later)
        await db.execute(stmt)

    await db.commit()
    return str(search_id)
