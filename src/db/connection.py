import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.shared.constants.db import DB_URL, BOOK_COLLECTION_NAME, VECTOR_SIZE
 


# =========================
# QDRANT INIT
# =========================
def init_db():
    client = QdrantClient(
        url=DB_URL
    )

    collections = client.get_collections().collections
    exists = any(c.name == BOOK_COLLECTION_NAME for c in collections)

    if not exists:
        client.create_collection(
            collection_name=BOOK_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

    print("✅ Connected to Qdrant & collection ready")
    return client


client = init_db()