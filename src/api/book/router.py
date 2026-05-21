import io
import uuid
from fastapi import  UploadFile, File, APIRouter
from PIL import Image
from src.db.connection import client as DB_Client
from src.shared.constants.db import BOOK_COLLECTION_NAME
from qdrant_client.models import PointStruct
from src.utils.embedding import get_img_embedding
 
router = APIRouter()
 
@router.post("/upload")
async def upload_book(
    file: UploadFile = File(...),
    title: str = "",
    author: str = ""
):
    if not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    vector = get_img_embedding(image)

    book_id = str(uuid.uuid4())

    DB_Client.upsert(
        collection_name=BOOK_COLLECTION_NAME,
        points=[
            PointStruct(
                id=book_id,
                vector=vector
            )
        ]
    )

    return {
        "status": "stored",
        "book_id": book_id
    }

@router.post("/search")
async def search_book(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    vector = get_img_embedding(image)

    results = DB_Client.search(
        collection_name=BOOK_COLLECTION_NAME,
        query_vector=vector,
        limit=5
    )

    return [
        {
            "id": r.id,
            "score": r.score,
            "title": r.payload.get("title"),
            "author": r.payload.get("author"),
        }
        for r in results
    ]
    
    