import io
import uuid
from fastapi import  UploadFile, File, APIRouter, Request
from PIL import Image
from src.db.connection import client as DB_Client, init_db
from src.shared.constants.db import BOOK_COLLECTION_NAME
from qdrant_client.models import PointStruct, Distance, VectorParams
from src.utils.embedding import get_img_embedding
from pydantic import BaseModel
import urllib.request 
import httpx
from src.shared.constants.url import MAIN_BACKEND_URL
 
router = APIRouter()

 
@router.post("/upload")
async def upload_book(request: Request): 
    body = await request.json()  # Don't forget 'await' on request.json()
    image_url = body.get('image_url')
    
    if not image_url:
        return {"message": "Image url is required"}

    try:
        # 1. Download the image bytes using standard python libraries
        req = urllib.request.Request(
            image_url, 
            headers={'User-Agent': 'Mozilla/5.0'} # Prevents 403 blocks from sites
        )
        
        with urllib.request.urlopen(req) as response:
            image_bytes = response.read()

        # 2. Convert bytes to an in-memory stream and pass to PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        

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
            "vector_id": book_id
        }
    except Exception as error:
        print("error --->>", error)
        return {"message": "Internal server error"}



@router.post("/search")
async def search_book(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Only image files allowed"}

        image = Image.open(io.BytesIO(await file.read())).convert("RGB")

        vector = get_img_embedding(image)
        
        results = DB_Client.query_points(
            collection_name=BOOK_COLLECTION_NAME,
            query=vector,
            limit=10,
            with_payload=True   
        )
        # ✅ SAFE CHECK (IMPORTANT)
        if not results.points or len(results.points) == 0:
            return {"status": "book not matched"}
        results = results.points
        vector_ids = [product.id for product in results if product.score > 0.7]
        
        
        res = httpx.post(
            f"{MAIN_BACKEND_URL}/books/get/by-vector-id",
            json={"vector_ids": vector_ids},
            timeout=10.0
        )

        jsonRes = res.json()

        return {"data": jsonRes.get("data") or []}
    except Exception as error:
        print("error --->>", error)
        return {"message": "Internal server error"}

    

    
    
@router.delete("/delete-all")
async def delete_all_book():
    try:
        # 1. Delete the collection entirely
        DB_Client.delete_collection(collection_name=BOOK_COLLECTION_NAME)
        
        init_db()
        return {"status": "success", "message": "All items cleared and collection reset."}
        
    except Exception as error:
        print("error --->>", error)
        return {"message": "Internal server error"}
    
    