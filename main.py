import os
import io
import uuid

from fastapi import FastAPI, UploadFile, File
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel
from src.db.connection import client as DB_Client
from src.shared.constants.db import BOOK_COLLECTION_NAME, VECTOR_SIZE
from qdrant_client.models import VectorParams, Distance, PointStruct




# =========================
# APP INIT
# =========================
app = FastAPI()
 


# =========================
# LOAD CLIP MODEL (ONCE)
# =========================
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    vector = features[0].cpu().numpy()
    return vector.tolist()





# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "Server is running"}


# =========================
# UPLOAD BOOK (INDEX)
# =========================
@app.post("/upload-book")
async def upload_book(
    file: UploadFile = File(...),
    title: str = "",
    author: str = ""
):
    if not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    vector = get_image_embedding(image)

    book_id = str(uuid.uuid4())

    DB_Client.upsert(
        collection_name=BOOK_COLLECTION_NAME,
        points=[
            PointStruct(
                id=book_id,
                vector=vector,
                payload={
                    "title": title,
                    "author": author
                }
            )
        ]
    )

    return {
        "status": "stored",
        "book_id": book_id
    }


# =========================
# SEARCH BOOK BY IMAGE
# =========================
@app.post("/search-book")
async def search_book(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    vector = get_image_embedding(image)

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