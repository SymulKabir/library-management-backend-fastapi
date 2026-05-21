from fastapi import APIRouter, UploadFile, File
from .book.router import router as book_router

router = APIRouter()

router.include_router(book_router, prefix="/book", tags=["Books"])