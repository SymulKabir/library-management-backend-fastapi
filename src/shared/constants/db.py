import os

DB_URL = os.getenv("VECTOR_DATABASE_URI", "http://localhost:6333")

BOOK_COLLECTION_NAME = "BookCovers"

VECTOR_SIZE = 512  
