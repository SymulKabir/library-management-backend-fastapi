from fastapi import FastAPI, File

from src.db.connection import client as DB_Client
from src.api.router import router as api_router



app = FastAPI()
app.include_router(api_router, prefix='/api')

@app.get("/")
def home():
    return {"status": "Server is running"}


