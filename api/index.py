from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

@app.get("/")
async def root():
    return {
        "message": "Sanctions RAG API is running",
        "status": "active",
        "openai_key_set": os.getenv("OPENAI_API_KEY") is not None
    }

@app.get("/status")
async def status():
    return {
        "status": "online",
        "environment": os.getenv("VERCEL_ENV", "development"),
        "openai_key_set": os.getenv("OPENAI_API_KEY") is not None
    }

@app.post("/query")
async def query(query: Query):
    return {
        "response": f"Processing: {query.text}",
        "status": "success"
    }

@app.post("/add_documents")
async def add_documents():
    return {
        "message": "Document upload coming soon",
        "status": "pending"
    }
