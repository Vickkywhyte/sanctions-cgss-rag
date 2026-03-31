from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

class Document(BaseModel):
    text: str
    source: str

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
        "environment": "development",
        "openai_key_set": os.getenv("OPENAI_API_KEY") is not None
    }

@app.post("/query")
async def query(query: Query):
    # Simple response for now
    return {
        "response": f"Processing: {query.text}",
        "status": "success"
    }

@app.post("/add_documents")
async def add_documents(docs: list[Document]):
    return {
        "message": f"Received {len(docs)} documents",
        "status": "success"
    }
