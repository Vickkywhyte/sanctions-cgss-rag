from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
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

@app.get("/")
async def root():
    return {
        "message": "Sanctions RAG API is running",
        "status": "active"
    }

@app.get("/status")
async def status():
    return {
        "status": "online",
        "environment": os.getenv("VERCEL_ENV", "development")
    }

@app.post("/query")
async def query(query: Query):
    # Your RAG logic will go here
    return {
        "response": f"Processing query: {query.text}",
        "status": "success"
    }

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
