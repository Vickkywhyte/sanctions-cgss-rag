"""
Main FastAPI application for CGSS Sanctions Assistant
Modified to run on port 8001 (different from original)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import uvicorn

from rag_engine import (
    VectorStore, retrieve, generate,
    load_sanctions_document, extract_keywords
)

app = FastAPI(title="CGSS Sanctions Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
store = VectorStore()

# Load document on startup
@app.on_event("startup")
async def load_document():
    global store
    doc_path = "Sanctions-CGSS.txt"
    if os.path.exists(doc_path):
        load_sanctions_document(store, doc_path)
        print(f"✅ Document loaded. Total chunks: {store.count()}")
    else:
        print(f"⚠️ Warning: {doc_path} not found")

class QueryRequest(BaseModel):
    query: str
    api_key: str
    top_k: Optional[int] = 8

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Dict]
    prompt_used: str
    chunk_count: int

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serve the HTML interface."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>CGSS Sanctions Assistant</h1><p>Upload your index.html file to this directory.</p>")

@app.get("/status")
async def get_status():
    """Get the current status of the vector store."""
    return {
        "document_loaded": store.is_loaded(),
        "chunk_count": store.count(),
        "document_name": store.document_name if hasattr(store, 'document_name') else "Unknown",
        "chunk_size": "600 words (adjusted for textbook)"
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return the answer."""
    if not store.is_loaded():
        raise HTTPException(status_code=400, detail="No document loaded")
    
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key required")
    
    # Retrieve relevant chunks
    chunks = retrieve(store, request.query, top_k=request.top_k)
    
    if not chunks:
        return QueryResponse(
            answer="No relevant information found in the CGSS study guide.",
            retrieved_chunks=[],
            prompt_used="No context available",
            chunk_count=0
        )
    
    # Generate answer
    answer, prompt = generate(request.query, chunks, request.api_key)
    
    return QueryResponse(
        answer=answer,
        retrieved_chunks=chunks,
        prompt_used=prompt,
        chunk_count=len(chunks)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port!