from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import tempfile

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

# Global variables for RAG components
vectorstore = None
qa_chain = None

class Query(BaseModel):
    text: str

class Document(BaseModel):
    text: str
    source: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global vectorstore, qa_chain
    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not set")
            return
            
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Try to load existing vectorstore or create new one
        persist_directory = "/tmp/chroma_db"  # Vercel's writable directory
        
        if os.path.exists(persist_directory):
            # Load existing vectorstore
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print(f"Loaded existing vectorstore with {vectorstore._collection.count()} documents")
        else:
            # Create new vectorstore
            os.makedirs(persist_directory, exist_ok=True)
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            print("Created new vectorstore")
        
        # Initialize QA chain
        if vectorstore and vectorstore._collection.count() > 0:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            print("QA chain initialized")
        else:
            print("No documents in vectorstore - QA chain not initialized")
            
    except Exception as e:
        print(f"Error initializing RAG: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Sanctions RAG API is running",
        "status": "active",
        "rag_ready": qa_chain is not None
    }

@app.get("/status")
async def status():
    doc_count = 0
    if vectorstore:
        try:
            doc_count = vectorstore._collection.count()
        except:
            pass
    
    return {
        "status": "online",
        "environment": os.getenv("VERCEL_ENV", "development"),
        "rag_initialized": qa_chain is not None,
        "documents_loaded": doc_count,
        "openai_key_set": os.getenv("OPENAI_API_KEY") is not None
    }

@app.post("/query")
async def query(query: Query):
    try:
        if qa_chain:
            # Use RAG to answer
            response = qa_chain.run(query.text)
            return {
                "response": response,
                "status": "success",
                "source": "rag"
            }
        else:
            # Fallback response if RAG not ready
            return {
                "response": f"RAG system not fully initialized. Please add documents first. Your query was: {query.text}",
                "status": "warning",
                "source": "fallback"
            }
    except Exception as e:
        return {
            "response": f"Error processing query: {str(e)}",
            "status": "error"
        }

@app.post("/add_documents")
async def add_documents(docs: list[Document]):
    """Add documents to the vectorstore"""
    global vectorstore, qa_chain
    
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "message": "OpenAI API key not set",
                "status": "error"
            }
            
        embeddings = OpenAIEmbeddings()
        persist_directory = "/tmp/chroma_db"
        os.makedirs(persist_directory, exist_ok=True)
        
        if not vectorstore:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Add documents to vectorstore
        texts = [doc.text for doc in docs]
        metadatas = [{"source": doc.source} for doc in docs]
        
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        vectorstore.persist()
        
        # Reinitialize QA chain
        if vectorstore._collection.count() > 0:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )
        
        return {
            "message": f"Added {len(docs)} documents",
            "total_documents": vectorstore._collection.count(),
            "status": "success"
        }
    except Exception as e:
        return {
            "message": f"Error adding documents: {str(e)}",
            "status": "error"
        }

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
