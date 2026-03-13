"""
RAG Engine for Sanctions-CGSS Textbook
Fixed version with proper error handling and multiple model fallbacks
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import httpx

# ─────────────────────────────────────────────────────────────
# STEP 1: CHUNK
# ─────────────────────────────────────────────────────────────

def chunk(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += chunk_size - overlap
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def extract_keywords(text: str) -> List[str]:
    """Extract sanctions-related keywords."""
    terms = [
        'sanctions', 'ofac', 'compliance', 'freeze', 'blocked', 'ownership',
        'control', 'license', 'evasion', 'screening', 'due diligence', '50% rule',
        'ubo', 'beneficial', 'asset', 'freezing', 'designated', 'sdn', 'list',
        'un', 'european', 'uk', 'treasury', 'investigation', 'reporting',
        'iran', 'russia', 'north korea', 'cuba', 'syria', 'crimea'
    ]
    
    found = []
    text_lower = text.lower()
    for term in terms:
        if term in text_lower:
            found.append(term)
    return list(set(found))[:5]

# ─────────────────────────────────────────────────────────────
# STEP 2: VECTOR STORE
# ─────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.chunks = []
        self.metadata = []
        self.matrix = None
        self.is_fitted = False
        self.document_name = None

    def add_documents(self, chunks, meta, doc_name=None):
        self.chunks.extend(chunks)
        self.metadata.extend(meta)
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        self.is_fitted = True
        if doc_name:
            self.document_name = doc_name
        return len(chunks)

    def embed_query(self, query):
        return self.vectorizer.transform([query])

    def count(self):
        return len(self.chunks)

    def is_loaded(self):
        return self.count() > 0

# ─────────────────────────────────────────────────────────────
# STEP 3: RETRIEVE
# ─────────────────────────────────────────────────────────────

def retrieve(store, query, top_k=5):
    if not store.is_fitted or store.count() == 0:
        return []
    
    q_vec = store.embed_query(query)
    scores = cosine_similarity(q_vec, store.matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0.01:
            results.append({
                "text": store.chunks[idx],
                "score": float(scores[idx]),
                "source": store.metadata[idx]["source"],
                "chunk_index": store.metadata[idx]["chunk_index"],
                "keywords": extract_keywords(store.chunks[idx])
            })
    return results

# ─────────────────────────────────────────────────────────────
# STEP 4: GENERATE WITH MULTIPLE MODEL FALLBACKS
# ─────────────────────────────────────────────────────────────

def generate(query, chunks, api_key):
    if not chunks:
        return "No relevant information found in the document.", ""
    
    # Build context from chunks
    context_parts = []
    for i, ch in enumerate(chunks, 1):
        text = ch['text'].replace('\n', ' ').strip()
        if len(text) > 500:
            text = text[:500] + "..."
        context_parts.append(f"[Section {i} - Score: {ch['score']}]\n{text}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a Certified Global Sanctions Specialist (CGSS) expert. Answer the question using ONLY the context below.

Context:
{context}

Question: {query}

Answer:"""

    # List of models to try in order (free models first, then paid)
    models = [
        "mistralai/mistral-7b-instruct:free",     # Free Mistral
        "google/gemma-2-9b-it:free",               # Free Gemma  
        "meta-llama/llama-3-8b-instruct:free",    # Free Llama
        "microsoft/phi-3-mini-128k-instruct:free", # Free Phi-3
        "mistralai/mistral-7b-instruct",           # Paid Mistral
        "openai/gpt-3.5-turbo",                     # Paid GPT
    ]
    
    errors = []
    
    for model in models:
        try:
            print(f"🔄 Trying model: {model}")
            
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8001",
                    "X-Title": "CGSS Sanctions Assistant",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a sanctions compliance expert. Answer based only on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.3,  # Lower temperature for more factual answers
                },
                timeout=45
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                print(f"✅ Success with model: {model}")
                return answer, prompt
            else:
                error_msg = f"Model {model} failed: HTTP {response.status_code}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
                
                # Try to get more error details
                try:
                    error_detail = response.json()
                    print(f"   Details: {error_detail}")
                except:
                    pass
                    
        except Exception as e:
            error_msg = f"Model {model} error: {str(e)}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
    
    # If all models fail, return error message
    error_summary = "\n".join(errors[:3])  # Show first 3 errors
    return f"Error: All models failed. Please check your API key or try again later.\n\nDetails:\n{error_summary}", prompt

# ─────────────────────────────────────────────────────────────
# STEP 5: LOAD DOCUMENT
# ─────────────────────────────────────────────────────────────

def load_sanctions_document(store, path):
    """Load the Sanctions-CGSS document."""
    try:
        print(f"📖 Loading: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"📊 Size: {len(text):,} characters")
        
        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Create chunks
        chunks = chunk(text, chunk_size=600, overlap=120)
        
        # Create metadata
        source = os.path.basename(path)
        meta = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        
        # Add to store
        store.add_documents(chunks, meta, source)
        
        print(f"✅ Loaded {len(chunks)} chunks")
        return True
        
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return False
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return False
