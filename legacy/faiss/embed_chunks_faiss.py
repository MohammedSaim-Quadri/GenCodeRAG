"""
Legacy FAISS-based embedding pipeline.

This file is archived because the project now uses Qdrant
as the primary vector database for retrieval.

Kept only for historical reference and experimentation.
Not used in active production flow.
"""

from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
from pathlib import Path

CHUNK_FILE = Path("data/chunks/github_code_chunks.jsonl")
INDEX_FILE = Path("data/faiss/github_code.index")
META_FILE = Path("data/faiss/github_code.meta.json")
INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load model
print("🧠 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Or use a code-specific model

# Load code chunks
print("📦 Reading code chunks...")
chunks = [json.loads(line) for line in open(CHUNK_FILE, "r", encoding="utf-8")]
texts = [chunk["code"] for chunk in chunks]

# Compute embeddings
print(f"🔢 Computing embeddings for {len(texts)} chunks...")
embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)

# Create FAISS index
print("⚙️ Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index
faiss.write_index(index, str(INDEX_FILE))
print(f"✅ Saved FAISS index to {INDEX_FILE}")

# Save metadata
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Saved metadata to {META_FILE}")