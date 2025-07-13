import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict, Any
import time
from datetime import datetime
import re

# Load API token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
print(f"Token loaded: {'✓' if HF_TOKEN else '✗'}")

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# === Configuration ===
CONFIG = {
    "top_k": 5,  # Increased for more context
    "temperature": 0.7,
    "max_tokens": 1024,
    "model": "deepseek-ai/DeepSeek-V3-0324",
    "similarity_threshold": 0.3,  # Filter out low-similarity chunks
    "max_context_length": 8000,  # Prevent context overflow
}

# === Load FAISS index and metadata ===
INDEX_PATH = "data/faiss/github_code.index"
META_PATH = "data/faiss/github_code.meta.json"

print("📦 Loading FAISS index + metadata...")
try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✓ Loaded {index.ntotal} embeddings and {len(metadata)} metadata entries")
except Exception as e:
    print(f"❌ Error loading index/metadata: {e}")
    exit(1)

# === Load embedding model ===
print("🔤 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model loaded")

def format_code_snippet(code: str, max_lines: int = 50) -> str:
    """Format and truncate code snippets for better readability."""
    lines = code.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ['...', '# Code truncated for brevity']
    return '\n'.join(lines)

def extract_language_from_path(path: str) -> str:
    """Extract programming language from file path."""
    extension_map = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
        '.go': 'go', '.rs': 'rust', '.php': 'php', '.rb': 'ruby',
        '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
        '.sql': 'sql', '.html': 'html', '.css': 'css', '.sh': 'bash',
        '.m': 'objective-c', '.jl': 'julia', '.lua': 'lua',
        '.dart': 'dart', '.r': 'r', '.pl': 'perl', '.json': 'json'
}
    ext = Path(path).suffix.lower()
    return extension_map.get(ext, 'text')

def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate code chunks based on content similarity."""
    seen_codes = set()
    unique_chunks = []
    
    for chunk in chunks:
        # Create a simple hash of the code content
        code_hash = hash(chunk['code'][:200])  # Use first 200 chars for comparison
        if code_hash not in seen_codes:
            seen_codes.add(code_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks

def enhance_retrieval(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Enhanced retrieval with filtering and ranking."""
    print(f"\n🔍 Searching for relevant code (top-{top_k})...")
    
    # Encode query
    query_embed = embed_model.encode([query])
    
    # Search with higher k to allow filtering
    search_k = min(top_k * 3, len(metadata))
    D, I = index.search(query_embed, search_k)
    
    # Filter and enrich results
    enhanced_chunks = []
    for idx, (distance, meta_idx) in enumerate(zip(D[0], I[0])):
        if meta_idx >= len(metadata):
            continue
            
        meta = metadata[meta_idx]
        similarity = 1 - distance  # Convert distance to similarity
        
        # Filter by similarity threshold
        if similarity < CONFIG["similarity_threshold"]:
            continue
            
        # Enhance metadata
        enhanced_meta = {
            **meta,
            "similarity": round(similarity, 3),
            "language": extract_language_from_path(meta.get("path", "")),
            "rank": idx + 1,
            "formatted_code": format_code_snippet(meta["code"])
        }
        
        enhanced_chunks.append(enhanced_meta)
    
    # Remove duplicates and limit to top_k
    unique_chunks = deduplicate_chunks(enhanced_chunks)[:top_k]
    
    return unique_chunks

def create_enriched_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Create a more sophisticated prompt with better context organization."""
    
    # Organize chunks by language
    chunks_by_lang = {}
    for chunk in chunks:
        lang = chunk.get("language", "text")
        if lang not in chunks_by_lang:
            chunks_by_lang[lang] = []
        chunks_by_lang[lang].append(chunk)
    
    # Build context sections
    context_sections = []
    total_length = 0
    
    for lang, lang_chunks in chunks_by_lang.items():
        lang_section = f"## {lang.upper()} Examples:\n"
        
        for chunk in lang_chunks:
            repo = chunk.get("repo", "unknown")
            path = chunk.get("path", "unknown")
            func_name = chunk.get("function_name", "unknown")
            similarity = chunk.get("similarity", 0)
            
            chunk_context = f"""
### Source: {repo}/{path}
**Function:** `{func_name}` (Similarity: {similarity:.1%})
```{lang}
{chunk['formatted_code']}
```
"""
            
            # Check if adding this chunk would exceed context limit
            if total_length + len(chunk_context) > CONFIG["max_context_length"]:
                break
                
            lang_section += chunk_context
            total_length += len(chunk_context)
        
        context_sections.append(lang_section)
    
    # Combine all sections
    full_context = "\n".join(context_sections)
    
    # Create the final prompt
    prompt = f"""You are an expert programmer with access to relevant code examples. Generate high-quality, production-ready code based on the following context and requirements.

# CODE EXAMPLES FOR REFERENCE:
{full_context}

# TASK:
{query}

# INSTRUCTIONS:
- Analyze the provided code examples for patterns and best practices
- Generate clean, well-documented, and efficient code
- Include appropriate error handling and edge cases
- Add helpful comments explaining the logic
- Follow the coding style and patterns shown in the examples
- If multiple languages are shown, choose the most appropriate one for the task

# RESPONSE:
"""
    
    return prompt

def display_retrieval_results(chunks: List[Dict[str, Any]]):
    """Display enhanced retrieval results."""
    print(f"\n📊 Found {len(chunks)} relevant code examples:")
    print("-" * 80)
    
    for i, chunk in enumerate(chunks, 1):
        repo = chunk.get("repo", "unknown")
        path = chunk.get("path", "unknown")
        func_name = chunk.get("function_name", "unknown")
        similarity = chunk.get("similarity", 0)
        language = chunk.get("language", "text")
        
        print(f"{i}. [{language.upper()}] {repo}/{path}")
        print(f"   Function: {func_name}")
        print(f"   Similarity: {similarity:.1%}")
        print(f"   Code Length: {len(chunk['code'])} chars")
        print("-" * 40)

def make_api_request(prompt: str) -> str:
    """Make API request with enhanced error handling and retry logic."""
    payload = {
        "model": CONFIG["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": CONFIG["temperature"],
        "max_tokens": CONFIG["max_tokens"],
        "stream": False
    }
    
    print(f"\n🚀 Querying {CONFIG['model']}...")
    print(f"📝 Context length: {len(prompt)} characters")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Display usage stats if available
                if "usage" in result:
                    usage = result["usage"]
                    print(f"📊 Tokens used: {usage.get('total_tokens', 'N/A')}")
                
                return content
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
                    return f"Failed after {max_retries} attempts. Last error: {response.text}"
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            if attempt < max_retries - 1:
                print(f"🔄 Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                return f"Request failed after {max_retries} attempts: {e}"
    
    return "Failed to get response from API"

def save_interaction(query: str, response: str, chunks: List[Dict[str, Any]]):
    """Save the interaction for future reference."""
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "retrieved_chunks": [
            {
                "repo": chunk.get("repo"),
                "path": chunk.get("path"),
                "function": chunk.get("function_name"),
                "similarity": chunk.get("similarity")
            }
            for chunk in chunks
        ],
        "config": CONFIG
    }
    
    # Save to interactions log
    log_path = Path("interactions.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(interaction) + "\n")

# === Main Execution ===
def main():
    print("🤖 Enhanced RAG Code Generation System")
    print("=" * 50)
    
    # Get user query
    query = input("💬 Enter your code generation prompt:\n> ")
    
    if not query.strip():
        print("❌ Empty query. Please try again.")
        return
    
    # Enhanced retrieval
    start_time = time.time()
    retrieved_chunks = enhance_retrieval(query, CONFIG["top_k"])
    retrieval_time = time.time() - start_time
    
    if not retrieved_chunks:
        print("❌ No relevant code found. Try a different query.")
        return
    
    # Display results
    display_retrieval_results(retrieved_chunks)
    print(f"⏱️ Retrieval completed in {retrieval_time:.2f} seconds")
    
    # Create enriched prompt
    final_prompt = create_enriched_prompt(query, retrieved_chunks)
    
    # Make API request
    response = make_api_request(final_prompt)
    
    # Display response
    print("\n" + "=" * 80)
    print("🧠 GENERATED CODE:")
    print("=" * 80)
    print(response)
    
    # Save interaction
    save_interaction(query, response, retrieved_chunks)
    print(f"\n💾 Interaction saved to interactions.jsonl")

if __name__ == "__main__":
    main()
