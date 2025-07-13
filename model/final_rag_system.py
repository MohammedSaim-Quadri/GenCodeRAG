#import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
#from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict, Any
import time
from datetime import datetime

# Load API token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
print(f"Token loaded: {'✓' if HF_TOKEN else '✗'}")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
print(f"Qdrant Token loaded: {'✓' if QDRANT_API_KEY else '✗'}")

COLLECTION_NAME = "code_chunks"
EMBED_MODEL = "all-MiniLM-L6-v2"

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# === Configuration ===
LLM_MODEL = "deepseek-ai/DeepSeek-V3-0324"
TOP_K = 5
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# === Init clients ===
qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
# === Load embedding model ===
print("🔤 Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("✓ Embedding model loaded")

# === Load FAISS index and metadata ===
# INDEX_PATH = "data/faiss/github_code.index"
# META_PATH = "data/faiss/github_code.meta.json"

# print("📦 Loading FAISS index + metadata...")
# try:
#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "r", encoding="utf-8") as f:
#         metadata = json.load(f)
#     print(f"✓ Loaded {index.ntotal} embeddings and {len(metadata)} metadata entries")
# except Exception as e:
#     print(f"❌ Error loading index/metadata: {e}")
#     exit(1)

def search_qdrant(query: str, language: str = None):
    query_vector = embedder.encode(query).tolist()

    filters = []
    if language:
        filters.append(FieldCondition(key="language", match=MatchValue(value=language)))

    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "query_vector": query_vector,
        "limit": TOP_K * 3,
        "with_payload": True,
        "score_threshold": 0.75
    }

    if filters:
        search_kwargs["query_filter"] = Filter(must=filters)

    return qdrant.search(**search_kwargs)


def format_code_snippet(code: str, max_lines: int = 50) -> str:
    """Format and truncate code snippets for better readability."""
    lines = code.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ['...', '# Code truncated for brevity']
    return '\n'.join(lines)

# def extract_language_from_path(path: str) -> str:
#     """Extract programming language from file path."""
#     extension_map = {
#         '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
#         '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
#         '.go': 'go', '.rs': 'rust', '.php': 'php', '.rb': 'ruby',
#         '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
#         '.sql': 'sql', '.html': 'html', '.css': 'css', '.sh': 'bash',
#         '.m': 'objective-c', '.jl': 'julia', '.lua': 'lua',
#         '.dart': 'dart', '.r': 'r', '.pl': 'perl', '.json': 'json'
# }
#     ext = Path(path).suffix.lower()
#     return extension_map.get(ext, 'text')

# def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Remove duplicate code chunks based on content similarity."""
#     seen_codes = set()
#     unique_chunks = []
    
#     for chunk in chunks:
#         # Create a simple hash of the code content
#         code_hash = hash(chunk['code'][:200])  # Use first 200 chars for comparison
#         if code_hash not in seen_codes:
#             seen_codes.add(code_hash)
#             unique_chunks.append(chunk)
    
#     return unique_chunks

# def enhance_retrieval(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#     """Enhanced retrieval with filtering and ranking."""
#     print(f"\n🔍 Searching for relevant code (top-{top_k})...")
    
#     # Encode query
#     query_embed = embed_model.encode([query])
    
#     # Search with higher k to allow filtering
#     search_k = min(top_k * 3, len(metadata))
#     D, I = index.search(query_embed, search_k)
    
#     # Filter and enrich results
#     enhanced_chunks = []
#     for idx, (distance, meta_idx) in enumerate(zip(D[0], I[0])):
#         if meta_idx >= len(metadata):
#             continue
            
#         meta = metadata[meta_idx]
#         similarity = 1 - distance  # Convert distance to similarity
        
#         # Filter by similarity threshold
#         if similarity < CONFIG["similarity_threshold"]:
#             continue
            
#         # Enhance metadata
#         enhanced_meta = {
#             **meta,
#             "similarity": round(similarity, 3),
#             "language": extract_language_from_path(meta.get("path", "")),
#             "rank": idx + 1,
#             "formatted_code": format_code_snippet(meta["code"])
#         }
        
#         enhanced_chunks.append(enhanced_meta)
    
#     # Remove duplicates and limit to top_k
#     unique_chunks = deduplicate_chunks(enhanced_chunks)[:top_k]
    
#     return unique_chunks

def create_enriched_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Create a more sophisticated prompt with better context organization."""
    
    prompt_context = ""
    total_chars = 0
    max_chars = 20000
    for point in chunks:
        payload = point.payload
        lang = payload.get("language", "text")
        repo = payload.get("repo", "unknown")
        path = payload.get("path", "")
        func = payload.get("function_name", "")
        code = format_code_snippet(payload.get("code", ""))

        snippet = f"""### [{lang.upper()}] {repo}/{path}
Function: {func}
```{lang}
{code}
```"""
        if total_chars + len(snippet) > max_chars:
            break
        prompt_context += snippet
        total_chars += len(snippet)
    #         # Check if adding this chunk would exceed context limit
    #         if total_length + len(chunk_context) > CONFIG["max_context_length"]:
    #             break
                
    #         lang_section += chunk_context
    #         total_length += len(chunk_context)
        
    #     context_sections.append(lang_section)
    
    # # Combine all sections
    # full_context = "\n".join(context_sections)
    
    # Create the final prompt
    prompt = f"""You are an expert developer.

Use the following code snippets to generate a solution for the task.

# CODE CONTEXT:
{prompt_context}

# TASK:
{query}

# INSTRUCTIONS:
- Analyze the provided code examples for patterns and best practices
- Generate clean, well-documented, and efficient code
- Include appropriate error handling and edge cases
- Add helpful comments explaining the logic
- Generate high-quality, production-level code

# RESPONSE:
"""
    
    return prompt

def display_retrieval_results(chunks: List[Dict[str, Any]]):
    """Display enhanced retrieval results."""
    print(f"\\n📊 Found {len(chunks)} relevant code examples:")
    print("-" * 80)
    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload
        repo = payload.get("repo", "unknown")
        path = payload.get("path", "unknown")
        func_name = payload.get("function_name", "unknown")
        language = payload.get("language", "text")
        score = getattr(chunk, "score", 1.0)
        print(f"{i}. [{language.upper()}] {repo}/{path}")
        print(f"   Function: {func_name}")
        print(f"   Similarity: {score:.1%}")
        print(f"   Code Length: {len(payload['code'])} chars")
        print("-" * 40)

def infer_language_from_prompt(prompt: str) -> str:
    keyword_map = {
        "python": ["flask", "django", "pandas", "numpy", "python", "py"],
        "java": ["spring", "jdk", "java", "jvm"],
        "javascript": ["node", "express", "js", "javascript", "react"],
        "typescript": ["ts", "typescript", "nestjs"],
        "cpp": ["c++", "cpp", "stl"],
        "c": ["c ", "c-language", "c code"],
        "csharp": ["c#", "dotnet", "csharp"],
        "go": ["golang", "go "],
        "rust": ["rust"],
        "php": ["php", "laravel"],
        "ruby": ["ruby", "rails"],
        "bash": ["shell", "bash", "sh"],
        "kotlin": ["kotlin", "android"],
        "swift": ["swift", "ios"],
        "scala": ["scala"],
        "sql": ["sql", "database", "postgres", "mysql"],
    }

    prompt_lower = prompt.lower()
    for lang, keywords in keyword_map.items():
        for word in keywords:
            if word in prompt_lower:
                return lang
    return None 

def query_hf_llm(prompt: str) -> str:
    """Make API request with enhanced error handling and retry logic."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    
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

def log_interaction(query: str, language: str, response: str, chunk_ids):
    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "language": language,
        "response": response,
        "chunks_used": chunk_ids
    }

    with open("interactions_qdrant.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\\n")

# === Main Execution ===
def main():
    print("CodeGen (Qdrant-RAG) System")
    
    # Get user query
    query = input("💬 Enter your code generation prompt:\n> ")
    
    if not query.strip():
        print("❌ Empty query. Please try again.")
        return
    
    language = infer_language_from_prompt(query)
    if language:
        print(f"🌐 Detected language from prompt: {language}")
    else:
        print("🌐 No specific language detected. Using all contexts.")
    
    print("🔍 Searching Qdrant for relevant code chunks...")
    results = search_qdrant(query, language)
    
    # Display results
    display_retrieval_results(results)
    
    # Create enriched prompt
    final_prompt = create_enriched_prompt(query, results)
    
    # Make API request
    response = query_hf_llm(final_prompt)
    
    # Display response
    print("\n" + "=" * 80)
    print("🧠 GENERATED CODE:")
    print("=" * 80)
    print(response)
    
    # Save interaction
    chunk_ids = [r.payload.get("chunk_id") for r in results]
    log_interaction(query, language, response, chunk_ids)
    print("📦 Interaction logged successfully.")

if __name__ == "__main__":
    main()
