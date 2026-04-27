import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
from datetime import datetime
from logger import setup_logger
from settings import settings

logger = setup_logger(__name__)

qdrant = None
embedder = None
hf_client = None

def get_qdrant_client():
    global qdrant
    if qdrant is None:
        logger.info("Initializing Qdrant client...")
        qdrant = QdrantClient(
            url = settings.QDRANT_HOST,
            api_key=settings.QDRANT_API_KEY
        )
    return qdrant

def get_embedder():
    global embedder

    if embedder is None:
        try:
            logger.info(
                f"Loading primary embedding model: "
                f"{settings.PRIMARY_EMBED_MODEL}"
            )

            embedder = SentenceTransformer(
                settings.PRIMARY_EMBED_MODEL
            )

        except Exception as e:
            logger.warning(
                f"Primary embedding model failed: {e}"
            )

            logger.info(
                f"Falling back to: "
                f"{settings.FALLBACK_EMBED_MODEL}"
            )

            embedder = SentenceTransformer(
                settings.FALLBACK_EMBED_MODEL
            )

    return embedder

def get_hf_client():
    global hf_client
    if hf_client is None:
        logger.info("HF token loaded successfully")
        hf_client = InferenceClient(
            provider="auto",
            api_key=settings.HF_TOKEN
        )
    
    return hf_client

def search_qdrant(query: str, language: str = None):
    try:
        embedder = get_embedder()
        qdrant = get_qdrant_client()

        query_vector = embedder.encode(query).tolist()

        search_kwargs = {
            "collection_name": settings.COLLECTION_NAME,
            "query_vector": query_vector,
            "limit": settings.TOP_K * 3,
            "score_threshold": settings.SCORE_THRESHOLD
        }

        if language:
            search_kwargs["query_filter"] = {
                "must": [
                    {
                        "key": "language",
                        "match": {
                            "value": language
                        }
                    }
                ]
            }

        results = qdrant.search(**search_kwargs)

        return results[:settings.TOP_K]

    except Exception as e:
        logger.error(f"Qdrant error: {e}")
        return []

def format_code_snippet(code: str, max_lines: int = 50) -> str:
    """Format and truncate code snippets for better readability."""
    lines = code.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ['...', '# Code truncated for brevity']
    return '\n'.join(lines)

def create_enriched_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Create a more sophisticated prompt with better context organization."""
    
    prompt_context = ""
    total_chars = 0
    max_chars = 12000
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

    safe_query = query.replace("{", "").replace("}", "")
    prompt = f"""You are an expert developer.

Use the following code snippets to generate a solution for the task.

# CODE CONTEXT:
{prompt_context}

# TASK:
{safe_query}

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
    logger.info(f"Found {len(chunks)} relevant code examples for the query.")
    print("-" * 80)
    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload
        repo = payload.get("repo", "unknown")
        path = payload.get("path", "unknown")
        func_name = payload.get("function_name", "unknown")
        language = payload.get("language", "text")
        score = getattr(chunk, "score", 1.0)
        print(f"{i}. [{language.upper()}] {repo}/{path}")
        logger.info(f"Retrieved chunk {i}: {repo}/{path} (Function: {func_name}, Score: {score:.2%})")
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
    """Query Hugging Face LLM using InferenceClient (Chat Completions)."""
    print(f"📝 Context length: {len(prompt)} characters")

    try:
        client = get_hf_client()
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            top_p = 0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"❌ HF API Error: {e}")
        return "⚠️ Failed to generate a response from the LLM."

def log_interaction(query: str, language: str, response: str, chunk_ids):
    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "language": language,
        "response": response,
        "chunks_used": chunk_ids
    }

    with open("interactions_qdrant.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# === Main Execution ===
def main():
    print("CodeGen (Qdrant-RAG) System")
    
    # Get user query
    query = input("💬 Enter your code generation prompt:\n> ")
    
    if not query.strip():
        print("❌ Empty query. Please try again.")
        logger.warning("User entered an empty query.")
        return
    
    language = infer_language_from_prompt(query)
    if language:
        print(f"🌐 Detected language from prompt: {language}")
        logger.info(f"Detected language from prompt: {language}")
    else:
        print("🌐 No specific language detected. Using all contexts.")
        logger.info("No specific language detected from prompt.")
    
    print("🔍 Searching Qdrant for relevant code chunks...")
    logger.info("Initiating search in Qdrant for relevant code chunks.")
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
    logger.info("Generated code response received from LLM.")
    print("=" * 80)
    print(response)
    
    # Save interaction
    chunk_ids = [r.payload.get("chunk_id") for r in results]
    log_interaction(query, language, response, chunk_ids)
    print("📦 Interaction logged successfully.")
    logger.info("Interaction logged successfully.")

if __name__ == "__main__":
    main()
