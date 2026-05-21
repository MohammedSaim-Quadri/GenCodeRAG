"""
V1 compatibility shim.

This module re-exports symbols from core/ to maintain backwards
compatibility with code that imports from model.final_rag_system.

Do not add new functionality here. New code should import from core/ directly.
"""
from qdrant_client import QdrantClient
from core.embeddings import EmbeddingService
from core.retrieval import RetrievalService, InteractionLogger
from core.llm import LLMService
from core.prompts import (
    infer_language_from_prompt,
    create_enriched_prompt,
    format_code_snippet,
)
from settings import settings

_embedder = None
_client = None
_retrieval = None
_llm = None
_logger = InteractionLogger()


def _get_retrieval() -> RetrievalService:
    global _embedder, _client, _retrieval
    if _retrieval is None:
        _embedder = EmbeddingService()
        _client = QdrantClient(url=settings.QDRANT_HOST, api_key=settings.QDRANT_API_KEY)
        _retrieval = RetrievalService(client=_client, embedder=_embedder)
    return _retrieval


def _get_llm() -> LLMService:
    global _llm
    if _llm is None:
        _llm = LLMService()
    return _llm


def search_qdrant(query: str, language: str | None = None):
    return _get_retrieval().search(query, language)


def query_hf_llm(prompt: str) -> str:
    return _get_llm().complete(prompt)


def log_interaction(query, language, response, chunk_ids):
    _logger.log(query, language, response, chunk_ids)


# Legacy — kept for any code that referenced these directly.
# New code should not use these.
def get_qdrant_client():
    return _client


def get_embedder():
    return _embedder.model


__all__ = [
    "search_qdrant",
    "query_hf_llm",
    "log_interaction",
    "infer_language_from_prompt",
    "create_enriched_prompt",
    "format_code_snippet",
    "get_qdrant_client",
    "get_embedder",
]