from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from core.embeddings import EmbeddingService
from settings import settings
from logger import setup_logger
import json
from datetime import datetime

logger = setup_logger(__name__)

class RetrievalService:
    """
    Handles semantic search against Qdrant collection.
    Accepts client and embedder via constructor injection for testability.
    """

    def __init__(
            self,
            client: QdrantClient,
            embedder: EmbeddingService,
            collection_name: str = settings.COLLECTION_NAME,
            top_k: int = settings.TOP_K,
            score_threshold: float = settings.SCORE_THRESHOLD
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection_name = collection_name
        self._top_k = top_k
        self._score_threshold = score_threshold

    
    def search(
            self,
            query: str,
            language: str | None = None
    ) -> list[Any]:
        try:
            query_vector = self._embedder.encode(query)

            search_kwargs: dict[str, Any] = {
                "collection_name": self._collection_name,
                "query_vector": query_vector,
                "limit": self._top_k * 3,
                "score_threshold": self._score_threshold,
            }

            if language:
                search_kwargs["query_filter"] = Filter(
                    must=[
                        FieldCondition(
                            key="language",
                            match=MatchValue(value=language),
                        )
                    ]
                )

            results = self._client.search(**search_kwargs)
            return results[: self._top_k]
        
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
        
class InteractionLogger:
    """
    Appends query/response records to JSONL file.
    """
    def __init__(self, log_path: str = "interactions_qdrant.jsonl") -> None:
        self._log_path = log_path

    def log(
            self,
            query: str,
            language: str | None,
            response: str,
            chunk_ids: list[str],
    ) -> None:
        record ={
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "language": language,
            "response": response,
            "chunk_ids": chunk_ids,
        }

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")