from unittest.mock import MagicMock

from core.retrieval import RetrievalService
from core.embeddings import EmbeddingService
from qdrant_client import QdrantClient


def test_search_qdrant_success():
    mock_embedder = MagicMock(spec=EmbeddingService)
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3]

    mock_client = MagicMock(spec=QdrantClient)
    mock_client.search.return_value = [MagicMock()]

    service = RetrievalService(client=mock_client, embedder=mock_embedder)
    result = service.search("Create login API", language="python")

    assert len(result) == 1
    mock_embedder.encode.assert_called_once_with("Create login API")
    mock_client.search.assert_called_once()


def test_search_qdrant_failure_returns_empty():
    mock_embedder = MagicMock(spec=EmbeddingService)
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3]

    mock_client = MagicMock(spec=QdrantClient)
    mock_client.search.side_effect = Exception("Qdrant failure")

    service = RetrievalService(client=mock_client, embedder=mock_embedder)
    result = service.search("Create login API", language="python")

    assert result == []