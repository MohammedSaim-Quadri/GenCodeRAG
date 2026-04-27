from unittest.mock import patch, MagicMock

from model.final_rag_system import search_qdrant


@patch("model.final_rag_system.get_qdrant_client")
@patch("model.final_rag_system.get_embedder")
def test_search_qdrant_success(mock_embedder, mock_qdrant):
    # Mock embedding output
    mock_embedder.return_value.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    # Mock Qdrant search result
    fake_result = [MagicMock()]
    mock_qdrant.return_value.search.return_value = fake_result

    result = search_qdrant(
        query="Create login API",
        language="python"
    )

    assert result == fake_result


@patch("model.final_rag_system.get_qdrant_client")
@patch("model.final_rag_system.get_embedder")
def test_search_qdrant_failure_returns_empty(mock_embedder, mock_qdrant):
    mock_embedder.return_value.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    mock_qdrant.return_value.search.side_effect = Exception("Qdrant failure")

    result = search_qdrant(
        query="Create login API",
        language="python"
    )

    assert result == []