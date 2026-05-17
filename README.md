# GenCodeRAG — AI Code Generation using GitHub + RAG

## Project Overview

GenCodeRAG is an AI-powered code generation system that uses GitHub repositories as contextual knowledge and Retrieval-Augmented Generation (RAG) to generate high-quality, production-ready code.

Instead of generating code only from a user prompt, the system first retrieves relevant real-world code examples from GitHub repositories, stores them in a vector database using Qdrant, and then enriches the final prompt before sending it to a Large Language Model (LLM).

This improves:

- Code quality
- Code relevance
- Real-world implementation patterns
- Production readiness
- Developer productivity

The project is designed for research, learning, and scalable future production deployment.

---

# Architecture Overview

The system follows this pipeline:

```text
GitHub Scraper
↓
Code Cleaning + Function Extraction
↓
Embedding Generation
↓
Qdrant Vector Database
↓
Retrieval + Prompt Enrichment
↓
LLM Code Generation
↓
Streamlit UI
```

---

# Flow Explanation

## 1. GitHub Scraper

Fetches top-starred repositories from GitHub using the GitHub API and downloads useful code files across multiple programming languages.

---

## 2. Code Cleaning + Function Extraction

Processes raw code files and extracts reusable functions and logic blocks using:

- AST parsing (Python)
- Tree-Sitter parsing (JavaScript)
- Heuristic extraction for other languages

---

## 3. Embedding Generation

Converts extracted code chunks into vector embeddings using Sentence Transformers.

---

## 4. Qdrant Vector Database

Stores vector embeddings for fast semantic similarity search.

---

## 5. Retrieval + Prompt Enrichment

When a user enters a prompt, relevant code chunks are retrieved from Qdrant and injected into the final LLM prompt.

---

## 6. LLM Code Generation

Uses Hugging Face Inference API with DeepSeek-V3 to generate high-quality code using retrieved context.

---

## 7. Streamlit UI

Provides a simple interactive frontend for users to generate code.

---

# Project Structure

```text
GenCodeRAG/
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
│
├── model/
│   ├── __init__.py
│   └── final_rag_system.py
│
├── pipeline/
│   ├── __init__.py
│   ├── github_scraper.py
│   ├── clean_code.py
│   ├── parser_utils.py
│   └── prepare_dataset.py
│
├── qdrant/
│   ├── __init__.py
│   └── embed_chunks_qdrant.py
│
├── legacy/
│   ├── __init__.py
│   └── embed_chunks_faiss.py
│
├── tests/
│   ├── test_clean_code.py
│   ├── test_language_detection.py
│   ├── test_parser_utils.py
│   └── test_qdrant_search.py
│
├── data/
├── datasets/
│
├── config.py
├── settings.py
├── logger.py
│
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .env.example
├── README.md
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

# Folder Description

## `app/`

Frontend UI built using Streamlit.

---

## `model/`

Core RAG system including:

- Retrieval
- Prompt creation
- LLM querying
- Interaction logging
- Embedding search

---

## `pipeline/`

Data pipeline scripts for:

- GitHub scraping
- Code cleaning
- Function extraction
- Dataset preparation

---

## `qdrant/`

Embedding generation and upload to the Qdrant vector database.

---

## `legacy/`

Archived FAISS-based vector search implementation (replaced by Qdrant).

---

## `tests/`

Unit and integration test suite using `pytest`.

---

# Features

- Multi-language GitHub scraping
- Python AST function extraction
- JavaScript Tree-Sitter parsing
- Multi-language heuristic extraction
- Sentence Transformer embeddings
- Qdrant vector retrieval
- Hugging Face LLM integration
- Streamlit frontend
- Context-aware code generation
- Structured JSON logging
- Prompt injection defense
- Docker support
- GitHub Actions CI/CD
- Unit and integration testing
- Production-style package structure

---

# Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd GenCodeRAG
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Runtime Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

---

## Install Project in Editable Mode

```bash
pip install -e .
```

---

# Environment Variables

Create a `.env` file in the project root.

---

## Required Variables

```env
HF_TOKEN=
QDRANT_HOST=
QDRANT_API_KEY=
GITHUB_TOKEN=
PRIMARY_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code
FALLBACK_EMBED_MODEL=microsoft/codebert-base
COLLECTION_NAME=code_chunks
```

---

# Environment Variable Explanation

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face API token for LLM inference |
| `QDRANT_HOST` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant API key (if cloud-hosted) |
| `GITHUB_TOKEN` | GitHub Personal Access Token |
| `PRIMARY_EMBED_MODEL` | Main embedding model |
| `FALLBACK_EMBED_MODEL` | Backup embedding model |
| `COLLECTION_NAME` | Qdrant collection name |

---

# Example `.env.example`

```env
HF_TOKEN=your_huggingface_token
QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
GITHUB_TOKEN=your_github_token

PRIMARY_EMBED_MODEL=jinaai/jina-embeddings-v2-base-code
FALLBACK_EMBED_MODEL=microsoft/codebert-base

COLLECTION_NAME=code_chunks
```

---

# Execution Order (Very Important)

The pipeline must be executed in this order.

---

## Step 1 — Scrape GitHub Repositories

```bash
python pipeline/github_scraper.py
```

---

## Step 2 — Clean Code + Extract Functions

```bash
python pipeline/clean_code.py
```

---

## Step 3 — Upload Embeddings to Qdrant

```bash
python qdrant/embed_chunks_qdrant.py
```

---

## Step 4 — Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# Running Tests

```bash
pytest
```

---

# Docker Usage

## Build Docker Image

```bash
docker build -t gencoderag .
```

---

## Run Docker Container

```bash
docker run -p 8501:8501 gencoderag
```

---

# Example Usage

## User Prompt

```text
Create a Python function to hash passwords using bcrypt
```

---

## What Happens

1. System detects programming language
2. Searches Qdrant for similar code examples
3. Retrieves relevant code chunks
4. Builds enriched LLM prompt
5. Sends prompt to Hugging Face LLM
6. Returns production-ready code

---

# Testing

The project includes:

- Unit tests
- Integration tests
- Mocked Qdrant tests
- Parser tests
- CI/CD validation via GitHub Actions

---

# Production Improvements Implemented

- Lazy-loaded model initialization
- Structured JSON logging
- Deterministic Qdrant IDs
- Prompt injection defense
- Request debouncing
- Docker containerization
- GitHub Actions CI/CD
- Centralized configuration management
- Shared settings management
- Tree-Sitter parsing support
- Retry-safe GitHub scraping
- Safer API error handling

---

# Future Improvements

Planned upgrades:

- Expand Tree-Sitter support to all languages
- FastAPI backend support
- Cross-encoder reranking
- Metrics dashboard
- Cloud deployment
- Advanced caching
- Retrieval analytics
- Multi-model routing

---

# Technologies Used

- Python
- Streamlit
- Qdrant
- Sentence Transformers
- Hugging Face Inference API
- DeepSeek-V3
- Tree-Sitter
- Pytest
- Docker
- GitHub Actions

---

# Final Note

This project demonstrates how Retrieval-Augmented Generation (RAG) can significantly improve code generation quality by grounding LLM outputs in real-world engineering patterns from GitHub repositories.

It combines:

- Machine Learning
- Vector Databases
- LLM Systems
- Software Engineering
- Production Pipeline Design

into one practical AI engineering project.
