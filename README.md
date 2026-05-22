````markdown
# GenCodeRAG — AI Code Generation using GitHub + RAG

### Evolving toward an Autonomous Code Intelligence System

---

# Project Overview

GenCodeRAG is an AI-powered code generation and repository intelligence system that uses GitHub repositories as contextual knowledge and Retrieval-Augmented Generation (RAG) to generate high-quality, production-ready code.

Instead of generating code only from a user prompt, the system first retrieves relevant real-world code examples from GitHub repositories, stores them in a vector database using Qdrant, and enriches the final prompt before sending it to a Large Language Model (LLM).

This significantly improves:

- Code quality
- Context relevance
- Real-world engineering patterns
- Production-readiness
- Developer productivity

The project is evolving from a traditional RAG-based code generation platform into a repository-aware autonomous AI engineering system capable of:

- Repository understanding
- Structural code analysis
- Automated refactoring
- Documentation generation
- Multi-agent orchestration
- Pull request automation

---

# Architecture Overview

The current system follows this pipeline:

```text
GitHub Scraper
        ↓
Code Cleaning + Function Extraction
(Tree-sitter + AST Parsing)
        ↓
Embedding Generation
(core/embeddings.py)
        ↓
Qdrant Vector Database
        ↓
Retrieval + Prompt Enrichment
(core/retrieval.py + core/prompts.py)
        ↓
LLM Completion
(core/llm.py — HuggingFace or Ollama)
        ↓
Streamlit UI
```

---

# Why Retrieval-Augmented Generation (RAG)?

Even as the system evolves into an autonomous agent architecture, Retrieval-Augmented Generation remains a core component of the platform.

Instead of relying entirely on LLM memory, agents retrieve relevant real-world code patterns and repository context before making decisions.

RAG is used for:

- Grounding code generation in real-world implementations
- Cross-file repository understanding
- Consistent refactoring patterns
- Documentation style matching
- Repository-aware reasoning
- Reducing hallucinations during autonomous modifications

The long-term goal is not to replace RAG with agents — but to build retrieval-grounded autonomous systems.

---

# Flow Explanation

## 1. GitHub Scraper

Fetches top-starred repositories from GitHub using the GitHub API and downloads useful code files across multiple programming languages.

---

## 2. Code Cleaning + Function Extraction

Processes raw code files and extracts reusable functions and logic blocks using:

- Python AST parsing
- Tree-sitter parsing
- Heuristic extraction for additional languages

---

## 3. Embedding Generation

Converts extracted code chunks into vector embeddings using Sentence Transformers.

---

## 4. Qdrant Vector Database

Stores vector embeddings for fast semantic similarity search and metadata filtering.

---

## 5. Retrieval + Prompt Enrichment

When a user enters a prompt, relevant code chunks are retrieved from Qdrant and injected into the final LLM prompt.

---

## 6. LLM Code Generation

Uses Hugging Face Inference API or local Ollama models to generate high-quality code using retrieved contextual knowledge.

---

## 7. Streamlit UI

Provides an interactive frontend for generating code and querying indexed repositories.

---

# Architecture Versions

---

## v1.0.0 — RAG Code Generation

### Current Stable Version

Single-pipeline system:

```text
GitHub Scraper
→ AST Extraction
→ Qdrant Embeddings
→ HuggingFace LLM
→ Streamlit UI
```

Features:
- GitHub scraping
- AST extraction
- Qdrant embeddings
- HuggingFace inference
- Streamlit UI

---

## v1.1.0 — Core Service Refactor

### Current Development Version

Internal refactor introducing a clean service layer under `core/`.

Zero change to user-facing functionality while preparing the architecture for repository-level reasoning and autonomous workflows.

### Added Improvements

- LiteLLM abstraction layer
- Local Ollama support
- Dependency injection patterns
- Improved modularity
- Cleaner service boundaries

### Core Modules

- `core/embeddings.py`
  - `EmbeddingService`
  - Lazy-loaded SentenceTransformer support

- `core/retrieval.py`
  - `RetrievalService`
  - Interaction logging
  - Metadata filtering

- `core/llm.py`
  - Multi-provider LLM abstraction
  - HuggingFace + Ollama support

- `core/prompts.py`
  - Prompt construction
  - Prompt enrichment
  - Language detection

- `model/final_rag_system.py`
  - Preserved as a backwards-compatibility shim

---

## v2.0.0 — Repository Understanding

### In Development

The system evolves from function-level retrieval into repository-level understanding.

### Planned Features

- Full repository cloning
- Structural repository traversal
- Dependency graph generation
- Per-repository Qdrant collections
- Repository-aware semantic retrieval
- Complexity scoring
- FastAPI backend
- Extended Tree-sitter support
  - Python
  - TypeScript
  - Java
  - Go
  - Rust

### Example Future Usage

```bash
python -m gencoderag analyze \
  --repo https://github.com/tiangolo/fastapi \
  --question "How does dependency injection work?"
```

---

## v3.0.0 — Autonomous Agent System

### Planned

The long-term vision is a retrieval-grounded autonomous AI engineering system.

### Planned Agent System

- Reviewer Agent
- Refactor Agent
- Documentation Agent
- Test Agent
- GitHub PR Agent

### Planned Features

- MCP tool abstraction layer
- LangGraph orchestration
- Repository sandboxing
- Human approval checkpoints
- Automated PR generation
- Repository-level memory
- Sequential and graph-based workflows

---

# Planned Multi-Agent Workflow

```text
Repository Clone
        ↓
Repository Analysis
        ↓
Reviewer Agent
        ↓
Human Approval Checkpoint
        ↓
Refactor Agent
        ↓
Patch Generation
        ↓
Human Approval Checkpoint
        ↓
Test Agent
        ↓
Documentation Agent
        ↓
GitHub PR Agent
        ↓
Automated Pull Request
```

---

# Safety Constraints

The autonomous workflow is intentionally designed with safety boundaries.

### Planned Safety Features

- Agents cannot directly modify repositories without approval
- Refactors are generated as diffs/patches first
- Tests must pass before PR generation
- Filesystem access is sandboxed through MCP
- Shell execution is allowlisted
- Repository workspaces are isolated
- Human review checkpoints exist before critical actions

The goal is assisted autonomous engineering — not uncontrolled repository modification.

---

# Why MCP?

Model Context Protocol (MCP) is used as a tool abstraction layer between agents and external systems.

Instead of agents directly interacting with:
- filesystems
- shell commands
- GitHub APIs

they communicate through controlled MCP interfaces.

### Benefits

- Safer filesystem access
- Sandboxed shell execution
- Better testability
- Cleaner architecture boundaries
- Easier mocking during testing
- Future remote execution support
- Reduced coupling between agents and infrastructure

### Planned MCP Modules

```text
mcp/
├── filesystem.py
├── shell.py
└── github_mcp.py
```

---

# Project Structure

```text
GenCodeRAG/
│
├── core/                          # V2 service architecture
│   ├── embeddings.py              # EmbeddingService
│   ├── retrieval.py               # RetrievalService
│   ├── llm.py                     # LLM abstraction layer
│   └── prompts.py                 # Prompt generation logic
│
├── app/
│   └── streamlit_app.py           # Streamlit frontend
│
├── pipeline/
│   ├── github_scraper.py
│   ├── clean_code.py
│   ├── parser_utils.py
│   ├── prepare_dataset.py
│   ├── repo_cloner.py             # V2
│   └── repo_analyzer.py           # V2
│
├── qdrant/
│   ├── embed_chunks_qdrant.py
│   └── collections.py             # V2
│
├── agents/                        # V3
│   ├── reviewer.py
│   ├── refactor.py
│   ├── docs.py
│   ├── testing.py
│   └── pr_agent.py
│
├── graph/                         # V3
│   ├── workflow.py
│   └── state.py
│
├── mcp/                           # V3
│   ├── filesystem.py
│   ├── shell.py
│   └── github_mcp.py
│
├── api/
│   ├── main.py
│   └── routes/
│
├── model/
│   └── final_rag_system.py        # Legacy V1 compatibility
│
├── legacy/
│   └── faiss/
│
├── tests/
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.agent
│   └── docker-compose.yml
│
├── data/
│
├── config.py
├── settings.py
├── logger.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── pytest.ini
```

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

# Environment Variables

Create a `.env` file in the root directory.

---

## Required Variables

```env
# GitHub
GITHUB_TOKEN=your_github_token

# Qdrant
QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# HuggingFace
HF_TOKEN=your_huggingface_token

# LLM Provider
LLM_PROVIDER=huggingface

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b-instruct
```

---

# Example `.env.example`

```env
GITHUB_TOKEN=your_github_token

QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

HF_TOKEN=your_huggingface_token

LLM_PROVIDER=huggingface

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b-instruct
```

---

# Execution Order

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

# Switching to Local LLM (Ollama)

To run fully offline:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b-instruct
```

Ensure Ollama is running:

```bash
ollama pull qwen2.5-coder:7b-instruct
ollama serve
```

---

# Example Usage

## User Prompt

```text
Create a Python function to hash passwords using bcrypt
```

---

## Internal Workflow

1. Detect programming language
2. Retrieve semantically relevant code from Qdrant
3. Build enriched prompt
4. Query configured LLM
5. Return grounded production-ready code

---

# Current Features

- Multi-language GitHub scraping
- Python AST parsing
- Tree-sitter parsing support
- Multi-language code extraction
- Sentence Transformer embeddings
- Qdrant semantic retrieval
- Metadata filtering
- LiteLLM integration
- HuggingFace support
- Ollama support
- Streamlit frontend
- Prompt injection defense
- Structured JSON logging
- Dockerized deployment
- GitHub Actions CI
- Unit testing

---

# Tech Stack

## Backend
- Python
- FastAPI
- Streamlit

## AI / ML
- HuggingFace
- Ollama
- LiteLLM
- SentenceTransformers

## Vector Database
- Qdrant

## Parsing
- Tree-sitter
- Python AST

## Infrastructure
- Docker
- GitHub Actions
- Pytest

---

# Why Qdrant?

Qdrant provides:

- Fast vector similarity search
- Metadata filtering
- Repository-level indexing
- Scalable deployment
- Production-grade API support

The system uses:
- global knowledge collections
- repository-scoped working collections

to support retrieval-grounded repository reasoning.

---

# Running Tests

```bash
pytest
```

---

# Long-Term Vision

The long-term goal is to evolve GenCodeRAG into a retrieval-grounded autonomous AI software engineering platform capable of:

- Understanding large repositories
- Detecting technical debt
- Refactoring legacy systems
- Generating documentation
- Running tests
- Opening pull requests automatically

while remaining grounded through Retrieval-Augmented Generation.

---

# Final Note

This project demonstrates how Retrieval-Augmented Generation (RAG), vector databases, repository analysis, and modern LLM systems can be combined into a scalable AI engineering platform.

It combines:

- Machine Learning
- Vector Databases
- LLM Systems
- Software Engineering
- Repository Analysis
- Autonomous Agents
- Production Infrastructure

into one practical end-to-end AI engineering system.
````
