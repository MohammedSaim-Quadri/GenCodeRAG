# GenCodeRAG — AI Code Generation using GitHub + RAG

## Project Overview

GenCodeRAG is an AI-powered code generation system that uses GitHub repositories as contextual knowledge and Retrieval-Augmented Generation (RAG) to generate high-quality, production-ready code.

Instead of generating code only from a user prompt, the system first retrieves relevant real-world code examples from GitHub repositories, stores them in a vector database using Qdrant, and then enriches the final prompt before sending it to a Large Language Model (LLM).

This improves:

- code quality
- code relevance
- real-world implementation patterns
- production-readiness
- developer productivity

The project is designed for research, learning, and scalable future production deployment.

---

## Architecture Overview

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

Flow Explanation
1. GitHub Scraper

Fetches top-starred repositories from GitHub using GitHub API and downloads useful code files across multiple programming languages.

2. Code Cleaning + Function Extraction

Processes raw code files and extracts reusable functions and logic blocks using AST (for Python) and heuristic extraction (for other languages).

3. Embedding Generation

Converts extracted code chunks into vector embeddings using Sentence Transformers.

4. Qdrant Vector Database

Stores vector embeddings for fast semantic similarity search.

5. Retrieval + Prompt Enrichment

When a user enters a prompt, relevant code chunks are retrieved from Qdrant and injected into the final LLM prompt.

6. LLM Code Generation

Uses Hugging Face Inference API with DeepSeek-V3 to generate high-quality code using retrieved context.

7. Streamlit UI

Provides a simple interactive frontend for users to generate code.

Project Structure
GenCodeRAG/
│
├── app/
│   └── streamlit_app.py
│
├── model/
│   └── final_rag_system.py
│
├── pipeline/
│   ├── github_scraper.py
│   ├── clean_code.py
│   └── prepare_dataset.py
│
├── qdrant/
│   └── embed_chunks_qdrant.py
│
├── legacy/
│   └── faiss/
|       └── embed_chunks_faiss.py
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── README.md
│
└── data/
Folder Description
app/

Frontend UI built using Streamlit.

model/

Core RAG system including:

retrieval
prompt creation
LLM querying
interaction logging
pipeline/

Data pipeline scripts for:

GitHub scraping
code cleaning
dataset preparation
qdrant/

Embedding generation and upload to Qdrant vector database.

faiss/

Archived legacy FAISS-based vector search implementation
(replaced by Qdrant)

Installation
Clone Repository
git clone <your-repository-url>
cd GenCodeRAG
Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python -m venv venv
source venv/bin/activate
Install Runtime Dependencies
pip install -r requirements.txt
Install Development Dependencies
pip install -r requirements-dev.txt
Environment Variables

Create a .env file in the root directory.

Required variables:

HF_TOKEN=
QDRANT_HOST=
QDRANT_API_KEY=
GITHUB_TOKEN=
Variable Explanation
HF_TOKEN

Hugging Face API token for LLM inference.

QDRANT_HOST

Qdrant server URL.

Example:

http://localhost:6333
QDRANT_API_KEY

Qdrant API key (if cloud-hosted).

GITHUB_TOKEN

GitHub Personal Access Token for scraping repositories.

.env.example

Create a file named .env.example

HF_TOKEN=your_huggingface_token
QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
GITHUB_TOKEN=your_github_token
Execution Order (Very Important)

The pipeline must be executed in this order:

Step 1 — Scrape GitHub Repositories
python pipeline/github_scraper.py
Step 2 — Clean Code + Extract Functions
python pipeline/clean_code.py
Step 3 — Upload Embeddings to Qdrant
python qdrant/embed_chunks_qdrant.py
Step 4 — Launch Streamlit App
streamlit run app/streamlit_app.py
Example Usage
User Prompt
Create a Python function to hash passwords using bcrypt
What Happens
System detects programming language
Searches Qdrant for similar code examples
Retrieves relevant code chunks
Builds enriched LLM prompt
Sends prompt to Hugging Face LLM
Returns production-ready code
Current Features
Multi-language GitHub scraping
Python AST function extraction
Multi-language heuristic extraction
Sentence Transformer embeddings
Qdrant vector retrieval
Hugging Face LLM integration
Streamlit frontend
Context-aware code generation
Interaction logging
Future Improvements

Planned upgrades:

Replace regex extraction with Tree-Sitter parser
Switch to code-specialized embedding models
Add unit tests and integration tests
Docker containerization
GitHub Actions CI/CD
Production-grade logging
Prompt injection defense
Better caching and rate limiting
Full package structure using pyproject.toml

Final Note

This project demonstrates how Retrieval-Augmented Generation (RAG) can significantly improve code generation quality by grounding LLM outputs in real-world engineering patterns from GitHub repositories.

It combines:

Machine Learning
Vector Databases
LLM Systems
Software Engineering
Production Pipeline Design

into one practical AI engineering project.