from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    QDRANT_HOST: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    COLLECTION_NAME: str = "code_chunks"

    # Hugging Face
    HF_TOKEN: str | None = None
    LLM_MODEL: str = "deepseek-ai/DeepSeek-V3-0324"

    # GitHub
    GITHUB_TOKEN: str | None = None

    # Embeddings
    PRIMARY_EMBED_MODEL: str = "jinaai/jina-embeddings-v2-base-code"
    FALLBACK_EMBED_MODEL: str = "microsoft/codebert-base"
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.65

    # Generation
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 1024

    # Scraper
    MAX_REPOS: int = 30
    PER_PAGE: int = 30
    STARS: int = 500
    MAX_FILES_PER_REPO: int = 20

    # LLM Provider
    LLM_PROVIDER: str = "huggingface"   # options: "huggingface", "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder:7b-instruct"

    REPO_WORKSPACE_DIR: str = "workspace"
    MAX_FILE_SIZE_BYTES: int = 100_000 

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()