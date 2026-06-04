from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # GROQ
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-70b-8192"

    # Redis & Database
    REDIS_URL: str = "redis://localhost:6379"
    DATABASE_URL: str = "sqlite:///./data/db/documind.db"

    # Models (fastembed / ONNX model names)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval
    MAX_CHUNKS_RETRIEVED: int = 20

    # Paths
    UPLOAD_DIR: str = "data/uploads"
    INDEX_DIR: str = "data/indexes"
    DB_DIR: str = "data/db"

    # OCR (optional — only needed for scanned/image-based PDFs)
    TESSERACT_CMD: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
