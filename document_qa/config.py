"""Central settings for document chunking, storage, and model selection."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Hold configuration values shared by the RAG components.

    Pydantic settings can load matching values from environment variables,
    allowing local configuration without changing application code.
    """
    chunk_size: int = 800
    chunk_overlap: int = 100
    collection_name: str = "documents"
    chroma_path: str = ".chroma"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-5-nano"


settings = Settings()
