"""Create and access the local vector store used for document retrieval."""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from ..config import settings


def get_vector_store() -> Chroma:
    """Open or create the persistent Chroma collection for document chunks.

    The collection uses the configured embedding model and storage path.

    Returns:
        The configured Chroma vector store.
    """
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path
    )

    return store
