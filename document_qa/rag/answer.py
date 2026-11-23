"""Retrieve document evidence and generate answers from it."""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ..config import settings
from .store import get_vector_store

def retrieve_chunks(question: str, k: int = 4) -> list[Document]:
    """Find up to ``k`` stored chunks most similar to a question.

    The vector store embeds the question and searches its persisted Chroma
    collection for matching document chunks.

    Args:
        question: Question used for similarity search.
        k: Maximum number of chunks to retrieve.

    Returns:
        Matching chunks with their source and page metadata.
    """

    store = get_vector_store()
    results = store.similarity_search(question, k=k)

    return results

def answer_question(question: str) -> tuple[str, list[Document]]:
    """Answer a question using retrieved document chunks as context.

    The model is instructed to use only the retrieved excerpts and to say
    when they do not contain the answer. If retrieval finds no chunks, no
    chat-model request is made.

    Args:
        question: Question to answer from the indexed documents.

    Returns:
        The answer text and the chunks supplied to the model as evidence.
    """
    chunks = retrieve_chunks(question)
    if not chunks:
        return "No relevant document text found", []
    context = "\n\n".join(
        f"Source: {chunk.metadata['source']}, page {chunk.metadata['page']}\n"
        f"{chunk.page_content}"
        for chunk in chunks
    )
    model = ChatOpenAI(model=settings.chat_model)
    response = model.invoke([
        ("system", "Answer only from the supplied excerpts. If the answer is absent, say so."),
        ("human", f"Excerpts:\n{context}\n\nQuestion: {question}"),
    ])
    return (response.text, chunks)
