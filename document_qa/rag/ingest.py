"""Load, split, and index PDF text for document question answering."""

from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..config import settings
from .store import get_vector_store

def load_pdf_pages(pdf_path: str) -> list[Document]:
    """Return one document for each PDF page with extractable text.

    Each document contains the page text and metadata identifying its source
    and one-based page number. Pages without extractable text are skipped.

    Args:
        pdf_path: Path to the PDF file to read.

    Returns:
        Documents in the same order as their pages in the PDF.
    """
    reader = PdfReader(pdf_path)
    document_list = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_content = page.extract_text() or ""
        metadata = {"source": Path(pdf_path).name, "page": page_number}

        if not page_content.strip():
            continue

        document = Document(page_content=page_content,
                            metadata=metadata,
                            )
        document_list.append(document)

    return document_list

def chunking_document_pages(document_list: list[Document]) -> list[Document]:
    """Split page documents into overlapping chunks for retrieval.

    The splitter preserves each document's source and page metadata.

    Args:
        document_list: PDF page documents to split.

    Returns:
        Smaller documents in their original order.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    smaller_documents = splitter.split_documents(document_list)
    return smaller_documents

def index_pdf(pdf_path: str) -> list[str]:
    """Embed a PDF's text chunks and store them in Chroma.

    Pages without extractable text are skipped. If no chunks remain, the
    vector store is not opened and no embedding request is made.

    Args:
        pdf_path: Path to the PDF file to index.

    Returns:
        Chroma IDs for the stored chunks, or an empty list if there are none.
    """
    document_list = load_pdf_pages(pdf_path)
    document_chunks = chunking_document_pages(document_list)

    if not document_chunks:
        return []

    store = get_vector_store()
    ids = store.add_documents(document_chunks)

    return ids
