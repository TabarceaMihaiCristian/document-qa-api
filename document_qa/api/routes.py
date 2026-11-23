"""HTTP endpoints for PDF uploads and document-based questions."""

import shutil
from typing import Annotated
from tempfile import TemporaryDirectory
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from ..schemas import QuestionRequest, QuestionResponse, SourceReference, UploadResponse
from ..rag.answer import answer_question
from ..rag.ingest import index_pdf

router = APIRouter()

@router.post("/ask")
def ask_question(request: QuestionRequest) -> QuestionResponse:
    """Answer a question and return distinct source-page references.

    Args:
        request: Validated question submitted to the API.

    Returns:
        The answer and unique pages from the retrieved document chunks.
    """
    answer, documents = answer_question(request.question)

    sources = []
    seen = set()
    for document in documents:
        source = document.metadata["source"]
        page = document.metadata["page"]
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceReference(source=source, page=page))

    return QuestionResponse(answer=answer, sources=sources)

@router.post("/documents")
def upload_documents(file: Annotated[UploadFile, File()]) -> UploadResponse:
    """Index an uploaded PDF and return the IDs of its stored chunks.

    The upload is copied to a temporary file so the PDF reader can reopen it.
    The temporary directory is removed after indexing or if indexing fails.

    Args:
        file: PDF uploaded through the request form.

    Returns:
        IDs of the chunks added to the vector store.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    filename = Path(file.filename or "upload.pdf").name

    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / filename

        with pdf_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        ids = index_pdf(str(pdf_path))
        return UploadResponse(chunk_ids=ids)
