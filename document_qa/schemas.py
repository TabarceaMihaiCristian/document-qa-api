"""Request and response models for the document Q&A API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

class QuestionRequest(BaseModel):
    """Question submitted to the API for document-based answering."""
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class SourceReference(BaseModel):
    """Filename and one-based page number supporting an answer."""
    source: str
    page: Annotated[int, Field(ge=1)]

class QuestionResponse(BaseModel):
    """Generated answer and references to the retrieved source pages."""
    answer: str
    sources: list[SourceReference]

class UploadResponse(BaseModel):
    """IDs of document chunks stored after a PDF upload."""
    chunk_ids: list[str]
