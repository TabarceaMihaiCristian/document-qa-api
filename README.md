# Document Q&A API

A learning POC for asking questions about an uploaded PDF. It uses FastAPI for the HTTP API, LangChain to connect the document and model steps, Chroma as a local vector database, and the OpenAI API for text embeddings and answer generation.

I built this project to get hands-on experience with LangChain and vector search. My earlier backend work involved document ingestion and RAG workflows; this POC lets me work directly with the parsing, chunking, embedding, retrieval, and answering steps.

## How it works

1. `POST /documents` receives a PDF and copies it to a temporary file. The temporary file is removed after indexing.
2. PyPDF extracts text page by page. Pages without extractable text are skipped.
3. LangChain's `RecursiveCharacterTextSplitter` divides each page into chunks of up to 800 characters, with 100 characters of overlap. The overlap keeps nearby sentences together when they fall around a split. Each chunk keeps the original filename and page number as metadata.
4. `OpenAIEmbeddings` converts each chunk into a vector. Chroma stores the vectors, chunk text, and metadata in the local `.chroma` directory.
5. `POST /ask` embeds a question and retrieves up to four similar chunks from Chroma. The question and retrieved excerpts are sent to `ChatOpenAI` with an instruction to answer from those excerpts. The API returns the answer and distinct source-page references.

This is a basic retrieval-augmented generation (RAG) flow: retrieve relevant document text first, then use that text as context for an LLM response.

## Run locally

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and an OpenAI Platform API key with access to the configured models. Keep the key in the `OPENAI_API_KEY` environment variable.
Make sure to not commit it.

```powershell
uv sync
uv run python -m uvicorn document_qa.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. Upload a text-based PDF through `/documents`, then ask a question through `/ask`. API requests may send document chunks and questions to OpenAI.

The project settings in `document_qa/config.py` define the chunk size, overlap, model names, Chroma collection, and storage path. The `.chroma` directory is created relative to the directory from which the app is started and is excluded from Git.

## Learning resources and development process

I used the [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview), especially its [Chroma integration guide](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma).
Also, for quick explanations I used ChatGPT and self-made Codex skills for routine code-quality and docstring creation.

The project uses `pyproject.toml` and `uv.lock` for dependency management. The code is organized into API, configuration, schema, and RAG modules under `document_qa/`.
