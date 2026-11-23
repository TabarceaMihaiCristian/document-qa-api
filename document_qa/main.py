"""Create the FastAPI application and register its document Q&A routes."""
from fastapi import FastAPI
from .api.routes import router

app = FastAPI(title="Document Q&A API")

app.include_router(router)
