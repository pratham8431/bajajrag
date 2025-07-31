from fastapi import FastAPI
from app.api.endpoints.query import router as query_router

app = FastAPI(
    title="HackRx RAG API",
    description="LLM-powered document retrieval and reasoning",
    version="1.0.0"
)

# Register endpoints
app.include_router(query_router, prefix="/api/v1/hackrx")
