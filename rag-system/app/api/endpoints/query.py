# app/api/endpoints/query.py

import os
import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment
load_dotenv()
from ...utils.config import config

# Validate configuration
config.validate()

# App modules
from ...ingestion.pipeline           import ingest_document
from ...query.query_parser          import parse_query
from ...embeddings.embedder         import _embed_batch_sync, _embed_batch_async, client
from ...vectorstore.pinecone_client import query_pinecone
from ...query.evaluator             import evaluate_answer
from ...query.retriever             import hybrid_retriever
from ...query.formatter             import advanced_formatter

# Router setup
router = APIRouter()
from ...utils.logger import setup_logger
logger = setup_logger(__name__)

# Pydantic models
from pydantic import BaseModel, validator, HttpUrl
from typing import List, Dict, Any

class RunRequest(BaseModel):
    documents: HttpUrl          # URL to the PDF
    questions: List[str]        # List of questions
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question is required')
        if len(v) > 10:
            raise ValueError('Maximum 10 questions allowed per request')
        return v

class ClauseRef(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

class QAItem(BaseModel):
    question: str
    answer: str
    justification: str
    sources: List[ClauseRef]

class RunResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/run", response_model=RunResponse)
async def run_handler(req: RunRequest):
    # ─── Ingest & index ───────────────────────────────────────────────────────
    try:
        # Ingest the document: download, parse, chunk, embed, upsert
        embedded_chunks = await ingest_document(
            url=req.documents,
            document_name=os.path.basename(req.documents)
        )
        logger.info(f"Ingested and indexed {len(embedded_chunks)} chunks")
    except Exception as e:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

    # Build a lookup for quick chunk_text retrieval
    chunk_map = {c["id"]: c for c in embedded_chunks}

    # ─── Question Answering ───────────────────────────────────────────────────
    results: List[QAItem] = []
    for question in req.questions:
        # 1) Parse the question into structured JSON
        try:
            parsed = await parse_query(question)
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query parsing failed: {e}")

        # 2) Embed the question itself
        try:
            is_azure = 'AzureOpenAI' in str(type(client))
            if is_azure:
                # Azure OpenAI - synchronous
                question_embedding = (_embed_batch_sync([question]))[0]
            else:
                # OpenAI - asynchronous
                question_embedding = (await _embed_batch_async([question]))[0]
        except Exception as e:
            logger.error(f"Question embedding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Question embedding failed: {e}")

        # 3) Perform hybrid search (semantic + keyword + exact match)
        try:
            retrieved_chunks = await hybrid_retriever.hybrid_search(
                query=question,
                chunks=embedded_chunks,
                top_k=10
            )
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks using hybrid search for question: {question}")
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {e}")

        # 4) Prepare contexts for the evaluator
        contexts = []
        for chunk in retrieved_chunks:
            contexts.append({
                "chunk_text": chunk.get("chunk_text", ""),
                "metadata": chunk.get("metadata", {}),
                "search_method": chunk.get("method", "unknown"),
                "score": chunk.get("score", 0)
            })

        # 5) Evaluate answer using the LLM
        try:
            eval_res = await evaluate_answer(
                question=question,
                structured_query=parsed,
                contexts=contexts
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

        # 6) Format structured response
        structured_response = advanced_formatter.format_structured_response(
            question=question,
            answer=eval_res["answer"],
            justification=eval_res["justification"],
            retrieved_chunks=retrieved_chunks,
            parsed_query=parsed
        )

        # 7) Append the structured response
        results.append(structured_response)

    # ─── Return Structured Response ────────────────────────────────────────────
    return RunResponse(results=results)
