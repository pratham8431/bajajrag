# app/api/endpoints/query.py

import os
import logging
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# App modules
from app.ingestion.pipeline           import ingest_document
from app.query.query_parser          import parse_query
from app.embeddings.embedder         import _embed_batch
from app.vectorstore.pinecone_client import query_pinecone
from app.query.evaluator             import evaluate_answer

# Router setup
router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Pydantic models
class RunRequest(BaseModel):
    documents: str              # URL to the PDF
    questions: List[str]        # List of questions

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
    results: List[QAItem]

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
            parsed = parse_query(question)
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Query parsing failed: {e}")

        # 2) Embed the question itself
        try:
            question_embedding = (await _embed_batch([question]))[0]
        except Exception as e:
            logger.error(f"Question embedding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Question embedding failed: {e}")

        # 3) Retrieve top-K relevant chunks from Pinecone
        try:
            matches = query_pinecone(question_embedding, top_k=5)
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            raise HTTPException(status_code=500, detail=f"Pinecone query failed: {e}")

        logger.info(f"Retrieved {len(matches)} chunks for question: {question}")

        # 4) Prepare contexts for the evaluator
        contexts = []
        for m in matches:
            chunk = chunk_map.get(m["id"])
            if chunk:
                contexts.append({
                    "chunk_text": chunk["chunk_text"],
                    "metadata": m["metadata"]
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

        # 6) Build source references
        sources = [
            ClauseRef(id=m["id"], score=m["score"], metadata=m["metadata"])
            for m in matches
        ]

        # 7) Append the QAItem
        results.append(QAItem(
            question=question,
            answer=eval_res["answer"],
            justification=eval_res["justification"],
            sources=sources
        ))

    # ─── Return Structured Response ────────────────────────────────────────────
    return RunResponse(results=results)
