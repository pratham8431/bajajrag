# app/ingestion/pipeline.py

import os
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

# PDF parsing
from app.ingestion.pdf_parser import download_pdf_from_url, extract_text_from_pdf
# Chunking
from app.chunking.chunker import chunk_text_by_page
# Embeddings
from app.embeddings.embedder import embed_chunks_openai
# Pinecone
from app.vectorstore.pinecone_client import upsert_to_pinecone
# Database
from app.db.db_utils import SessionLocal, init_db
from app.db.models import Document, Chunk

# — Initialize environment & DB —
load_dotenv()
init_db()  # create tables if they don’t yet exist

# — Logger setup —
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

async def ingest_document(
    url: str,
    document_name: str,
    max_chunks: int = None
) -> List[Dict[str, Any]]:
    """
    Full ingestion pipeline:
      1) Download PDF → extract text pages
      2) Chunk into semantic sections
      3) Persist Document + Chunk metadata in Postgres
      4) Generate embeddings
      5) Upsert embeddings to Pinecone
    Returns the list of embedded chunk dicts.
    """
    logger.info(f"▶️ Starting ingestion for '{document_name}' from URL: {url}")

    # 1) Download & parse PDF
    pdf_bytes = await download_pdf_from_url(url)
    pages = extract_text_from_pdf(pdf_bytes)
    logger.info(f"Parsed {len(pages)} pages from PDF")

    # 2) Chunk into semantic sections
    chunks = chunk_text_by_page(
        pages,
        document_name=document_name,
        max_total_chunks=max_chunks or int(os.getenv("MAX_CHUNKS", "500"))
    )
    logger.info(f"Created {len(chunks)} chunks")

    # 3) Persist to PostgreSQL
    db = SessionLocal()
    try:
        # 3a) Document record
        doc = Document(name=document_name, url=url)
        db.add(doc)
        db.flush()  # assigns doc.id

        # 3b) Chunk records
        for c in chunks:
            db.add(Chunk(
                id=c["id"],
                document_id=doc.id,
                page=c["metadata"].get("page"),
                section=c["metadata"].get("section"),
                metadata=c["metadata"]
            ))

        db.commit()
        logger.info(f"Persisted Document (id={doc.id}) and {len(chunks)} chunks to Postgres")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during ingestion: {e}")
        raise
    finally:
        db.close()

    # 4) Generate embeddings (batched + retry)
    embedded_chunks = await embed_chunks_openai(chunks)
    logger.info("Generated embeddings for all chunks")

    # 5) Upsert to Pinecone (batched + retry)
    upsert_to_pinecone(embedded_chunks)
    logger.info("Upserted embeddings to Pinecone")

    return embedded_chunks
