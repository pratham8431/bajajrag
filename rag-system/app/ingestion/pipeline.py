# app/ingestion/pipeline.py

import os
import logging
import asyncio
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

# Document parsing
from .pdf_parser import download_pdf_from_url, extract_text_from_pdf
from .docx_parser import download_docx_from_url, extract_text_from_docx, extract_tables_from_docx
from .email_parser import download_email_from_url, extract_text_from_email, extract_email_metadata
# Chunking
from app.chunking.chunker import chunk_text_by_page
# Embeddings
from ..embeddings.embedder import embed_chunks_openai, embed_chunks_openai_sync
from ..embeddings.embedder import client
# Pinecone
from ..vectorstore.faiss_client import upsert_to_faiss
# Database
from ..db.db_utils import SessionLocal, init_db
from ..db.models import Document, Chunk

# — Initialize environment & DB —
load_dotenv()
init_db()  # create tables if they don't yet exist

# — Logger setup —
from ..utils.logger import setup_logger
logger = setup_logger(__name__)

async def generate_document_title(document_text: str, document_name: str) -> str:
    """
    Generate a succinct document title using LLM.
    This title will be prepended to all chunks for better context.
    """
    try:
        # Reload environment variables to get the latest deployment name
        load_dotenv()
        from ..utils.config import config
        
        # Get the deployment name dynamically
        if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
            MODEL = config.AZURE_GPT35_DEPLOYMENT
            logger.info(f"Using Azure OpenAI deployment for title generation: {MODEL}")
            
            # Azure OpenAI - synchronous, run in thread
            def _generate_title_azure():
                prompt = f"""
                You are a document analyzer. Given the following document content, generate a concise, descriptive title (max 10 words) that captures the main topic or purpose of the document.
                
                Document Name: {document_name}
                Document Content (first 2000 characters): {document_text[:2000]}
                
                Return only the title, nothing else. Make it specific and informative.
                """
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You generate concise document titles."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                return resp.choices[0].message.content.strip()
            
            title = await asyncio.to_thread(_generate_title_azure)
        else:
            MODEL = "gpt-3.5-turbo"
            logger.info("Using OpenAI for title generation")
            
            prompt = f"""
            You are a document analyzer. Given the following document content, generate a concise, descriptive title (max 10 words) that captures the main topic or purpose of the document.
            
            Document Name: {document_name}
            Document Content (first 2000 characters): {document_text[:2000]}
            
            Return only the title, nothing else. Make it specific and informative.
            """
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You generate concise document titles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            title = resp.choices[0].message.content.strip()
        
        logger.info(f"Generated document title: '{title}'")
        return title
        
    except Exception as e:
        logger.error(f"Failed to generate document title: {e}")
        # Fallback to document name
        return document_name

async def ingest_document(
    url: str,
    document_name: str,
    max_chunks: int = None
) -> List[Dict[str, Any]]:
    """
    Advanced ingestion pipeline supporting multiple document types:
      1) Download document → detect type → extract text
      2) Generate document title using LLM
      3) Chunk into semantic sections
      4) Add contextual headers to chunks
      5) Persist Document + Chunk metadata in Postgres
      6) Generate embeddings (with headers)
      7) Upsert embeddings to FAISS
    Returns the list of embedded chunk dicts.
    """
    logger.info(f"▶️ Starting advanced ingestion for '{document_name}' from URL: {url}")

    # 1) Download & parse document (detect type automatically)
    document_type, pages, metadata = await _parse_document(url, document_name)
    logger.info(f"Parsed {len(pages)} sections from {document_type.upper()}")

    # 2) Generate document title using LLM
    # Combine first few pages for title generation
    title_text = ""
    for i, (page_num, text) in enumerate(pages[:3]):  # Use first 3 pages
        title_text += f"\nPage {page_num}: {text[:500]}"  # First 500 chars per page
    
    document_title = await generate_document_title(title_text, document_name)
    logger.info(f"Generated document title: '{document_title}'")

    # 3) Chunk into semantic sections
    chunks = chunk_text_by_page(
        pages,
        document_name=document_name,
        max_total_chunks=max_chunks or int(os.getenv("MAX_CHUNKS", "500"))
    )
    logger.info(f"Created {len(chunks)} chunks")

    # 4) Add contextual headers to chunks
    for chunk in chunks:
        # Create text_for_embedding with document title header
        header = f"Document Title: {document_title}\n\n"
        chunk["text_for_embedding"] = header + chunk["chunk_text"]
        # Keep original chunk_text for metadata
        chunk["metadata"]["document_title"] = document_title
    
    logger.info(f"Added contextual headers to {len(chunks)} chunks")

    # 5) Persist to PostgreSQL
    db = SessionLocal()
    try:
        # 5a) Document record
        doc = Document(name=document_name, url=url)
        db.add(doc)
        db.flush()  # assigns doc.id

        # 5b) Chunk records
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
        
        # Store document ID for later use
        document_id = doc.id
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error during ingestion: {e}")
        raise
    finally:
        db.close()

    # 6) Generate embeddings (batched + retry) - using text_for_embedding
    try:
        logger.info(f"Checking client type: {type(client)}")
        is_azure = 'AzureOpenAI' in str(type(client))
        logger.info(f"Is Azure client: {is_azure}")
        if is_azure:
            # Azure OpenAI - synchronous
            logger.info("Using Azure OpenAI sync embedding")
            embedded_chunks = embed_chunks_openai_sync(chunks)
        else:
            # OpenAI - asynchronous
            logger.info("Using OpenAI async embedding")
            embedded_chunks = await embed_chunks_openai(chunks)
    except Exception as e:
        logger.error(f"Embedding failed but document was already persisted: {e}")
        # Return chunks without embeddings for fallback
        return chunks
    logger.info("Generated embeddings for all chunks")

    # 7) Upsert to FAISS (batched + retry)
    upsert_to_faiss(embedded_chunks)
    logger.info("Upserted embeddings to FAISS")

    return embedded_chunks

async def _parse_document(url: str, document_name: str) -> Tuple[str, List[Tuple[int, str]], Dict[str, Any]]:
    """
    Parse document based on file extension or content type.
    Returns (document_type, pages, metadata)
    """
    file_extension = document_name.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        # Parse PDF
        pdf_bytes = await download_pdf_from_url(url)
        pages = extract_text_from_pdf(pdf_bytes)
        metadata = {"document_type": "pdf", "total_pages": len(pages)}
        return "pdf", pages, metadata
    
    elif file_extension in ['docx', 'doc']:
        # Parse DOCX
        docx_bytes = await download_docx_from_url(url)
        pages = extract_text_from_docx(docx_bytes)
        tables = extract_tables_from_docx(docx_bytes)
        metadata = {
            "document_type": "docx", 
            "total_paragraphs": len(pages),
            "total_tables": len(tables)
        }
        return "docx", pages, metadata
    
    elif file_extension in ['eml', 'msg']:
        # Parse Email
        email_bytes = await download_email_from_url(url)
        pages = extract_text_from_email(email_bytes)
        email_metadata = extract_email_metadata(email_bytes)
        metadata = {
            "document_type": "email",
            "subject": email_metadata.get("subject"),
            "from": email_metadata.get("from"),
            "to": email_metadata.get("to"),
            "date": email_metadata.get("date")
        }
        return "email", pages, metadata
    
    else:
        # Try to parse as PDF by default
        try:
            pdf_bytes = await download_pdf_from_url(url)
            pages = extract_text_from_pdf(pdf_bytes)
            metadata = {"document_type": "pdf", "total_pages": len(pages)}
            return "pdf", pages, metadata
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            raise ValueError(f"Unsupported document type: {file_extension}")
