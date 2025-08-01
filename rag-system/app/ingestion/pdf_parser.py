import aiohttp
import fitz  # PyMuPDF
from typing import List, Tuple

async def download_pdf_from_url(url: str, max_size_mb: int = 50) -> bytes:
    """
    Download a PDF from the given URL into memory.
    Raises on non-200 HTTP or if file is too large.
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch PDF ({resp.status})")
            
            # Check content length if available
            content_length = resp.headers.get('content-length')
            if content_length and int(content_length) > max_size_bytes:
                raise ValueError(f"PDF file too large: {int(content_length) // (1024*1024)}MB (max: {max_size_mb}MB)")
            
            # Read in chunks to avoid memory issues
            data = bytearray()
            async for chunk in resp.content.iter_chunked(8192):
                data.extend(chunk)
                if len(data) > max_size_bytes:
                    raise ValueError(f"PDF file too large: {len(data) // (1024*1024)}MB (max: {max_size_mb}MB)")
            
            return bytes(data)

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extracts text from each page of the PDF.
    Returns list of (1‐based page_number, text).
    """
    pages: List[Tuple[int, str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append((i + 1, text))
    return pages
