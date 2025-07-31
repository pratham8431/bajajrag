import aiohttp
import fitz  # PyMuPDF
from typing import List, Tuple

async def download_pdf_from_url(url: str) -> bytes:
    """
    Download a PDF from the given URL into memory.
    Raises on non-200 HTTP.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch PDF ({resp.status})")
            return await resp.read()

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
