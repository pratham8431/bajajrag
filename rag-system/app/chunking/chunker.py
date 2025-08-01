import re
import uuid
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

ARTICLE_HEADING_RE = re.compile(
    r'^(PART [IVXLC]+|ARTICLE\s+\d+[A-Z]?\.)',
    flags=re.MULTILINE
)

def split_into_sections(pages: List[Tuple[int, str]]) -> List[Dict]:
    full_text = []
    for pg, text in pages:
        clean = text.strip().replace("\xa0", " ")
        if len(clean) < 50:  # Skip very short pages
            continue
        full_text.append(f"\n\n<PAGE {pg}>\n{clean}")
    
    if not full_text:
        # If no content found, create a single section
        return [{"section_title": "Document", "section_text": "No readable content found in document."}]
    
    joined = "\n".join(full_text)
    
    # Try to split by article patterns first
    parts = ARTICLE_HEADING_RE.split(joined)
    sections = []
    
    if len(parts) > 1:
        # Found article patterns
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                title = parts[i].strip()
                body = parts[i + 1].strip()
                if body:  # Only add if there's content
                    sections.append({"section_title": title, "section_text": body})
    else:
        # No article patterns found, treat as single section
        sections.append({"section_title": "Document", "section_text": joined})
    
    return sections

from ..utils.config import config

def chunk_sections(
    sections: List[Dict],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Dict]:
    # Use config values if not provided
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for sec in sections:
        title = sec["section_title"]
        text  = sec["section_text"]
        for idx, piece in enumerate(splitter.split_text(text)):
            chunks.append({
                "id": str(uuid.uuid4()),
                "chunk_text": piece,
                "metadata": {"section": title, "chunk_index": idx}
            })
    return chunks

def chunk_text_by_page(
    pages: List[Tuple[int, str]],
    document_name: str,
    max_total_chunks: int = 500
) -> List[Dict]:
    sections = split_into_sections(pages)
    all_chunks = chunk_sections(sections)
    return all_chunks[:max_total_chunks]
