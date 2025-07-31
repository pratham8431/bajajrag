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
        if len(clean) < 200 and len(re.findall(r'\d{1,3}', clean)) > 5:
            continue
        full_text.append(f"\n\n<PAGE {pg}>\n{clean}")
    joined = "\n".join(full_text)
    joined = re.split(r'(?:CONTENTS|PREFACE)', joined, flags=re.IGNORECASE, maxsplit=1)[-1]
    parts = ARTICLE_HEADING_RE.split(joined)
    sections = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body  = parts[i+1].strip()
        sections.append({"section_title": title, "section_text": body})
    return sections

def chunk_sections(
    sections: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict]:
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
