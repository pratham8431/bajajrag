import re
from typing import List, Tuple

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text content
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove non-breaking spaces
    text = text.replace('\xa0', ' ')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract sections from text based on common patterns.
    
    Args:
        text: Full text content
        
    Returns:
        List of (section_title, section_content) tuples
    """
    # Common section patterns
    section_patterns = [
        r'^(PART [IVXLC]+|ARTICLE\s+\d+[A-Z]?\.)',
        r'^(Section \d+\.)',
        r'^(Chapter \d+\.)',
        r'^(\d+\.\s+[A-Z][^.]*\.)',
    ]
    
    sections = []
    lines = text.split('\n')
    current_section = ""
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line matches any section pattern
        is_section = False
        for pattern in section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                # Save previous section if exists
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                
                # Start new section
                current_section = line
                current_content = []
                is_section = True
                break
        
        if not is_section and current_section:
            current_content.append(line)
    
    # Add the last section
    if current_section and current_content:
        sections.append((current_section, '\n'.join(current_content)))
    
    return sections

def remove_headers_footers(text: str) -> str:
    """
    Remove common headers and footers from text.
    
    Args:
        text: Text content
        
    Returns:
        Text with headers and footers removed
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip common header/footer patterns
        if any(pattern in line.lower() for pattern in [
            'page', 'confidential', 'draft', 'internal use only',
            'copyright', 'all rights reserved', 'proprietary'
        ]):
            continue
            
        # Skip lines that are just numbers (page numbers)
        if re.match(r'^\d+$', line):
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Text content
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove empty lines
    lines = [line for line in lines if line]
    
    return '\n'.join(lines)
