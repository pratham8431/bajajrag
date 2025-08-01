import aiohttp
import email
from email import policy
from typing import List, Tuple, Dict, Any
import io

async def download_email_from_url(url: str) -> bytes:
    """
    Download an email file from the given URL into memory.
    Raises on non-200 HTTP.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch email ({resp.status})")
            return await resp.read()

def extract_text_from_email(email_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extracts text from email content.
    Returns list of (section_number, text).
    """
    sections: List[Tuple[int, str]] = []
    
    # Parse email
    msg = email.message_from_bytes(email_bytes, policy=policy.default)
    
    # Extract headers
    headers_text = ""
    for header, value in msg.items():
        headers_text += f"{header}: {value}\n"
    
    if headers_text.strip():
        sections.append((1, f"Email Headers:\n{headers_text.strip()}"))
    
    # Extract body
    body_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body_text += part.get_content()
    else:
        body_text = msg.get_content()
    
    if body_text.strip():
        sections.append((2, f"Email Body:\n{body_text.strip()}"))
    
    return sections

def extract_email_metadata(email_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts metadata from email.
    """
    msg = email.message_from_bytes(email_bytes, policy=policy.default)
    
    metadata = {
        "subject": msg.get("subject", ""),
        "from": msg.get("from", ""),
        "to": msg.get("to", ""),
        "date": msg.get("date", ""),
        "content_type": msg.get_content_type(),
        "has_attachments": bool(msg.get_payload())
    }
    
    return metadata
