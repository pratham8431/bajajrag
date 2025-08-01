# app/query/evaluator.py

import os
import openai
import asyncio
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

load_dotenv()
from ..utils.config import config

# Check if Azure OpenAI is configured
if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
    # Use Azure OpenAI
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=config.AZURE_OPENAI_API_KEY,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_version=config.AZURE_OPENAI_API_VERSION
    )
    logger.info("Using Azure OpenAI for answer evaluation")
else:
    # Fallback to OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Either Azure OpenAI or OpenAI API key is required")
    openai.api_key = api_key
    logger.info("Using OpenAI for answer evaluation")

def _evaluate_answer_azure(
    question: str,
    structured_query: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    model: str
) -> Dict[str, str]:
    """Synchronous Azure OpenAI answer evaluation."""
    # Build a context prompt with top-K chunks
    context_strs = []
    for c in contexts:
        meta = c["metadata"]
        # Use text_for_embedding if available (contains document title header), otherwise fall back to chunk_text
        chunk_text = c.get("text_for_embedding", c["chunk_text"])
        context_strs.append(f"Section: {meta.get('section')}\nText: {chunk_text}")

    prompt = f"""
You are a policy-underwriting assistant.  
Question: {question}  
Structured query: {structured_query}  

Relevant clauses:
{chr(10).join(context_strs)}

Using only the above clauses, answer the question.  
1) Provide a concise answer.  
2) Provide a justification, referencing the section titles.  
Return JSON: {{ "answer":"...", "justification":"..." }}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are an expert policy assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    logger.info(f"Azure evaluator response: {text}")
    try:
        # Handle markdown-wrapped JSON responses
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        if text.endswith("```"):
            text = text[:-3]  # Remove ```
        text = text.strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}, text: {text}")
        # Return a fallback response
        return {
            "answer": "Unable to parse response from AI model",
            "justification": f"Error parsing JSON response: {text[:100]}..."
        }

async def evaluate_answer(
    question: str,
    structured_query: Dict[str, Any],
    contexts: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Returns:
    {
      "answer": "...",
      "justification": "Based on Clause X on page Y...",
    }
    """
    # Reload environment variables to get the latest deployment name
    load_dotenv()
    from ..utils.config import config
    
    # Get the deployment name dynamically
    if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
        MODEL = config.AZURE_GPT35_DEPLOYMENT
        logger.info(f"Using Azure OpenAI deployment: {MODEL}")
        # Azure OpenAI - synchronous, run in thread
        return await asyncio.to_thread(_evaluate_answer_azure, question, structured_query, contexts, MODEL)
    else:
        MODEL = "gpt-3.5-turbo"
        logger.info("Using OpenAI for answer evaluation")
        # Build a context prompt with top-K chunks
        context_strs = []
        for c in contexts:
            meta = c["metadata"]
            # Use text_for_embedding if available (contains document title header), otherwise fall back to chunk_text
            chunk_text = c.get("text_for_embedding", c["chunk_text"])
            context_strs.append(f"Section: {meta.get('section')}\nText: {chunk_text}")

        prompt = f"""
You are a policy-underwriting assistant.  
Question: {question}  
Structured query: {structured_query}  

Relevant clauses:
{chr(10).join(context_strs)}

Using only the above clauses, answer the question.  
1) Provide a concise answer.  
2) Provide a justification, referencing the section titles.  
Return JSON: {{ "answer":"...", "justification":"..." }}
"""
        # OpenAI - asynchronous
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are an expert policy assistant."},
                {"role":"user","content":prompt}
            ],
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        logger.info(f"OpenAI evaluator response: {text}")
        try:
            # Handle markdown-wrapped JSON responses
            if text.startswith("```json"):
                text = text[7:]  # Remove ```json
            if text.endswith("```"):
                text = text[:-3]  # Remove ```
            text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}, text: {text}")
            # Return a fallback response
            return {
                "answer": "Unable to parse response from AI model",
                "justification": f"Error parsing JSON response: {text[:100]}..."
            }
