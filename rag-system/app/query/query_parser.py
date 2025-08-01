# app/query/query_parser.py

import os
import openai
import asyncio
import json
from dotenv import load_dotenv
from typing import Dict
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
    logger.info("Using Azure OpenAI for query parsing")
else:
    # Fallback to OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Either Azure OpenAI or OpenAI API key is required")
    openai.api_key = api_key
    logger.info("Using OpenAI for query parsing")

def _parse_query_azure(question: str, model: str) -> Dict:
    """Synchronous Azure OpenAI query parsing."""
    prompt = f"""
You are an advanced query parser for insurance, legal, and HR documents.
Extract structured information from the question below.

Return JSON with these fields:
{{
  "intent": "coverage_check|waiting_period|exclusion_check|benefit_calculation|policy_terms",
  "clause_type": "maternity|surgery|pre_existing|dental|vision|mental_health|etc",
  "conditions": ["waiting_period", "limitations", "exclusions", "requirements"],
  "policy_section": "health_coverage|life_insurance|disability|liability|etc",
  "specific_terms": ["24 months", "cataract surgery", "maternity expenses"],
  "comparison_type": "coverage_check|benefit_amount|waiting_period|eligibility",
  "document_type": "policy|contract|agreement|guidelines"
}}

Question: \"\"\"{question}\"\"\"

Focus on:
1. What type of coverage/benefit is being asked about?
2. What specific conditions or terms are mentioned?
3. What policy section would contain this information?
4. What type of comparison or check is needed?

Return only the JSON.
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You parse questions into JSON."},
                  {"role":"user","content":prompt}],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    return json.loads(text)

async def parse_query(question: str) -> Dict:
    """
    Advanced query parser for insurance/legal/HR documents.
    Extracts structured information for clause matching.
    """
    # Reload environment variables to get the latest deployment name
    load_dotenv()
    from ..utils.config import config
    
    # Get the deployment name dynamically
    if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
        MODEL = config.AZURE_GPT35_DEPLOYMENT
        logger.info(f"Using Azure OpenAI deployment: {MODEL}")
        # Azure OpenAI - synchronous, run in thread
        return await asyncio.to_thread(_parse_query_azure, question, MODEL)
    else:
        MODEL = "gpt-3.5-turbo"
        logger.info("Using OpenAI for query parsing")
        # OpenAI - asynchronous
        prompt = f"""
You are an advanced query parser for insurance, legal, and HR documents.
Extract structured information from the question below.

Return JSON with these fields:
{{
  "intent": "coverage_check|waiting_period|exclusion_check|benefit_calculation|policy_terms",
  "clause_type": "maternity|surgery|pre_existing|dental|vision|mental_health|etc",
  "conditions": ["waiting_period", "limitations", "exclusions", "requirements"],
  "policy_section": "health_coverage|life_insurance|disability|liability|etc",
  "specific_terms": ["24 months", "cataract surgery", "maternity expenses"],
  "comparison_type": "coverage_check|benefit_amount|waiting_period|eligibility",
  "document_type": "policy|contract|agreement|guidelines"
}}

Question: \"\"\"{question}\"\"\"

Focus on:
1. What type of coverage/benefit is being asked about?
2. What specific conditions or terms are mentioned?
3. What policy section would contain this information?
4. What type of comparison or check is needed?

Return only the JSON.
"""
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":"You parse questions into JSON."},
                      {"role":"user","content":prompt}],
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
