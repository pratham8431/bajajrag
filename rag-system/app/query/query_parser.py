# app/query/query_parser.py

import os
import openai
from dotenv import load_dotenv
from typing import Dict

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4"

def parse_query(question: str) -> Dict:
    """
    Calls GPT to extract a structured representation from the question.
    E.g.:
    {
      "intent": "check_coverage",
      "procedure": "knee surgery",
      "location": "Pune",
      "policy_age": "3 months"
    }
    """
    prompt = f"""
You are an intelligent parser. Extract JSON with keys:
- intent (e.g., check_coverage, calculate_waiting_period, list_conditions)
- procedure (if any)
- location (if any)
- duration (if any)
From the question below. If a field is absent, set it to null.
Question: \"\"\"{question}\"\"\"
Return only the JSON.
"""
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role":"system","content":"You parse questions into JSON."},
                  {"role":"user","content":prompt}],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    return openai.util.safe_load_json(text)
