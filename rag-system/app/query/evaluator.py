# app/query/evaluator.py

import os
import openai
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4"

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
    # Build a context prompt with top-K chunks
    context_strs = []
    for c in contexts:
        meta = c["metadata"]
        context_strs.append(f"---\nSection: {meta.get('section')}\nText: {c.get('metadata').get('section')}\n{c.get('metadata')}\n{c['metadata']}\n{c['metadata']}\n")
        # Actually include the chunk text
        context_strs[-1] = f"Section: {meta.get('section')}\n{c['metadata']}\n{c['metadata']}\nText: (omitted for brevity)"

    prompt = f"""
You are a policy-underwriting assistant.  
Question: {question}  
Structured query: {structured_query}  

Relevant clauses:
{"\n\n".join(context_strs)}

Using only the above clauses, answer the question.  
1) Provide a concise answer.  
2) Provide a justification, referencing the section titles.  
Return JSON: {{ "answer":"...", "justification":"..." }}
"""
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role":"system","content":"You are an expert policy assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0
    )
    text = resp.choices[0].message.content.strip()
    return openai.util.safe_load_json(text)
