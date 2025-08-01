from typing import List, Dict, Any
from ..query.retriever import hybrid_retriever

class AdvancedResponseFormatter:
    """
    Advanced response formatter for structured JSON output.
    Creates detailed responses with clause references and explanations.
    """
    
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def format_structured_response(
        self,
        question: str,
        answer: str,
        justification: str,
        retrieved_chunks: List[Dict[str, Any]],
        parsed_query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format response into structured JSON with detailed information.
        """
        # Extract clause references
        clause_references = self._extract_clause_references(retrieved_chunks)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(retrieved_chunks)
        
        # Determine response type
        response_type = self._determine_response_type(parsed_query)
        
        # Format coverage details
        coverage_details = self._extract_coverage_details(answer, retrieved_chunks)
        
        return {
            "question": question,
            "answer": answer,
            "justification": justification,
            "confidence_score": confidence_score,
            "response_type": response_type,
            "clause_references": clause_references,
            "coverage_details": coverage_details,
            "query_analysis": {
                "intent": parsed_query.get("intent"),
                "clause_type": parsed_query.get("clause_type"),
                "policy_section": parsed_query.get("policy_section"),
                "specific_terms": parsed_query.get("specific_terms", [])
            },
            "retrieval_metadata": {
                "total_chunks_retrieved": len(retrieved_chunks),
                "search_methods_used": list(set(chunk.get("method", "unknown") for chunk in retrieved_chunks)),
                "top_chunk_score": max([chunk.get("score", 0) for chunk in retrieved_chunks]) if retrieved_chunks else 0
            }
        }
    
    def _extract_clause_references(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract structured clause references from retrieved chunks."""
        references = []
        
        for chunk in chunks:
            reference = {
                "chunk_id": chunk.get("id"),
                "score": chunk.get("score", 0),
                "search_method": chunk.get("method", "unknown"),
                "metadata": chunk.get("metadata", {}),
                "text_snippet": chunk.get("chunk_text", "")[:200] + "..." if len(chunk.get("chunk_text", "")) > 200 else chunk.get("chunk_text", "")
            }
            references.append(reference)
        
        return references
    
    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not chunks:
            return 0.0
        
        # Average score of top chunks
        scores = [chunk.get("score", 0) for chunk in chunks]
        avg_score = sum(scores) / len(scores)
        
        # Bonus for multiple search methods
        methods = set(chunk.get("method", "unknown") for chunk in chunks)
        method_bonus = min(len(methods) * 0.1, 0.3)
        
        # Penalty for low scores
        if avg_score < 0.3:
            avg_score *= 0.5
        
        confidence = min(avg_score + method_bonus, 1.0)
        return round(confidence, 3)
    
    def _determine_response_type(self, parsed_query: Dict[str, Any]) -> str:
        """Determine the type of response based on parsed query."""
        intent = parsed_query.get("intent", "")
        
        if "coverage" in intent:
            return "coverage_decision"
        elif "waiting" in intent:
            return "waiting_period_info"
        elif "exclusion" in intent:
            return "exclusion_check"
        elif "benefit" in intent:
            return "benefit_calculation"
        else:
            return "general_inquiry"
    
    def _extract_coverage_details(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract specific coverage details from answer and chunks."""
        details = {
            "is_covered": None,
            "waiting_period": None,
            "limitations": [],
            "requirements": [],
            "exclusions": []
        }
        
        # Simple keyword extraction for coverage details
        answer_lower = answer.lower()
        
        # Check if covered
        if any(word in answer_lower for word in ["covered", "yes", "eligible"]):
            details["is_covered"] = True
        elif any(word in answer_lower for word in ["not covered", "no", "excluded"]):
            details["is_covered"] = False
        
        # Extract waiting period
        import re
        waiting_match = re.search(r'(\d+)\s*(month|year|day)', answer_lower)
        if waiting_match:
            details["waiting_period"] = f"{waiting_match.group(1)} {waiting_match.group(2)}s"
        
        # Extract limitations and requirements from chunks
        for chunk in chunks:
            chunk_text = chunk.get("chunk_text", "").lower()
            if "limit" in chunk_text or "maximum" in chunk_text:
                details["limitations"].append(chunk.get("chunk_text", "")[:100])
            if "require" in chunk_text or "must" in chunk_text:
                details["requirements"].append(chunk.get("chunk_text", "")[:100])
            if "exclude" in chunk_text or "not covered" in chunk_text:
                details["exclusions"].append(chunk.get("chunk_text", "")[:100])
        
        return details

# Global formatter instance
advanced_formatter = AdvancedResponseFormatter()
