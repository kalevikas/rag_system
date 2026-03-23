"""
Query Expansion Module
Enhances user queries to improve retrieval accuracy for imprecise or vague questions
"""
import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand and enhance user queries for better retrieval"""
    
    def __init__(self):
        """Initialize query expander with common term mappings"""
        # Common abbreviations and synonyms in technical documentation
        self.term_mappings = {
            # Common abbreviations
            "kb": ["knowledge base", "knowledge article"],
            "db": ["database"],
            "api": ["application programming interface", "api"],
            "ui": ["user interface", "interface"],
            "auth": ["authentication", "authorization"],
            "config": ["configuration", "settings"],
            "admin": ["administrator", "administration"],
            "mgmt": ["management"],
            "svc": ["service"],
            
            # Action verbs
            "setup": ["setup", "configure", "install", "set up"],
            "fix": ["fix", "resolve", "troubleshoot", "repair", "solve"],
            "create": ["create", "add", "new", "generate"],
            "delete": ["delete", "remove", "erase"],
            "update": ["update", "modify", "change", "edit"],
            "enable": ["enable", "activate", "turn on"],
            "disable": ["disable", "deactivate", "turn off"],
            
            # Common questions
            "how": ["how to", "steps", "procedure", "guide", "instructions"],
            "what": ["what is", "definition", "explain", "describe"],
            "why": ["why", "reason", "cause"],
            "when": ["when", "timing", "schedule"],
        }
        
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple variants for better retrieval
        
        Args:
            query: Original user query
            
        Returns:
            List of query variants including original
        """
        variants = [query]  # Always include original
        
        query_lower = query.lower()
        
        # Add question word expansion
        if any(q in query_lower for q in ["how", "what", "why", "when", "where"]):
            # Add variant with "to" for how-to questions
            if "how" in query_lower and "to" not in query_lower:
                variants.append(query_lower.replace("how", "how to"))
                variants.append(query_lower.replace("how", "steps to"))
                variants.append(query_lower.replace("how", "procedure for"))
        
        # Add variants for common abbreviations
        for abbr, expansions in self.term_mappings.items():
            if abbr in query_lower.split():
                for expansion in expansions:
                    variants.append(query_lower.replace(abbr, expansion))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)
        
        if len(unique_variants) > 1:
            logger.debug(f"Query expanded from '{query}' to {len(unique_variants)} variants")
        
        return unique_variants
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from query for better matching
        
        Args:
            query: User query
            
        Returns:
            List of key terms
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'can', 'could', 'would', 'should',
            'i', 'you', 'me', 'my', 'we', 'our'
        }
        
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return key_terms
    
    def enhance_query_context(self, query: str) -> Dict[str, Any]:
        """
        Generate enhanced query context for better retrieval
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with enhanced query information
        """
        return {
            'original': query,
            'variants': self.expand_query(query),
            'key_terms': self.extract_key_terms(query),
            'is_question': any(q in query.lower() for q in ['how', 'what', 'why', 'when', 'where', 'who', '?']),
            'is_troubleshooting': any(t in query.lower() for t in ['fix', 'error', 'issue', 'problem', 'not working', 'failed']),
            'is_howto': any(h in query.lower() for h in ['how', 'steps', 'procedure', 'guide'])
        }
    
    def get_primary_query(self, query: str, conservative: bool = True) -> str:
        """
        Get the best primary query for initial retrieval
        
        Args:
            query: Original query
            conservative: If True, only enhance clearly incomplete queries
            
        Returns:
            Enhanced primary query
        """
        # If conservative mode, don't modify most queries
        if conservative:
            # Only help very short or clearly incomplete queries
            if len(query.split()) <= 2:
                context = self.enhance_query_context(query)
                # Only for single-word troubleshooting terms
                if context['is_troubleshooting'] and len(query.split()) == 1:
                    return f"{query} troubleshooting"
            return query
        
        # Aggressive mode (original behavior)
        context = self.enhance_query_context(query)
        
        # For how-to questions, ensure "how to" is present
        if context['is_howto'] and 'how to' not in query.lower():
            return f"how to {query}"
        
        # For troubleshooting, add context
        if context['is_troubleshooting'] and 'how to' not in query.lower():
            return f"how to resolve {query}"
        
        return query
