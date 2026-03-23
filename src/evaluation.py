"""
Evaluation Module with RAGAS Metrics
Tracks faithfulness, answer relevance, context precision, and context recall
"""
import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path


"""
RAG Evaluation using RAGAS metrics
"""
from typing import List, Dict, Any
import logging

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_relevancy
except ImportError:
    ragas_evaluate = None
    faithfulness = answer_relevancy = context_precision = context_recall = context_relevancy = None

logger = logging.getLogger(__name__)

class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG pipeline
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate(self, queries: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """
        Evaluate pipeline using RAGAS metrics
        Args:
            queries: List of user queries
            ground_truths: List of ground truth answers
        Returns:
            Dict of metric scores
        """
        if ragas_evaluate is None:
            logger.warning("RAGAS not installed. Please install ragas to use evaluation.")
            return {"error": "RAGAS not installed"}

        results = []
        for query, ground_truth in zip(queries, ground_truths):
            # Retrieve context and generate answer
            docs = self.pipeline.retrieve(query)
            context = "\n\n".join([doc["text"] for doc in docs])
            answer = self.pipeline.llm.generate_with_context(query, context)
            # Evaluate with RAGAS
            metrics = ragas_evaluate(
                query=query,
                answer=answer,
                contexts=[context],
                ground_truth=ground_truth,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall, context_relevancy]
            )
            results.append({
                "query": query,
                "answer": answer,
                "ground_truth": ground_truth,
                "metrics": metrics
            })
        logger.info(f"Evaluated {len(results)} queries with RAGAS.")
        return {"results": results}

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from .rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    evaluator = RAGASEvaluator(pipeline)
    queries = ["What is RAG?"]
    ground_truths = ["RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval of external documents with generative models."]
    scores = evaluator.evaluate(queries, ground_truths)
    print(scores)
    logging.warning("RAGAS not available. Install with: pip install ragas")

logger = logging.getLogger(__name__)

# ...existing code for RAGEvaluator and PerformanceTracker...
