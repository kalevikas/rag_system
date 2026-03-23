
"""
Re-ranking module using cross-encoder for improved relevance
"""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
	"""Re-rank retrieved documents using cross-encoder"""
    
	def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
		"""
		Initialize re-ranker
        
		Args:
			model_name: Cross-encoder model name
		"""
		self.model_name = model_name
		self.model = None
		self._load_model()
    
	def _load_model(self):
		"""Load cross-encoder model"""
		try:
			logger.info(f"Loading cross-encoder model: {self.model_name}")
			self.model = CrossEncoder(self.model_name, max_length=512)
			logger.info("Cross-encoder loaded successfully")
		except Exception as e:
			logger.error(f"Error loading cross-encoder: {e}")
			raise
    
	def rerank(self,
			   query: str,
			   documents: List[Dict[str, Any]],
			   top_k: Optional[int] = None) -> List[Dict[str, Any]]:
		"""
		Re-rank documents based on query relevance
        
		Args:
			query: Query string
			documents: List of retrieved documents with metadata
			top_k: Number of top results to return (None = return all)
            
		Returns:
			Re-ranked list of documents
		"""
		if not documents:
			return []
        
		logger.info(f"Re-ranking {len(documents)} documents")
        
		# Prepare query-document pairs
		pairs = []
		for doc in documents:
			text = doc.get("text", "")
			pairs.append([query, text])
        
		# Compute cross-encoder scores
		try:
			scores = self.model.predict(pairs)
            
			# Add rerank scores to documents
			for i, doc in enumerate(documents):
				doc["rerank_score"] = float(scores[i])
				doc["original_rank"] = i + 1
            
			# Sort by rerank score
			reranked = sorted(
				documents,
				key=lambda x: x["rerank_score"],
				reverse=True
			)
            
			# Add new rank
			for i, doc in enumerate(reranked):
				doc["rerank_position"] = i + 1
            
			# Return top-k if specified
			if top_k is not None:
				reranked = reranked[:top_k]
            
			logger.info(f"Re-ranking complete. Returning top {len(reranked)} documents")
			return reranked
            
		except Exception as e:
			logger.error(f"Error during re-ranking: {e}")
			return documents
    
	def compute_relevance_score(self, query: str, document: str) -> float:
		"""
		Compute relevance score for a single query-document pair
        
		Args:
			query: Query string
			document: Document text
            
		Returns:
			Relevance score
		"""
		try:
			score = self.model.predict([[query, document]])
			return float(score[0])
		except Exception as e:
			logger.error(f"Error computing relevance score: {e}")
			return 0.0


class RerankerPipeline:
	"""Pipeline for retrieval + re-ranking"""
    
	def __init__(self, retriever, reranker: Reranker, initial_k: int = 20, final_k: int = 5):
		"""
		Initialize re-ranker pipeline
        
		Args:
			retriever: Initial retriever (hybrid or dense)
			reranker: Reranker instance
			initial_k: Number of documents to retrieve initially
			final_k: Number of documents to return after re-ranking
		"""
		self.retriever = retriever
		self.reranker = reranker
		self.initial_k = initial_k
		self.final_k = final_k
        
		logger.info(f"RerankerPipeline initialized")
		logger.info(f"  - Initial retrieval: top-{initial_k}")
		logger.info(f"  - Final results: top-{final_k}")
    
	def retrieve_and_rerank(self,
						   query: str,
						   metadata_filter: Optional[Dict[str, Any]] = None,
						   score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
		"""
		Retrieve documents and re-rank them
        
		Args:
			query: Query string
			metadata_filter: Metadata filters
			score_threshold: Minimum score threshold for initial retrieval
            
		Returns:
			Re-ranked documents
		"""
		logger.info(f"Retrieve and rerank for: '{query[:50]}...'")
        
		# Step 1: Initial retrieval
		initial_results = self.retriever.retrieve(
			query=query,
			top_k=self.initial_k,
			score_threshold=score_threshold,
			metadata_filter=metadata_filter
		)
        
		if not initial_results:
			logger.warning("No documents retrieved")
			return []
        
		logger.info(f"Retrieved {len(initial_results)} documents")
        
		# Step 2: Re-rank
		reranked_results = self.reranker.rerank(
			query=query,
			documents=initial_results,
			top_k=self.final_k
		)
        
		return reranked_results


# Example usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
    
	# Initialize reranker
	reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
	# Sample documents
	query = "What is machine learning?"
	documents = [
		{
			"id": "doc1",
			"text": "Machine learning is a subset of artificial intelligence that focuses on teaching computers to learn from data.",
			"score": 0.85
		},
		{
			"id": "doc2",
			"text": "Deep learning is a type of machine learning based on artificial neural networks.",
			"score": 0.80
		},
		{
			"id": "doc3",
			"text": "Python is a popular programming language for data science.",
			"score": 0.75
		}
	]
    
	# Re-rank
	reranked = reranker.rerank(query, documents, top_k=2)
    
	print("\nRe-ranked Results:")
	for doc in reranked:
		print(f"Rank {doc['rerank_position']}: Score={doc['rerank_score']:.4f}")
		print(f"  {doc['text'][:60]}...")
		print()
