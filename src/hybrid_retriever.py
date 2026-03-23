
"""
Hybrid Retriever: Combines Dense (Vector) + Sparse (BM25) Search
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BM25Retriever:
	"""BM25 sparse retrieval for keyword matching"""
    
	def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
		"""
		Initialize BM25 retriever
        
		Args:
			documents: List of documents to index
			k1: BM25 k1 parameter (term frequency saturation)
			b: BM25 b parameter (length normalization)
		"""
		self.documents = documents
		self.k1 = k1
		self.b = b
        
		# Tokenize documents
		self.tokenized_docs = [doc.page_content.lower().split() for doc in documents]
		
		# Initialize BM25 only if we have documents
		if len(self.tokenized_docs) > 0:
			self.bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)
		else:
			self.bm25 = None
			logger.warning("BM25Retriever initialized with 0 documents - BM25 disabled until documents are added")
        
		logger.info(f"BM25Retriever initialized with {len(documents)} documents")
    
	def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		"""
		Search using BM25
        
		Args:
			query: Query string
			top_k: Number of results to return
            
		Returns:
			List of results with scores
		"""
		# Tokenize query
		tokenized_query = query.lower().split()
		# If BM25 is not initialized (no documents), return empty results
		if self.bm25 is None:
			logger.debug("BM25 search requested but BM25 index is empty")
			return []

		# Get BM25 scores
		scores = self.bm25.get_scores(tokenized_query)
        
		# Get top-k indices
		top_indices = np.argsort(scores)[::-1][:top_k]
        
		# Format results
		results = []
		for idx in top_indices:
			if scores[idx] > 0:  # Only include non-zero scores
				results.append({
					"index": int(idx),
					"score": float(scores[idx]),
					"text": self.documents[idx].page_content,
					"metadata": self.documents[idx].metadata,
					"method": "bm25"
				})
        
		logger.debug(f"BM25 found {len(results)} results")
		return results
    
	def update_documents(self, documents: List[Document]):
		"""Update the document index"""
		self.documents = documents
		self.tokenized_docs = [doc.page_content.lower().split() for doc in documents]
		if len(self.tokenized_docs) > 0:
			self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
		else:
			self.bm25 = None
			logger.warning("BM25 index updated with 0 documents - BM25 disabled until documents are added")
		logger.info(f"BM25 index updated with {len(documents)} documents")


class HybridRetriever:
	"""
	Hybrid retriever combining dense vector search and sparse BM25 search
	"""
    
	def __init__(self,
				 vector_store,
				 embedding_manager,
				 documents: List[Document],
				 dense_weight: float = 0.7,
				 sparse_weight: float = 0.3,
				 bm25_k1: float = 1.5,
				 bm25_b: float = 0.75):
		"""
		Initialize hybrid retriever
        
		Args:
			vector_store: Vector store for dense retrieval
			embedding_manager: Embedding manager for query encoding
			documents: List of documents (for BM25 indexing)
			dense_weight: Weight for dense retrieval scores
			sparse_weight: Weight for sparse retrieval scores
			bm25_k1: BM25 k1 parameter
			bm25_b: BM25 b parameter
		"""
		self.vector_store = vector_store
		self.embedding_manager = embedding_manager
		self.documents = documents
		self.dense_weight = dense_weight
		self.sparse_weight = sparse_weight
        
		# Initialize BM25
		self.bm25_retriever = BM25Retriever(
			documents=documents,
			k1=bm25_k1,
			b=bm25_b
		)
        
		# Validate weights
		if not np.isclose(dense_weight + sparse_weight, 1.0):
			logger.warning(f"Weights don't sum to 1.0: {dense_weight} + {sparse_weight}")
        
		logger.info(f"HybridRetriever initialized")
		logger.info(f"  - Dense weight: {dense_weight}")
		logger.info(f"  - Sparse weight: {sparse_weight}")
		logger.info(f"  - Total documents: {len(documents)}")
    
	def retrieve(self,
				 query: str,
				 top_k: int = 10,
				 score_threshold: Optional[float] = None,
				 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""
		Hybrid retrieval combining dense and sparse search
        
		Args:
			query: Query string
			top_k: Number of results to return
			score_threshold: Minimum score threshold
			metadata_filter: Metadata filters for vector search
            
		Returns:
			List of retrieved documents with scores
		"""
		logger.info(f"Hybrid retrieval for query: '{query[:50]}...'")
        
		# 1. Dense retrieval (vector search)
		query_embedding = self.embedding_manager.embed_query(query)
		dense_results = self.vector_store.search(
			query_embedding=query_embedding,
			top_k=top_k * 2,  # Get more candidates for fusion
			score_threshold=None,  # Apply threshold after fusion
			metadata_filter=metadata_filter
		)
		# logger.info(f"[DEBUG] Dense results: {dense_results}")
        
		# 2. Sparse retrieval (BM25)
		sparse_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        
		# 3. Normalize and combine scores
		combined_results = self._fusion_scores(
			dense_results, 
			sparse_results, 
			top_k=top_k
		)
        
		# 4. Apply score threshold if provided
		if score_threshold is not None:
			combined_results = [
				r for r in combined_results 
				if r["combined_score"] >= score_threshold
			]
        
		logger.info(f"Retrieved {len(combined_results)} results after fusion")
        
		return combined_results
    
	def _fusion_scores(self,
					  dense_results: List[Dict[str, Any]],
					  sparse_results: List[Dict[str, Any]],
					  top_k: int) -> List[Dict[str, Any]]:
		"""
		Reciprocal Rank Fusion (RRF) — correctly combines dense and sparse results.

		The previous implementation computed sparse contributions but never applied
		them (values were assigned to a discarded local variable). RRF sidesteps
		score-normalisation entirely by using only rank positions.

		Formula: score(d) = dense_weight/(K+r_dense) + sparse_weight/(K+r_sparse)
		where K=60 is the standard RRF constant.
		"""
		K = 60  # standard RRF constant

		scores: Dict[str, float] = {}   # text -> combined RRF score
		doc_map: Dict[str, Any] = {}    # text -> result dict (best representative)

		# Dense results
		for rank, result in enumerate(dense_results):
			text = result.get("text", "")
			if not text:
				continue
			rrf = self.dense_weight * (1.0 / (K + rank + 1))
			scores[text] = scores.get(text, 0.0) + rrf
			if text not in doc_map:
				doc_map[text] = result

		# Sparse (BM25) results — uses text as the common key, no ID mapping needed
		for rank, result in enumerate(sparse_results):
			text = result.get("text", "")
			if not text:
				continue
			rrf = self.sparse_weight * (1.0 / (K + rank + 1))
			scores[text] = scores.get(text, 0.0) + rrf
			if text not in doc_map:
				doc_map[text] = result

		# Sort by fused score and return top-k
		sorted_texts = sorted(scores, key=lambda t: scores[t], reverse=True)[:top_k]
		results = []
		for text in sorted_texts:
			doc = dict(doc_map[text])
			doc["combined_score"] = scores[text]
			results.append(doc)

		logger.debug(
			f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
			f"→ {len(results)} merged (top-{top_k})"
		)
		return results
    
	def update_documents(self, documents: List[Document]):
		"""Update document index (mainly for BM25)"""
		self.documents = documents
		self.bm25_retriever.update_documents(documents)
		logger.info(f"Hybrid retriever updated with {len(documents)} documents")


# Example usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
    
	# This is a simplified example
	# In practice, you'd initialize with actual vector store and embedding manager
    
	sample_docs = [
		Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"id": "doc1"}),
		Document(page_content="Deep learning uses neural networks with multiple layers.", metadata={"id": "doc2"}),
		Document(page_content="Natural language processing enables computers to understand human language.", metadata={"id": "doc3"})
	]
    
	bm25 = BM25Retriever(sample_docs)
	results = bm25.search("machine learning neural networks", top_k=2)
    
	print("\nBM25 Results:")
	for r in results:
		print(f"Score: {r['score']:.4f} - {r['text'][:50]}...")
