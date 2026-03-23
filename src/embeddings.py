
"""
Embedding Pipeline using BGE-Large for high-quality embeddings
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class EmbeddingManager:
	"""Manages document embedding generation using state-of-the-art models"""
    
	def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", 
				 batch_size: int = 32, normalize: bool = True):
		"""
		Initialize embedding manager
        
		Args:
			model_name: HuggingFace model name (default: BGE-large for high accuracy)
			batch_size: Batch size for encoding
			normalize: Whether to normalize embeddings
		"""
		self.model_name = model_name
		self.batch_size = batch_size
		self.normalize = normalize
		self.model = None
		self._load_model()
    
	def _load_model(self):
		"""Load the sentence transformer model"""
		try:
			logger.info(f"Loading embedding model: {self.model_name}")
			self.model = SentenceTransformer(self.model_name)
            
			# Get model info
			self.embedding_dimension = self.model.get_sentence_embedding_dimension()
			self.max_seq_length = self.model.max_seq_length
            
			logger.info(f"Model loaded successfully")
			logger.info(f"  - Embedding dimension: {self.embedding_dimension}")
			logger.info(f"  - Max sequence length: {self.max_seq_length}")
            
		except Exception as e:
			logger.error(f"Error loading model {self.model_name}: {e}")
			raise
    
	def embed_documents(self, documents: List[Document]) -> np.ndarray:
		"""
		Generate embeddings for a list of documents
        
		Args:
			documents: List of Document objects
            
		Returns:
			numpy array of embeddings with shape (len(documents), embedding_dim)
		"""
		if not self.model:
			raise ValueError("Model not loaded")
        
		# Extract text from documents
		texts = [doc.page_content for doc in documents]
        
		return self.embed_texts(texts)
    
	def embed_texts(self, texts: List[str]) -> np.ndarray:
		"""
		Generate embeddings for a list of texts
        
		Args:
			texts: List of text strings
            
		Returns:
			numpy array of embeddings
		"""
		if not self.model:
			raise ValueError("Model not loaded")
        
		logger.info(f"Generating embeddings for {len(texts)} texts...")
        
		# For BGE models, add instruction prefix for better retrieval
		if "bge" in self.model_name.lower():
			# Add instruction for passage embedding
			texts = [f"Represent this document for retrieval: {text}" for text in texts]
        
		# Generate embeddings
		embeddings = self.model.encode(
			texts,
			batch_size=self.batch_size,
			show_progress_bar=True,
			normalize_embeddings=self.normalize,
			convert_to_numpy=True
		)
        
		logger.info(f"Generated embeddings with shape: {embeddings.shape}")
		return embeddings
    
	def embed_query(self, query: str) -> np.ndarray:
		"""
		Generate embedding for a single query
        
		Args:
			query: Query string
            
		Returns:
			numpy array of query embedding
		"""
		if not self.model:
			raise ValueError("Model not loaded")
        
		# For BGE models, add instruction prefix for query
		if "bge" in self.model_name.lower():
			query = f"Represent this query for retrieving relevant documents: {query}"
        
		embedding = self.model.encode(
			query,
			normalize_embeddings=self.normalize,
			convert_to_numpy=True
		)
        
		return embedding
    
	def get_embedding_dimension(self) -> int:
		"""Get the dimension of the embeddings"""
		return self.embedding_dimension
    
	def batch_embed(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
		"""
		Embed texts in batches (useful for large datasets)
        
		Args:
			texts: List of texts to embed
			batch_size: Optional custom batch size
            
		Returns:
			numpy array of embeddings
		"""
		batch_size = batch_size or self.batch_size
        
		all_embeddings = []
        
		for i in range(0, len(texts), batch_size):
			batch = texts[i:i + batch_size]
			logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
			embeddings = self.embed_texts(batch)
			all_embeddings.append(embeddings)
        
		return np.vstack(all_embeddings)


class EmbeddingCache:
	"""Cache for embeddings to avoid recomputation"""
    
	def __init__(self, max_size: int = 10000):
		"""
		Initialize embedding cache
        
		Args:
			max_size: Maximum number of embeddings to cache
		"""
		self.cache: Dict[str, np.ndarray] = {}
		self.max_size = max_size
		logger.info(f"EmbeddingCache initialized with max_size={max_size}")
    
	def get(self, text: str) -> Optional[np.ndarray]:
		"""Get embedding from cache"""
		return self.cache.get(text)
    
	def put(self, text: str, embedding: np.ndarray):
		"""Put embedding in cache"""
		if len(self.cache) >= self.max_size:
			# Remove oldest entry (simple FIFO)
			self.cache.pop(next(iter(self.cache)))
        
		self.cache[text] = embedding
    
	def clear(self):
		"""Clear the cache"""
		self.cache.clear()
		logger.info("Embedding cache cleared")
    
	def size(self) -> int:
		"""Get current cache size"""
		return len(self.cache)


# Example usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
    
	# Initialize embedding manager
	embedding_manager = EmbeddingManager(
		model_name="BAAI/bge-large-en-v1.5",
		batch_size=32,
		normalize=True
	)
    
	# Create sample documents
	sample_texts = [
		"This is a sample document about machine learning.",
		"Natural language processing is a subfield of AI.",
		"Vector embeddings capture semantic meaning."
	]
    
	# Generate embeddings
	embeddings = embedding_manager.embed_texts(sample_texts)
	print(f"\nGenerated embeddings shape: {embeddings.shape}")
    
	# Generate query embedding
	query = "What is machine learning?"
	query_embedding = embedding_manager.embed_query(query)
	print(f"Query embedding shape: {query_embedding.shape}")
