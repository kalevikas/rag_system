"""
Hybrid Chunking Strategy
Combines recursive, semantic, and sentence-based chunking
"""
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
import nltk

"""
Hybrid Chunking Strategy
Combines recursive, semantic, and sentence-based chunking
"""
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces"""
        pass


class RecursiveChunker(BaseChunker):
    """Recursive character-based chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 separators: List[str] = None):
        """
        Initialize recursive chunker
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            is_separator_regex=False
        )
        logger.info(f"RecursiveChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents recursively"""
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Recursive chunking: {len(documents)} docs -> {len(chunks)} chunks")
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on embedding similarity"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5",
                 buffer_size: int = 1, breakpoint_threshold: float = 0.5):
        """
        Initialize semantic chunker
        
        Args:
            embedding_model: Model to use for embeddings
            buffer_size: Number of sentences to group
            breakpoint_threshold: Threshold for semantic breaks
        """
        self.model = SentenceTransformer(embedding_model)
        self.buffer_size = buffer_size
        self.breakpoint_threshold = breakpoint_threshold
        logger.info(f"SemanticChunker initialized: model={embedding_model}")
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents semantically"""
        from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
        from langchain.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        splitter = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.breakpoint_threshold
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"Semantic chunking: {len(documents)} docs -> {len(chunks)} chunks")
        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-based chunking"""
    
    def __init__(self, sentences_per_chunk: int = 5, overlap: int = 1):
        """
        Initialize sentence chunker
        
        Args:
            sentences_per_chunk: Number of sentences per chunk
            overlap: Number of overlapping sentences between chunks
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap
        logger.info(f"SentenceChunker initialized: sentences={sentences_per_chunk}")
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents by sentences"""
        chunks = []
        
        for doc in documents:
            # Split into sentences
            sentences = nltk.sent_tokenize(doc.page_content)
            
            # Create chunks
            for i in range(0, len(sentences), self.sentences_per_chunk - self.overlap):
                chunk_sentences = sentences[i:i + self.sentences_per_chunk]
                if not chunk_sentences:
                    continue
                
                chunk_text = " ".join(chunk_sentences)
                
                # Create new document with metadata
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "sentence",
                        "sentence_start": i,
                        "sentence_end": i + len(chunk_sentences)
                    }
                )
                chunks.append(chunk_doc)
        
        logger.info(f"Sentence chunking: {len(documents)} docs -> {len(chunks)} chunks")
        return chunks


class HybridChunker:
    """
    Hybrid chunking strategy combining multiple approaches
    Uses recursive chunking first, then applies semantic boundaries
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hybrid chunker
        
        Args:
            config: Configuration dictionary with chunking parameters
        """
        self.config = config
        
        # Initialize chunkers
        recursive_config = config.get("recursive", {})
        self.recursive_chunker = RecursiveChunker(
            chunk_size=recursive_config.get("chunk_size", 1000),
            chunk_overlap=recursive_config.get("chunk_overlap", 200),
            separators=recursive_config.get("separators", ["\n\n", "\n", ". ", " ", ""])
        )
        
        logger.info("HybridChunker initialized")
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Apply hybrid chunking strategy
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents with metadata
        """
        logger.info(f"Starting hybrid chunking on {len(documents)} documents")
        
        # Step 1: Apply recursive chunking
        chunks = self.recursive_chunker.chunk(documents)
        
        # Step 2: Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_method"] = "hybrid"
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            chunk.metadata["chunk_words"] = len(chunk.page_content.split())
        
        logger.info(f"Hybrid chunking complete: {len(chunks)} total chunks")
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        chunk_words = [len(chunk.page_content.split()) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "avg_words_per_chunk": sum(chunk_words) / len(chunks),
            "total_chars": sum(chunk_sizes),
            "total_words": sum(chunk_words)
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        "recursive": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", ". ", " ", ""]
        }
    }
    
    # Create sample documents
    sample_doc = Document(
        page_content="This is a sample document. " * 100,
        metadata={"source": "test.pdf"}
    )
    
    # Initialize and test chunker
    chunker = HybridChunker(config)
    chunks = chunker.chunk([sample_doc])
    stats = chunker.get_chunk_stats(chunks)
    
    print("\nChunk Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
