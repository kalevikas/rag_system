
"""
Qdrant Vector Store with Metadata Filtering Support
"""
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range, SearchParams
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Manages document embeddings in Qdrant vector database"""

    def clear_collection(self):
        """
        Delete all points in the collection (full reset, keeps collection schema)
        """
        try:
            # Use Filter(must=[]) to match all points (canonical Qdrant way)
            from qdrant_client.models import Filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[])
            )
            logger.info(f"All points deleted from collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def __init__(self, 
                 collection_name: str = "pdf_documents",
                 host: str = "localhost",
                 port: int = 6333,
                 vector_size: int = 1024,
                 distance_metric: str = "Cosine",
                 use_cloud: bool = False,
                 cloud_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        Initialize Qdrant vector store
        
        Args:
            collection_name: Name of the collection
            host: Qdrant host (for local deployment)
            port: Qdrant port
            vector_size: Dimension of embeddings
            distance_metric: Distance metric (Cosine, Euclid, Dot)
            use_cloud: Whether to use Qdrant Cloud
            cloud_url: Qdrant Cloud URL
            api_key: API key for Qdrant Cloud
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize client
        if use_cloud and cloud_url and api_key:
            logger.info(f"Connecting to Qdrant Cloud: {cloud_url}")
            self.client = QdrantClient(url=cloud_url, api_key=api_key)
        else:
            logger.info(f"Connecting to local Qdrant: {host}:{port}")
            self.client = QdrantClient(host=host, port=port)
        
        # Set distance metric
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        self.distance_metric = distance_map.get(distance_metric, Distance.COSINE)
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                )
                logger.info(f"Collection created successfully")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
            # Get collection info
            info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection info: {info.points_count} points")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[Document], 
                     embeddings: np.ndarray,
                     batch_size: int = 100) -> List[str]:
        # Debug: Print first few chunks' content and metadata
        # for idx, doc in enumerate(documents[:3]):
        #     logger.info(f"[DEBUG] Chunk {idx} page_content: {getattr(doc, 'page_content', None)[:100] if getattr(doc, 'page_content', None) else None}")
        #     logger.info(f"[DEBUG] Chunk {idx} metadata: {getattr(doc, 'metadata', None)}")
        """
        Add documents with embeddings to Qdrant
        
        Args:
            documents: List of Document objects
            embeddings: Corresponding embeddings
            batch_size: Batch size for upload
            
        Returns:
            List of point IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        logger.info(f"Adding {len(documents)} documents to Qdrant...")
        
        points = []
        point_ids = []
        

        import uuid
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID (UUID)
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            # Ensure metadata fields are present
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            payload = {
                "text": getattr(doc, "page_content", ""),
                "metadata": meta,
                "source": meta.get("source") or meta.get("file_name") or "unknown_source",
                "file_name": meta.get("file_name") or "unknown_file",
                "page_number": meta.get("page_number") if meta.get("page_number") is not None else 0,
                "chunk_id": meta.get("chunk_id", i),
                "doc_type": meta.get("doc_type", "pdf"),
                "created_at": meta.get("processed_date", datetime.now().isoformat())
            }

            # logger.info(f"[DEBUG] Upserting point_id={point_id} payload={payload}")

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

            # Upload in batches
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Uploaded batch of {len(points)} points")
                points = []
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Uploaded final batch of {len(points)} points")
        
        logger.info(f"Successfully added {len(documents)} documents")
        return point_ids
    
    def search(self,
               query_embedding: np.ndarray,
               top_k: int = 5,
               score_threshold: Optional[float] = None,
               metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            metadata_filter: Dictionary of metadata filters
            
        Returns:
            List of search results with documents and scores
        """
        # Build filter if provided
        filter_query = None
        if metadata_filter:
            filter_query = self._build_filter(metadata_filter)
        
        # Search
        try:
            # Use the QdrantClient.query_points method for vector search (latest Qdrant client)

            # logger.info(f"[DEBUG] Qdrant query_points: collection={self.collection_name}, query={query_embedding.tolist()}, filter={filter_query}, top_k={top_k}, score_threshold={score_threshold}")
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                query_filter=filter_query,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold
            )

            # logger.info(f"[DEBUG] Raw Qdrant results type: {type(results)}")
            # logger.info(f"[DEBUG] Raw Qdrant results: {results}")
            
            # Extract scored points - handle different return types
            scored_points = []
            if hasattr(results, 'points'):
                # QueryResponse object with .points attribute
                scored_points = results.points
                # logger.info(f"[DEBUG] Extracted points from .points attribute")
            elif isinstance(results, tuple) and len(results) == 2 and results[0] == 'points':
                # Tuple format ('points', [ScoredPoint, ...])
                scored_points = results[1]
                # logger.info(f"[DEBUG] Extracted points from tuple")
            elif isinstance(results, list):
                # Direct list of ScoredPoint objects
                scored_points = results
                # logger.info(f"[DEBUG] Using results as list directly")
            else:
                # Fallback: try to iterate
                scored_points = results
                # logger.info(f"[DEBUG] Using results directly (unknown type)")

            # logger.info(f"[DEBUG] Number of scored_points: {len(scored_points) if hasattr(scored_points, '__len__') else 'unknown'}")

            formatted_results = []
            for idx, result in enumerate(scored_points):
                # logger.info(f"[DEBUG] Raw Qdrant result #{idx}: {result}")
                # For ScoredPoint objects, use attribute access
                payload = getattr(result, 'payload', {})
                # logger.info(f"[DEBUG] Parsed payload for result #{idx}: {payload}")
                formatted_results.append({
                    "id": getattr(result, 'id', None),
                    "score": getattr(result, 'score', None),
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                    "source": payload.get("source", ""),
                    "page_number": payload.get("page_number", 0)
                })

            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []
    
    def _build_filter(self, metadata_filter: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from metadata dictionary
        
        Args:
            metadata_filter: Dictionary with filter conditions
            
        Returns:
            Qdrant Filter object
        """
        conditions = []
        
        for key, value in metadata_filter.items():
            if isinstance(value, str):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, (int, float)):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, dict) and "gte" in value:
                # Range query
                conditions.append(
                    FieldCondition(key=key, range=Range(gte=value["gte"], lte=value.get("lte")))
                )
        
        return Filter(must=conditions) if conditions else None
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vector_size": self.vector_size,
                "distance_metric": str(self.distance_metric)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize vector store
    vector_store = QdrantVectorStore(
        collection_name="test_collection",
        host="localhost",
        port=6333,
        vector_size=1024
    )
    
    # Get collection info
    info = vector_store.get_collection_info()
    print("\nCollection Info:")
    for key, value in info.items():
        print(f"{key}: {value}")
