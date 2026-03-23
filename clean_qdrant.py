"""
Script to clean (delete all points from) Qdrant collection.
Usage: python clean_qdrant.py
"""
import logging
from src.vector_store import QdrantVectorStore

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Update these parameters if needed
    COLLECTION_NAME = "pdf_documents"  # or your actual collection name
    HOST = "localhost"
    PORT = 6333
    VECTOR_SIZE = 1024

    vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        host=HOST,
        port=PORT,
        vector_size=VECTOR_SIZE
    )
    print(f"Cleaning Qdrant collection: {COLLECTION_NAME}")
    vector_store.clear_collection()
    print("Qdrant collection cleaned (all points deleted).")
