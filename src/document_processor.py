"""
PDF Document Processor with Rich Metadata Extraction
Supports PyPDF and PyMuPDF for robust text extraction
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document
import pymupdf

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents with metadata extraction"""
    def __init__(self, use_pymupdf: bool = True):
        self.use_pymupdf = use_pymupdf
        logger.info(f"PDFProcessor initialized with {'PyMuPDF' if use_pymupdf else 'PyPDF'}")

    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        try:
            doc = pymupdf.open(pdf_path)
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": doc.page_count,
                "file_size": os.path.getsize(pdf_path),
                "file_name": Path(pdf_path).name,
                "file_path": str(Path(pdf_path).resolve()),
                "processed_date": datetime.now().isoformat()
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                "file_name": Path(pdf_path).name,
                "file_path": str(Path(pdf_path).resolve()),
                "processed_date": datetime.now().isoformat(),
                "error": str(e)
            }

    def load_single_pdf(self, pdf_path: str) -> List[Document]:
        logger.info(f"Loading PDF: {pdf_path}")
        try:
            file_metadata = self.extract_metadata(pdf_path)
            if self.use_pymupdf:
                loader = PyMuPDFLoader(str(pdf_path))
            else:
                loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            for i, doc in enumerate(documents):
                doc.metadata.update(file_metadata)
                doc.metadata["page_number"] = i + 1
                doc.metadata["source"] = pdf_path
                doc.metadata["doc_type"] = "pdf"
                doc.metadata["char_count"] = len(doc.page_content)
                doc.metadata["word_count"] = len(doc.page_content.split())
            logger.info(f"Loaded {len(documents)} pages from {Path(pdf_path).name}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []

    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        pdf_dir = Path(directory)
        if not pdf_dir.exists():
            logger.error(f"Directory does not exist: {directory}")
            return []
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(pdf_dir.glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        all_documents = []
        for pdf_file in pdf_files:
            documents = self.load_single_pdf(str(pdf_file))
            all_documents.extend(documents)
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {"total_documents": 0}
        total_chars = sum(doc.metadata.get("char_count", 0) for doc in documents)
        total_words = sum(doc.metadata.get("word_count", 0) for doc in documents)
        unique_files = set(doc.metadata.get("file_name", "") for doc in documents)
        stats = {
            "total_documents": len(documents),
            "total_pages": len(documents),
            "unique_files": len(unique_files),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_page": total_chars / len(documents) if documents else 0,
            "avg_words_per_page": total_words / len(documents) if documents else 0,
            "file_names": list(unique_files)
        }
        return stats
