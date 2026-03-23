"""
RAG Pipeline Orchestrator - Multi-Company Edition
Each company has its own Qdrant collection; retrieval is scoped to one company.
"""
import os
import yaml
import logging
from typing import Any, Dict, Generator, List, Optional

from langchain_core.documents import Document

from .chunking import HybridChunker
from .embeddings import EmbeddingManager
from .vector_store import QdrantVectorStore
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker, RerankerPipeline
from .llm_handler import LLMHandler, PromptTemplates
from .chat_memory import ConversationMemory
from .company_manager import get_company_manager

logger = logging.getLogger(__name__)


def _load_config() -> Dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


class CompanyRAGPipeline:
    """
    A pipeline scoped to ONE company's Qdrant collection.
    Supports: ingest (any pre-loaded Documents), retrieve, answer.
    """

    def __init__(
        self,
        company_name: str,
        config: Optional[Dict] = None,
    ):
        self.company_name = company_name
        self._cfg = config or _load_config()

        mgr = get_company_manager()
        company_rec = mgr.get_or_create(company_name)
        collection_name = company_rec["collection"]

        vs_cfg = self._cfg["vectorstore"]
        host = vs_cfg.get("host", "localhost")
        port = vs_cfg.get("port", 6333)
        vector_size = vs_cfg.get("vector_size", 1024)
        distance = vs_cfg.get("distance_metric", "Cosine")
        # Support Qdrant Cloud via env vars (for Streamlit Cloud deployment)
        cloud_url = vs_cfg.get("cloud_url") or os.environ.get("QDRANT_URL")
        api_key_qdrant = vs_cfg.get("api_key") or os.environ.get("QDRANT_API_KEY")
        use_cloud = vs_cfg.get("use_cloud", False) or bool(cloud_url)

        em_cfg = self._cfg["embedding"]
        self.embedding_manager = EmbeddingManager(
            model_name=em_cfg.get("model_name", "BAAI/bge-large-en-v1.5"),
            batch_size=em_cfg.get("batch_size", 32),
            normalize=em_cfg.get("normalize", True),
        )

        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            host=host,
            port=port,
            vector_size=vector_size,
            distance_metric=distance,
            use_cloud=use_cloud,
            cloud_url=cloud_url,
            api_key=api_key_qdrant,
        )

        chunking_cfg = self._cfg.get("chunking", {})
        self.chunker = HybridChunker(chunking_cfg)

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            documents=[],
            dense_weight=self._cfg.get("hybrid_search", {}).get("dense_weight", 0.7),
            sparse_weight=self._cfg.get("hybrid_search", {}).get("sparse_weight", 0.3),
        )

        reranker_cfg = self._cfg.get("reranking", {})
        self.reranker = RerankerPipeline(
            retriever=self.retriever,
            reranker=Reranker(reranker_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
            initial_k=20,
            final_k=reranker_cfg.get("top_k", 5),
        )

        llm_cfg = self._cfg.get("llm", {})
        openai_key = self._cfg.get("openai", {}).get("api_key")
        self.llm = LLMHandler(
            api_key=openai_key,
            model=llm_cfg.get("model", "gpt-4o-mini"),
        )
        self.memory = ConversationMemory(max_turns=20)
        logger.info(
            f"[CompanyRAGPipeline] Ready - company='{company_name}' "
            f"collection='{collection_name}'"
        )

    def ingest_documents(
        self,
        documents: List[Document],
        source_type: str = "unknown",
        extra_metadata: Optional[Dict] = None,
    ) -> int:
        """Chunk, embed, and store pre-loaded Documents. Returns chunk count."""
        if not documents:
            return 0
        for doc in documents:
            doc.metadata["company"] = self.company_name
            doc.metadata["source_type"] = source_type
            if extra_metadata:
                doc.metadata.update(extra_metadata)

        chunks = self.chunker.chunk(documents)
        if not chunks:
            return 0

        embeddings = self.embedding_manager.embed_documents(chunks)
        self.vector_store.add_documents(chunks, embeddings)

        try:
            self.retriever.update_documents(chunks)
        except Exception:
            pass

        mgr = get_company_manager()
        mgr.update_doc_count(self.company_name, len(chunks), source_type)

        logger.info(
            f"[{self.company_name}] Ingested {len(chunks)} chunks "
            f"(from {len(documents)} docs, type={source_type})"
        )
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid retrieval + cross-encoder reranking."""
        results = self.reranker.retrieve_and_rerank(query)
        logger.info(f"[{self.company_name}] Retrieved {len(results)} docs")
        return results

    _FALLBACK = (
        "❌ **Not Found in Documents**\n\n"
        "I couldn't find relevant information about this in the uploaded documents. "
        "Please make sure the relevant documents have been uploaded and indexed for this user."
    )

    @staticmethod
    def _best_source_label(doc: Dict[str, Any]) -> str:
        """Return the most user-friendly source label from a retrieval result."""
        meta = doc.get("metadata", {}) or {}
        raw_source = (doc.get("source") or meta.get("source") or "").strip()
        file_name = (meta.get("file_name") or "").strip()
        scraped_url = (meta.get("scraped_url") or "").strip()
        api_url = (meta.get("api_url") or "").strip()

        candidate = raw_source
        if candidate and candidate not in ("unknown_source", "unknown_file"):
            candidate = os.path.basename(candidate)
            if candidate.lower().startswith("tmp") and file_name:
                candidate = file_name
            return candidate

        if file_name:
            return file_name
        if scraped_url:
            return scraped_url
        if api_url:
            return api_url
        return "document"

    def answer(self, query: str) -> Dict[str, Any]:
        """Full RAG: retrieve → rerank → LLM. Returns dict with answer and metadata."""
        docs = self.retrieve(query)

        # ── Fallback: no relevant documents found ──────────────────────────────
        if not docs:
            self.memory.add_message("user", query)
            self.memory.add_message("assistant", self._FALLBACK)
            return {
                "answer": self._FALLBACK,
                "sources": [],
                "company": self.company_name,
                "chunks_used": 0,
                "not_found": True,
            }

        context_parts = []
        sources = []
        for doc in docs:
            text = doc.get("text", "")
            src_label = self._best_source_label(doc)
            if text:
                context_parts.append(f"Excerpt Source: {src_label}\n{text}")
            if src_label not in sources and src_label != "document":
                sources.append(src_label)

        context = "\n\n---\n\n".join(context_parts)
        history = self.memory.get_history_summary()
        self.memory.add_message("user", query)

        system_msg, user_prompt = PromptTemplates.rag_answer_prompt(context, query, history)
        answer_text = self.llm.generate(user_prompt, system_message=system_msg, max_tokens=2048)
        self.memory.add_message("assistant", answer_text)

        return {
            "answer": answer_text,
            "sources": sources,
            "company": self.company_name,
            "chunks_used": len(docs),
        }

    def stream_answer(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming RAG: retrieve → rerank → stream LLM tokens one by one.

        Yields dicts:
          {"token": "<text chunk>"}                               — streamed piece
          {"done": True, "sources": [...], "chunks_used": N,      — final sentinel
           "not_found": bool}
        """
        docs = self.retrieve(query)

        # ── Fallback: no docs found ────────────────────────────────────────────
        if not docs:
            self.memory.add_message("user", query)
            self.memory.add_message("assistant", self._FALLBACK)
            yield {"token": self._FALLBACK}
            yield {"done": True, "sources": [], "chunks_used": 0, "not_found": True}
            return

        context_parts = []
        sources = []
        for doc in docs:
            text = doc.get("text", "")
            src_label = self._best_source_label(doc)
            if text:
                context_parts.append(f"Excerpt Source: {src_label}\n{text}")
            if src_label not in sources and src_label != "document":
                sources.append(src_label)

        context = "\n\n---\n\n".join(context_parts)
        history = self.memory.get_history_summary()
        self.memory.add_message("user", query)

        system_msg, user_prompt = PromptTemplates.rag_answer_prompt(context, query, history)

        full_response: List[str] = []
        for chunk in self.llm.generate_stream(user_prompt, system_message=system_msg, max_tokens=2048):
            full_response.append(chunk)
            yield {"token": chunk}

        full_text = "".join(full_response)
        self.memory.add_message("assistant", full_text)
        yield {"done": True, "sources": sources, "chunks_used": len(docs), "not_found": False}

    def clear_memory(self):
        self.memory.clear()


# Pipeline pool - reuse per-company instances
_pipeline_pool: Dict[str, CompanyRAGPipeline] = {}
_shared_config: Optional[Dict] = None


def _get_config() -> Dict:
    global _shared_config
    if _shared_config is None:
        _shared_config = _load_config()
    return _shared_config


def get_pipeline(company_name: str) -> CompanyRAGPipeline:
    """Return (or create) a CompanyRAGPipeline for the given company."""
    key = company_name.strip()
    if key not in _pipeline_pool:
        logger.info(f"[PipelinePool] Creating pipeline for company='{key}'")
        _pipeline_pool[key] = CompanyRAGPipeline(key, config=_get_config())
    return _pipeline_pool[key]


def invalidate_pipeline(company_name: str):
    """Force re-creation of a pipeline (e.g. after collection reset)."""
    _pipeline_pool.pop(company_name.strip(), None)


# Backward-compatible alias
class RAGPipeline(CompanyRAGPipeline):
    """Legacy shim that defaults to the collection_name in config.yaml."""

    def __init__(self, **kwargs):
        cfg = _get_config()
        default_collection = cfg["vectorstore"].get("collection_name", "default")
        super().__init__(company_name=default_collection, config=cfg)

    def ingest_pdf(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        from .multi_format_processor import load_pdf
        docs = load_pdf(pdf_path)
        if metadata:
            for d in docs:
                d.metadata.update(metadata)
        return self.ingest_documents(docs, source_type="pdf")
