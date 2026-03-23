"""
Multi-Company RAG Web Application
- Each company has its own Qdrant vector collection (index)
- Supports: PDF, Excel, CSV, Word, Text/Markdown, URL, REST API
- Frontend lets users select / create a company before ingestion
"""
import os
import sys
import json
import yaml
import hashlib
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context

# ── Path setup ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from config.config import get_config
cfg = get_config()

from src.utils import setup_logging
from src.rag_pipeline import get_pipeline, invalidate_pipeline
from src.company_manager import get_company_manager
from src.multi_format_processor import (
    load_file, load_url, load_api, get_supported_extensions
)

# ── Logging ─────────────────────────────────────────────────────────────────
log_dir = cfg.get_logs_dir()
os.makedirs(log_dir, exist_ok=True)
setup_logging(log_level=cfg.log_level, log_file=os.path.join(log_dir, "web.log"))
logger = logging.getLogger(__name__)

# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))

UPLOAD_FOLDER = cfg.get_pdf_dir()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

ALLOWED_EXTENSIONS = {
    "pdf", "xlsx", "xls", "csv", "docx", "doc", "txt", "md", "markdown"
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    p = os.path.join(cfg.get_data_dir(), "ingest_manifest.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(m: dict):
    os.makedirs(cfg.get_data_dir(), exist_ok=True)
    p = os.path.join(cfg.get_data_dir(), "ingest_manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)


# ── Company routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", user="Admin", role="Engineering")


@app.route("/api/companies", methods=["GET"])
def list_companies():
    """Return all registered companies."""
    mgr = get_company_manager()
    return jsonify({"companies": mgr.list_companies()})


@app.route("/api/companies", methods=["POST"])
def create_company():
    """Create (or get) a company entry."""
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Company name is required"}), 400
    mgr = get_company_manager()
    rec = mgr.get_or_create(name)
    return jsonify({"company": name, **rec})


@app.route("/api/companies/<company_name>", methods=["DELETE"])
def delete_company(company_name: str):
    """Delete a company and drop its Qdrant collection."""
    mgr = get_company_manager()
    rec = mgr.get(company_name)
    if not rec:
        return jsonify({"error": "Company not found"}), 404
    # Drop the Qdrant collection
    try:
        from src.vector_store import QdrantVectorStore
        vs = QdrantVectorStore(
            collection_name=rec["collection"],
            **{k: v for k, v in cfg.vectorstore.items() if k != "collection_name"}
        )
        vs.delete_collection()
    except Exception as e:
        logger.warning(f"Could not drop collection: {e}")
    mgr.delete(company_name)
    invalidate_pipeline(company_name)
    return jsonify({"success": True})


# ── File management routes ──────────────────────────────────────────────────

@app.route("/files")
def files():
    """List uploaded files (all companies, or filtered by company query param)."""
    company = request.args.get("company", "")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    files_list = []
    for fname in sorted(os.listdir(UPLOAD_FOLDER)):
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext in ALLOWED_EXTENSIONS:
            path = os.path.join(UPLOAD_FOLDER, fname)
            size = os.path.getsize(path)
            files_list.append({
                "name": fname,
                "size": f"{size / 1024:.1f} KB",
                "ext": ext
            })
    return jsonify({"files": files_list})


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/files/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404


@app.route("/files-all", methods=["DELETE"])
def delete_all_files():
    """Delete ALL ingested files from the upload folder."""
    deleted = 0
    errors = 0
    for fname in list(os.listdir(UPLOAD_FOLDER)):
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext in ALLOWED_EXTENSIONS:
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, fname))
                deleted += 1
            except Exception:
                errors += 1
    return jsonify({"success": True, "deleted": deleted, "errors": errors})


# ── Ingest: File upload (any supported format) ──────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload a file and ingest it into the specified company's collection.
    Form fields:
        file     - the file itself
        company  - company name (required)
    """
    try:
        company = (request.form.get("company") or "").strip()
        if not company:
            return jsonify({"error": "Company name is required"}), 400

        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        ext = filename.rsplit(".", 1)[-1].lower()
        logger.info(f"Ingesting {filename} (type={ext}) -> company='{company}'")

        # Load documents using multi-format processor
        documents = load_file(filepath)
        if not documents:
            return jsonify({"error": "Could not extract content from file"}), 400

        # Ingest into company pipeline
        pipeline = get_pipeline(company)
        chunks = pipeline.ingest_documents(documents, source_type=ext,
                                           extra_metadata={"file_name": filename})

        # Update manifest
        try:
            manifest = load_manifest()
            manifest[os.path.abspath(filepath)] = hash_file(filepath)
            save_manifest(manifest)
        except Exception:
            pass

        logger.info(f"[OK] {filename} -> {company}: {chunks} chunks")
        return jsonify({
            "success": True,
            "filename": filename,
            "company": company,
            "doc_type": ext,
            "documents_loaded": len(documents),
            "chunks_created": chunks
        })

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Ingest: URL scraping ─────────────────────────────────────────────────────

@app.route("/scrape", methods=["POST"])
def scrape():
    """
    Scrape a URL and ingest into a company's collection.
    JSON body: { "company": "...", "url": "...", "mode": "auto|advanced|fallback", "depth": 2 }
    """
    try:
        data = request.get_json() or {}
        company = (data.get("company") or "").strip()
        url = (data.get("url") or "").strip()
        mode = data.get("mode", "auto")
        depth = int(data.get("depth", 1))

        if not company:
            return jsonify({"error": "Company name is required"}), 400
        if not url:
            return jsonify({"error": "URL is required"}), 400

        logger.info(f"Scraping {url} (mode={mode}, depth={depth}) -> company='{company}'")

        # WordPress REST API shortcut
        if "/wp-json/" in url and "/posts" in url:
            return _handle_wordpress_api(url, company)

        # Selenium advanced scraper
        documents = []
        if mode in ("auto", "advanced"):
            try:
                from src.web_scraper import WebScraper
                scraper = WebScraper(headless=False, follow_links=(depth > 1), max_depth=depth)
                documents = scraper.scrape_urls([url], max_pages=30) or []
                logger.info(f"Advanced scraper: {len(documents)} docs")
            except Exception as e:
                logger.warning(f"Advanced scraper failed: {e}")
                documents = []

        # Fallback: requests + BeautifulSoup
        if not documents:
            documents = load_url(url, use_selenium=False)

        if not documents:
            return jsonify({"error": "Could not extract any content from the URL"}), 400

        pipeline = get_pipeline(company)
        chunks = pipeline.ingest_documents(documents, source_type="url",
                                           extra_metadata={"scraped_url": url})

        logger.info(f"[OK] {url} -> {company}: {chunks} chunks, {len(documents)} docs")
        return jsonify({
            "success": True,
            "company": company,
            "url": url,
            "total_documents": len(documents),
            "chunks_created": chunks
        })

    except Exception as e:
        logger.error(f"Scrape error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _handle_wordpress_api(url: str, company: str) -> "flask.Response":
    """Extract all posts from a WordPress REST API and ingest."""
    try:
        import requests as req
        from bs4 import BeautifulSoup
        from langchain_core.documents import Document

        resp = req.get(url, params={"per_page": 50}, timeout=30)
        resp.raise_for_status()
        posts = resp.json()

        if not isinstance(posts, list):
            return jsonify({"error": "WordPress API did not return a list"}), 400

        def html2text(html):
            soup = BeautifulSoup(html or "", "html.parser")
            for t in soup(["script","style","nav","footer"]):
                t.decompose()
            import re
            txt = soup.get_text("\n", strip=True)
            return re.sub(r'\n{3,}', '\n\n', txt)

        docs = []
        for post in posts:
            title = html2text(post.get("title", {}).get("rendered", ""))
            content = html2text(post.get("content", {}).get("rendered", ""))
            link = post.get("link", url)
            body = f"### {title}\nURL: {link}\n\n{content}"
            docs.append(Document(page_content=body, metadata={"source": link, "title": title}))

        pipeline = get_pipeline(company)
        chunks = pipeline.ingest_documents(docs, source_type="wordpress", extra_metadata={"api_url": url})

        return jsonify({"success": True, "company": company, "total_documents": len(docs), "chunks_created": chunks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Ingest: REST API data ─────────────────────────────────────────────────────

@app.route("/ingest-api", methods=["POST"])
def ingest_api():
    """
    Fetch data from a REST API endpoint and ingest into a company collection.
    JSON body:
    {
      "company": "ManageEngine",
      "url": "https://api.example.com/data",
      "method": "GET",
      "headers": {},
      "params": {},
      "body": {},
      "json_path": "results"   (optional dot-path into response)
    }
    """
    try:
        data = request.get_json() or {}
        company = (data.get("company") or "").strip()
        url = (data.get("url") or "").strip()

        if not company:
            return jsonify({"error": "Company name is required"}), 400
        if not url:
            return jsonify({"error": "API URL is required"}), 400

        documents = load_api(
            url=url,
            method=data.get("method", "GET"),
            headers=data.get("headers"),
            params=data.get("params"),
            body=data.get("body"),
            json_path=data.get("json_path"),
        )

        if not documents:
            return jsonify({"error": "API returned no usable data"}), 400

        pipeline = get_pipeline(company)
        chunks = pipeline.ingest_documents(documents, source_type="api",
                                           extra_metadata={"api_url": url})

        return jsonify({
            "success": True,
            "company": company,
            "api_url": url,
            "records_fetched": len(documents),
            "chunks_created": chunks
        })

    except Exception as e:
        logger.error(f"API ingest error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Query ────────────────────────────────────────────────────────────────────

@app.route("/ask", methods=["POST"])
def ask():
    """
    Answer a question scoped to a specific company.
    JSON body: { "company": "ManageEngine", "question": "..." }
    """
    try:
        data = request.get_json() or {}
        company = (data.get("company") or "").strip()
        question = (data.get("question") or data.get("query") or "").strip()

        if not company:
            return jsonify({"error": "Company name is required"}), 400
        if not question:
            return jsonify({"error": "Question is required"}), 400

        logger.info(f"[ask] company='{company}' question='{question[:80]}'")
        pipeline = get_pipeline(company)
        result = pipeline.answer(question)

        return jsonify({
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "company": result.get("company", company),
            "chunks_used": result.get("chunks_used", 0)
        })

    except Exception as e:
        logger.error(f"Ask error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/ask-stream", methods=["POST"])
def ask_stream():
    """
    Streaming SSE endpoint — yields tokens as Server-Sent Events.
    JSON body: { "company": "ManageEngine", "question": "..." }
    Each SSE event:
      data: {"token": "<text>"}\n\n
    Final event:
      data: {"done": true, "sources": [...], "chunks_used": N}\n\n
    """
    data = request.get_json() or {}
    company = (data.get("company") or "").strip()
    question = (data.get("question") or data.get("query") or "").strip()

    if not company:
        return jsonify({"error": "Company name is required"}), 400
    if not question:
        return jsonify({"error": "Question is required"}), 400

    logger.info(f"[ask-stream] company='{company}' question='{question[:80]}'")

    def generate():
        try:
            pipeline = get_pipeline(company)
            for event in pipeline.stream_answer(question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/clear-memory", methods=["POST"])
def clear_memory():
    """Clear chat memory for a company."""
    data = request.get_json() or {}
    company = (data.get("company") or "").strip()
    if company:
        try:
            get_pipeline(company).clear_memory()
        except Exception:
            pass
    return jsonify({"success": True})


# ── Misc ─────────────────────────────────────────────────────────────────────

@app.route("/supported-formats")
def supported_formats():
    """Return list of supported file extensions."""
    return jsonify({"extensions": get_supported_extensions()})


@app.route("/collection-info")
def collection_info():
    """Return Qdrant collection info for a company."""
    company = request.args.get("company", "").strip()
    if not company:
        return jsonify({"error": "company param required"}), 400
    try:
        pipeline = get_pipeline(company)
        info = pipeline.vector_store.get_collection_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Multi-Company RAG Web App on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
