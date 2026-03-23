"""
Unified RAG Web Application
Connects PDF upload, web scraping, and vector search in one interface
- Flat directory structure (no pdf_rag/ web_scraping_rag/ subfolders)
- Single config.yaml for all settings
- Combined retrieval from both PDF and web sources
"""
import os
import sys
import time
import json
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# Setup Python path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Load config
from config.config import Config, get_config

cfg = get_config()

# Import RAG components
from src.rag_pipeline import RAGPipeline
from src.document_processor import PDFProcessor
from src.chunking import HybridChunker
from src.embeddings import EmbeddingManager
from src.vector_store import QdrantVectorStore
from src.utils import setup_logging

# Setup logging
log_dir = cfg.get_logs_dir()
os.makedirs(log_dir, exist_ok=True)
setup_logging(log_level=cfg.log_level, log_file=os.path.join(log_dir, "web.log"))

import logging
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))

# Initialize RAG pipeline
pipeline = None

def init_pipeline():
    """Initialize RAG pipeline once"""
    global pipeline
    if pipeline is None:
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        logger.info("[SUCCESS] RAG pipeline initialized")
    return pipeline

# File upload config
UPLOAD_FOLDER = cfg.get_pdf_dir()
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max


def allowed_file(filename):
    """Check if file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def hash_file(filepath: str) -> str:
    """Compute SHA256 hash of file"""
    import hashlib
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def hash_url(url: str) -> str:
    """Return SHA256 hash of a URL"""
    return hashlib.sha256(url.encode()).hexdigest()


def load_manifest() -> dict:
    """Load ingest manifest"""
    manifest_path = os.path.join(cfg.get_data_dir(), "ingest_manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(manifest: dict):
    """Save ingest manifest"""
    data_dir = cfg.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, "ingest_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def index():
    """Main page"""
    return render_template("index.html", user="Admin", role="engineering")


@app.route("/files")
def files():
    """List all PDF files"""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    files_list = []
    for fname in sorted(os.listdir(UPLOAD_FOLDER)):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(UPLOAD_FOLDER, fname)
            try:
                size = os.path.getsize(path)
                files_list.append({"name": fname, "size": f"{(size/1024):.2f} KB"})
            except Exception:
                pass
    return jsonify({"files": files_list})


@app.route("/download/<path:filename>")
def download(filename):
    """Download PDF file"""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/files/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    """Delete PDF file"""
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404


@app.route("/upload", methods=["POST"])
def upload():
    """Upload and ingest PDF file"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files allowed"}), 400
        
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"Processing PDF: {filename}")
        
        # Process and ingest
        pp = init_pipeline()
        
        processor = PDFProcessor()
        chunker = HybridChunker(cfg.chunking)
        embedder = EmbeddingManager(**cfg.embedding)
        vectorstore = QdrantVectorStore(**cfg.vectorstore)
        
        docs = processor.load_single_pdf(filepath)
        if not docs:
            return jsonify({"error": "Failed to load PDF pages"}), 500
        
        chunks = chunker.chunk(docs)
        embeddings = embedder.embed_documents(chunks)
        vectorstore.add_documents(chunks, embeddings)
        
        # Update manifest
        try:
            file_hash = hash_file(filepath)
            manifest = load_manifest()
            manifest[os.path.abspath(filepath)] = file_hash
            save_manifest(manifest)
        except Exception:
            pass
        
        # Refresh pipeline retriever
        try:
            all_docs = processor.load_directory(UPLOAD_FOLDER, recursive=True)
            pp.retriever.update_documents(all_docs)
        except Exception:
            pass
        
        logger.info(f"[SUCCESS] PDF ingested: {filename}")
        return jsonify({"success": True, "filename": filename})
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/scrape", methods=["POST"])
def scrape():
    """Scrape URL and ingest content"""
    try:
        data = request.get_json() or {}
        url = data.get("url", "").strip()
        mode = data.get("mode", "auto")  # auto, advanced, fallback
        
        if not url:
            return jsonify({"error": "Missing URL"}), 400
        
        logger.info(f"Scraping {url} (mode: {mode})")
        
        pp = init_pipeline()
        
        # Try advanced scraper (Selenium-based) - with timeout handling
        advanced_success = False
        if mode in ["auto", "advanced"]:
            try:
                from src.web_scraper import WebScraper
                
                logger.info("Attempting advanced scraper (Selenium with link discovery)...")
                # Use headless=False for better reliability and max_depth=2 for comprehensive scraping
                scraper = WebScraper(headless=False, follow_links=True, max_depth=2)
                documents = scraper.scrape_urls([url])
                
                if documents:
                    logger.info(f"[SUCCESS] Advanced scraper extracted {len(documents)} documents")
                    advanced_success = True
                    
                    # Log HTML parsing details
                    html_parsed = sum(1 for d in documents if d.metadata.get('html_parsed', False))
                    extraction_methods = set(d.metadata.get('extraction_method', 'unknown') for d in documents)
                    logger.info(f"  - HTML parsing: {html_parsed}/{len(documents)} documents")
                    logger.info(f"  - Extraction methods: {', '.join(extraction_methods)}")
                    
                    # Log scroll extraction details
                    scroll_extractions = sum(1 for d in documents if d.metadata.get('scroll_extraction', False))
                    logger.info(f"  - Scroll extraction used for: {scroll_extractions}/{len(documents)} documents")
                    
                    # Log content statistics
                    total_chars = sum(d.metadata.get('char_count', 0) for d in documents)
                    total_words = sum(d.metadata.get('word_count', 0) for d in documents)
                    logger.info(f"  - Total content: {total_chars} characters, {total_words} words")
                    
                    # Categorize URLs (seed vs discovered)
                    seed_urls = set([url])
                    discovered_urls = []
                    for doc in documents:
                        doc_url = doc.metadata.get('source', '')
                        if doc_url not in seed_urls:
                            discovered_urls.append(doc_url)
                    
                    logger.info(f"  - Seed URL: {len(seed_urls)}")
                    logger.info(f"  - Discovered links: {len(discovered_urls)}")
                    
                    # Process documents
                    processor = PDFProcessor()
                    chunker = HybridChunker(cfg.chunking)
                    embedder = EmbeddingManager(**cfg.embedding)
                    vectorstore = QdrantVectorStore(**cfg.vectorstore)
                    
                    # Save as PDFs
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas
                    
                    saved_pdfs = []
                    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    
                    for i, doc in enumerate(documents, start=1):
                        pdf_name = f"scraped_{timestamp}_{i}.pdf"
                        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_name)
                        
                        try:
                            c = canvas.Canvas(pdf_path, pagesize=letter)
                            width, height = letter
                            margin = 40
                            y = height - margin
                            for line in str(doc.page_content).splitlines():
                                if y < margin:
                                    c.showPage()
                                    y = height - margin
                                c.drawString(margin, y, line[:200])
                                y -= 12
                            c.save()
                            saved_pdfs.append(pdf_name)
                        except Exception:
                            pass
                    
                    # Chunk & embed documents
                    chunks = chunker.chunk(documents)
                    embeddings = embedder.embed_documents(chunks)
                    vectorstore.add_documents(chunks, embeddings)
                    
                    # Update manifest to track all scraped URLs (including discovered ones)
                    try:
                        manifest = load_manifest()
                        for doc in documents:
                            doc_url = doc.metadata.get('source', '')
                            if doc_url:
                                manifest[doc_url] = hashlib.sha256(doc_url.encode()).hexdigest()
                        
                        # Also track the saved PDFs
                        for pdf in saved_pdfs:
                            path = os.path.join(UPLOAD_FOLDER, pdf)
                            try:
                                manifest[os.path.abspath(path)] = hash_file(path)
                            except Exception:
                                pass
                        save_manifest(manifest)
                    except Exception:
                        pass
                    
                    # Refresh retriever
                    try:
                        all_docs = processor.load_directory(UPLOAD_FOLDER, recursive=True)
                        pp.retriever.update_documents(all_docs)
                    except Exception:
                        pass
                    
                    return jsonify({
                        "pdf_filename": saved_pdfs,
                        "title": documents[0].metadata.get("title", url),
                        "domain": documents[0].metadata.get("domain", ""),
                        "seed_urls": len(seed_urls),
                        "discovered_urls": len(discovered_urls),
                        "total_documents": len(documents),
                        "text_length": sum(len(d.page_content or "") for d in documents),
                        "chunks_created": len(chunks),
                        "html_parsing": "comprehensive",
                        "parsing_method": documents[0].metadata.get("extraction_method", "unknown"),
                        "method": "advanced"
                    })
            
            except Exception as e:
                if mode == "advanced":
                    logger.error(f"Advanced scraper forced but failed: {str(e)}")
                    return jsonify({"error": f"Advanced scraping failed: {str(e)}"}), 500
                logger.warning(f"Advanced scraper failed (will fallback): {str(e)}")
                advanced_success = False
        
        # Fallback to lightweight scraper
        if not advanced_success:
            logger.info("Falling back to lightweight scraper (requests + BeautifulSoup)")
            import requests
            from bs4 import BeautifulSoup
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            # Add proper headers to avoid 403 Forbidden
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            try:
                r = requests.get(url, timeout=15, headers=headers)
                r.raise_for_status()
            except Exception as e:
                logger.error(f"Fallback scraper failed to fetch URL: {str(e)}")
                return jsonify({"error": f"Failed to scrape URL: {str(e)}"}), 500
            
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Better title extraction
            title = None
            if soup.h1:
                title = soup.h1.get_text(strip=True)
            elif soup.title:
                title = soup.title.string
            elif soup.h2:
                title = soup.h2.get_text(strip=True)
            else:
                title = url
            
            # Better content extraction - remove noise elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Try to find main content area
            main_content = (
                soup.find('article') or 
                soup.find('main') or 
                soup.find('div', class_=lambda x: x and 'content' in x.lower()) or
                soup.find('div', id=lambda x: x and 'content' in x.lower()) or
                soup.find('body')
            )
            
            # Extract structured content
            text_lines = []
            
            # Extract headers
            for header in main_content.find_all(['h2', 'h3', 'h4']):
                text_lines.append(header.get_text(strip=True))
            
            # Extract paragraphs
            for para in main_content.find_all('p'):
                p_text = para.get_text(strip=True)
                if p_text and len(p_text) > 20:  # Skip small fragments
                    text_lines.append(p_text)
            
            # Extract list items
            for li in main_content.find_all('li'):
                li_text = li.get_text(strip=True)
                if li_text:
                    text_lines.append(f"• {li_text}")
            
            # Extract table content
            for table in main_content.find_all('table'):
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    text_lines.append(" | ".join(cells))
            
            # Join all content
            text = "\n".join(text_lines)
            
            # Fallback to full text if not enough content extracted
            if not text or len(text) < 100:
                logger.warning(f"Minimal content found ({len(text)} chars), using full page text extraction")
                text = soup.get_text(separator="\n")
            
            domain = requests.utils.urlparse(url).netloc
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            pdf_name = f"scraped_{timestamp}.pdf"
            pdf_path = os.path.join(UPLOAD_FOLDER, pdf_name)
            
            # Clean up text
            import re
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive blank lines
            text = re.sub(r' +', ' ', text)  # Remove excessive spaces
            
            logger.info(f"Extracted content: {len(text)} characters, {len(text.split())} words")
            
            # Write PDF with better content preservation
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.units import inch
            
            try:
                pdf_doc = SimpleDocTemplate(pdf_path, pagesize=letter)
                story = []
                
                # Add title
                title_style = ParagraphStyle(
                    'CustomTitle',
                    fontSize=16,
                    textColor='#000000',
                    spaceAfter=12,
                    alignment=1  # Center
                )
                story.append(Paragraph(f"<b>{title}</b>", title_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Add source URL
                story.append(Paragraph(f"<i>Source: {url}</i>", ParagraphStyle('Source', fontSize=10)))
                story.append(Paragraph(f"<i>Scraped: {timestamp}</i>", ParagraphStyle('Time', fontSize=10)))
                story.append(Spacer(1, 0.3*inch))
                
                # Add content in chunks
                body_style = ParagraphStyle(
                    'CustomBody',
                    fontSize=10,
                    leading=14,
                    spaceAfter=6
                )
                
                # Split text into paragraphs
                paragraphs = text.split('\n\n')
                for para_text in paragraphs:
                    if para_text.strip():
                        try:
                            story.append(Paragraph(para_text.strip(), body_style))
                            story.append(Spacer(1, 0.1*inch))
                        except:
                            # Fallback for problematic text
                            story.append(Paragraph(para_text[:500].strip(), body_style))
                
                pdf_doc.build(story)
                logger.info(f"Successfully created PDF: {pdf_name}")
                
            except Exception as pdf_err:
                logger.warning(f"Platypus PDF creation failed, falling back to Canvas: {pdf_err}")
                # Fallback to simpler Canvas method
                c = canvas.Canvas(pdf_path, pagesize=letter)
                width, height = letter
                margin = 40
                y = height - margin
                
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin, y, f"Title: {title[:100]}")
                y -= 18
                c.setFont("Helvetica", 9)
                c.drawString(margin, y, f"Source: {url[:100]}")
                y -= 12
                c.drawString(margin, y, f"Scraped: {timestamp}")
                y -= 30
                
                c.setFont("Helvetica", 9)
                for line in text.split('\n'):
                    if y < margin:
                        c.showPage()
                        y = height - margin
                    c.drawString(margin, y, line[:200])
                    y -= 12
                
                c.save()
                logger.info(f"Created PDF with Canvas: {pdf_name}")
            
            # Ingest
            try:
                processor = PDFProcessor()
                chunker = HybridChunker(cfg.chunking)
                embedder = EmbeddingManager(**cfg.embedding)
                vectorstore = QdrantVectorStore(**cfg.vectorstore)
                
                docs = processor.load_single_pdf(pdf_path)
                if docs:
                    chunks = chunker.chunk(docs)
                    embeddings = embedder.embed_documents(chunks)
                    vectorstore.add_documents(chunks, embeddings)
                    
                    # Update manifest
                    try:
                        file_hash = hash_file(pdf_path)
                        manifest = load_manifest()
                        manifest[os.path.abspath(pdf_path)] = file_hash
                        save_manifest(manifest)
                    except Exception:
                        pass
                    
                    # Refresh retriever
                    try:
                        all_docs = processor.load_directory(UPLOAD_FOLDER, recursive=True)
                        pp.retriever.update_documents(all_docs)
                    except Exception:
                        pass
            except Exception:
                pass
            
            logger.info(f"[SUCCESS] Fallback scraper completed: {pdf_name}")
            
            # Count chunks created if docs were processed
            chunks_count = 0
            try:
                if docs:
                    chunks_count = len(chunks)
            except:
                chunks_count = 0
            
            return jsonify({
                "pdf_filename": pdf_name,
                "title": title,
                "domain": domain,
                "seed_urls": 1,
                "discovered_urls": 0,
                "total_documents": 1,
                "text_length": len(text),
                "chunks_created": chunks_count,
                "html_parsing": "basic",
                "parsing_method": "requests_beautifulsoup",
                "method": "fallback"
            })
    
    except Exception as e:
        logger.error(f"Scrape error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/scrape-batch", methods=["POST"])
def scrape_batch():
    """Scrape multiple URLs and ingest content"""
    try:
        data = request.get_json() or {}
        urls = data.get("urls", [])
        
        if not urls or not isinstance(urls, list):
            return jsonify({"error": "Missing or invalid URLs list"}), 400
        
        logger.info(f"Batch scraping {len(urls)} URLs")
        
        pp = init_pipeline()
        
        # Load manifest to track processed URLs
        manifest = load_manifest()
        
        # Filter new URLs
        new_urls = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            url_hash = hash_url(url)
            if url not in manifest or manifest[url] != url_hash:
                new_urls.append(url)
                manifest[url] = url_hash
        
        if not new_urls:
            logger.info("All URLs already processed")
            return jsonify({
                "status": "skipped",
                "message": "All URLs already processed",
                "seed_urls": 0,
                "total_documents": 0,
                "chunks_created": 0,
                "vectors_stored": 0
            })
        
        logger.info(f"Processing {len(new_urls)} new URLs: {new_urls}")
        
        # Scrape URLs with advanced scraper
        try:
            from src.web_scraper import WebScraper
            
            logger.info("Advanced scraper: scraping multiple URLs with link discovery...")
            scraper = WebScraper(headless=True, follow_links=True, max_depth=2)
            documents = scraper.scrape_urls(new_urls)
            
            if not documents:
                logger.warning("No documents scraped successfully")
                return jsonify({"error": "No documents scraped successfully"}), 400
            
            logger.info(f"Scraped {len(documents)} documents with advanced scraper")
            
            # Categorize URLs (seed vs discovered)
            seed_url_set = set(new_urls)
            seed_scraped = []
            discovered_urls = []
            
            for doc in documents:
                doc_url = doc.metadata.get('source', '')
                if doc_url:
                    url_hash = hash_url(doc_url)
                    manifest[doc_url] = url_hash
                    
                    if doc_url in seed_url_set:
                        seed_scraped.append(doc_url)
                    else:
                        discovered_urls.append(doc_url)
            
            logger.info(f"Seed URLs scraped: {len(seed_scraped)}")
            logger.info(f"Discovered links: {len(discovered_urls)}")
            
            # Save documents as PDFs
            processor = PDFProcessor()
            chunker = HybridChunker(cfg.chunking)
            embedder = EmbeddingManager(**cfg.embedding)
            vectorstore = QdrantVectorStore(**cfg.vectorstore)
            
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            saved_pdfs = []
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            
            for i, doc in enumerate(documents, start=1):
                pdf_name = f"scraped_{timestamp}_{i}.pdf"
                pdf_path = os.path.join(UPLOAD_FOLDER, pdf_name)
                
                try:
                    # Try using Platypus for better content preservation
                    from reportlab.lib.styles import ParagraphStyle
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.units import inch
                    
                    pdf_doc = SimpleDocTemplate(pdf_path, pagesize=letter)
                    story = []
                    
                    title = doc.metadata.get('title', f'Document {i}')
                    source = doc.metadata.get('source', '')
                    
                    # Add metadata
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        fontSize=14,
                        textColor='#000000',
                        spaceAfter=10
                    )
                    story.append(Paragraph(f"<b>{title}</b>", title_style))
                    story.append(Paragraph(f"<i>Source: {source}</i>", ParagraphStyle('Source', fontSize=9)))
                    story.append(Paragraph(f"<i>Scraped: {timestamp}</i>", ParagraphStyle('Time', fontSize=9)))
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Add content
                    body_style = ParagraphStyle(
                        'CustomBody',
                        fontSize=10,
                        leading=12,
                        spaceAfter=6
                    )
                    
                    content = str(doc.page_content)
                    for para in content.split('\n\n'):
                        if para.strip():
                            try:
                                story.append(Paragraph(para.strip()[:2000], body_style))
                            except:
                                pass
                    
                    pdf_doc.build(story)
                    saved_pdfs.append(pdf_name)
                    logger.info(f"Saved PDF: {pdf_name}")
                    
                except Exception as e:
                    logger.warning(f"Platypus failed for {pdf_name}, using Canvas: {e}")
                    # Fallback to Canvas
                    try:
                        c = canvas.Canvas(pdf_path, pagesize=letter)
                        width, height = letter
                        margin = 40
                        y = height - margin
                        font_size = 9
                        line_height = 12
                        
                        c.setFont("Helvetica-Bold", 10)
                        title = doc.metadata.get('title', f'Document {i}')[:150]
                        c.drawString(margin, y, title)
                        y -= line_height + 4
                        
                        c.setFont("Helvetica", 8)
                        source = doc.metadata.get('source', '')[:150]
                        c.drawString(margin, y, f"Source: {source}")
                        y -= line_height
                        
                        c.setFont("Helvetica", font_size)
                        for line in str(doc.page_content).split('\n'):
                            if y < margin:
                                c.showPage()
                                y = height - margin
                            c.drawString(margin, y, line[:200])
                            y -= line_height
                        
                        c.save()
                        saved_pdfs.append(pdf_name)
                        logger.info(f"Saved PDF with Canvas: {pdf_name}")
                    except Exception as canvas_err:
                        logger.error(f"Error saving PDF {pdf_name}: {canvas_err}")
        
        except Exception as batch_scrape_error:
            logger.error(f"Batch scraping error: {str(batch_scrape_error)}", exc_info=True)
            
            # Try fallback method for batch scraping
            logger.info("Attempting fallback batch scraping method...")
            try:
                import requests
                from bs4 import BeautifulSoup
                from urllib.parse import urlparse
                
                processor = PDFProcessor()
                chunker = HybridChunker(cfg.chunking)
                embedder = EmbeddingManager(**cfg.embedding)
                vectorstore = QdrantVectorStore(**cfg.vectorstore)
                
                documents = []
                saved_pdfs = []
                timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                }
                
                for idx, url in enumerate(new_urls, 1):
                    try:
                        logger.info(f"Fallback scraping [{idx}/{len(new_urls)}]: {url}")
                        r = requests.get(url, timeout=15, headers=headers)
                        r.raise_for_status()
                        
                        soup = BeautifulSoup(r.text, "html.parser")
                        
                        # Extract title and content
                        title = None
                        if soup.h1:
                            title = soup.h1.get_text(strip=True)
                        elif soup.title:
                            title = soup.title.string
                        else:
                            title = f"Document {idx}"
                        
                        # Remove noise
                        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                            element.decompose()
                        
                        # Extract content
                        main_content = soup.find(['article', 'main', 'body'])
                        text_lines = []
                        
                        if main_content:
                            for header in main_content.find_all(['h2', 'h3', 'h4']):
                                text_lines.append(header.get_text(strip=True))
                            for para in main_content.find_all('p'):
                                p_text = para.get_text(strip=True)
                                if len(p_text) > 20:
                                    text_lines.append(p_text)
                            for li in main_content.find_all('li'):
                                text_lines.append(f"• {li.get_text(strip=True)}")
                        
                        text = "\n".join(text_lines)
                        if not text or len(text) < 100:
                            text = soup.get_text(separator="\n")
                        
                        # Clean text
                        import re
                        text = re.sub(r'\n\s*\n', '\n\n', text)
                        
                        logger.info(f"Fallback: Extracted {len(text)} chars from {title}")
                        
                        # Save as PDF
                        from reportlab.lib.pagesizes import letter
                        from reportlab.pdfgen import canvas
                        
                        pdf_name = f"scraped_{timestamp}_{idx}.pdf"
                        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_name)
                        
                        try:
                            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                            from reportlab.lib.styles import ParagraphStyle
                            from reportlab.lib.units import inch
                            
                            pdf_doc = SimpleDocTemplate(pdf_path, pagesize=letter)
                            story = []
                            
                            title_style = ParagraphStyle('Title', fontSize=12, spaceAfter=6)
                            body_style = ParagraphStyle('Body', fontSize=9, leading=11)
                            
                            story.append(Paragraph(f"<b>{title}</b>", title_style))
                            story.append(Paragraph(f"<i>{url}</i>", ParagraphStyle('URL', fontSize=8)))
                            story.append(Spacer(1, 0.1*inch))
                            
                            for para in text.split('\n\n'):
                                if para.strip():
                                    story.append(Paragraph(para.strip()[:1500], body_style))
                            
                            pdf_doc.build(story)
                        except:
                            # Simple canvas fallback
                            c = canvas.Canvas(pdf_path, pagesize=letter)
                            width, height = letter
                            y = height - 40
                            c.setFont("Helvetica-Bold", 11)
                            c.drawString(40, y, title[:100] if isinstance(title, str) else "Document")
                            y -= 16
                            c.setFont("Helvetica", 8)
                            c.drawString(40, y, url[:100])
                            y -= 30
                            c.setFont("Helvetica", 9)
                            for line in text.split('\n'):
                                if y < 40:
                                    c.showPage()
                                    y = height - 40
                                c.drawString(40, y, line[:180])
                                y -= 11
                            c.save()
                        
                        saved_pdfs.append(pdf_name)
                        
                        # Create document object
                        from langchain_core.documents import Document
                        doc = Document(
                            page_content=text,
                            metadata={
                                'source': url,
                                'title': title if isinstance(title, str) else "Document",
                                'domain': urlparse(url).netloc,
                            }
                        )
                        documents.append(doc)
                        logger.info(f"Fallback: Saved PDF {pdf_name}")
                        
                    except Exception as url_err:
                        logger.error(f"Failed to fallback scrape {url}: {url_err}")
                        continue
                
                if not documents:
                    return jsonify({"error": "Failed to scrape any URLs with fallback method"}), 400
                
                # Process documents
                logger.info(f"Processing {len(documents)} documents with fallback...")
                chunks = chunker.chunk(documents)
                embeddings = embedder.embed_documents(chunks)
                vectors_stored = vectorstore.add_documents(chunks, embeddings)
                
                # Update manifest and retriever
                save_manifest(manifest)
                try:
                    all_docs = processor.load_directory(UPLOAD_FOLDER, recursive=True)
                    pp.retriever.update_documents(all_docs)
                except:
                    pass
                
                logger.info(f"Batch fallback complete: {len(documents)} docs, {len(chunks)} chunks, {vectors_stored} vectors")
                
                return jsonify({
                    "status": "success_fallback",
                    "message": f"Scraped {len(documents)} documents using fallback method",
                    "pdf_filenames": saved_pdfs,
                    "seed_urls": len(new_urls),
                    "discovered_urls": 0,
                    "total_documents": len(documents),
                    "chunks_created": len(chunks),
                    "vectors_stored": vectors_stored
                })
            
            except Exception as fallback_err:
                logger.error(f"Fallback batch scraping also failed: {fallback_err}", exc_info=True)
                return jsonify({"error": f"Both advanced and fallback scraping failed: {str(fallback_err)}"}), 500
    
    except Exception as outer_err:
        logger.error(f"Outer batch scrape error: {str(outer_err)}")
        return jsonify({"error": str(outer_err)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Answer RAG questions"""
    try:
        data = request.get_json() or {}
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "Missing question"}), 400
        
        pp = init_pipeline()
        
        def generate():
            try:
                # Retrieve context
                try:
                    _ = pp.retrieve(question)
                except Exception:
                    pass
                
                # Generate answer
                answer = pp.answer(question)
                
                # Stream answer in chunks
                for i in range(0, len(answer), 200):
                    chunk = answer[i:i+200]
                    yield chunk.encode("utf-8")
                    time.sleep(0.02)
            except Exception as e:
                yield f"[Error] {e}".encode("utf-8")
        
        return Response(generate(), mimetype="text/plain")
    
    except Exception as e:
        logger.error(f"Ask error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/change-password", methods=["POST"])
def change_password():
    """Change password (placeholder)"""
    return jsonify({"success": True})


@app.route("/logout")
def logout():
    """Logout (placeholder)"""
    return "Logged out", 200


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("[START] Unified RAG Web Application")
    logger.info("="*80)
    logger.info(f"Root Dir: {ROOT_DIR}")
    logger.info(f"Config: {cfg.config_path}")
    logger.info(f"PDF Dir: {UPLOAD_FOLDER}")
    logger.info(f"Logs Dir: {cfg.get_logs_dir()}")
    logger.info("="*80)
    
    app.run(host="0.0.0.0", port=8080, debug=True)
