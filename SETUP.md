# Unified RAG Web Application - Setup & Usage

## Project Structure

```
rag/
├── web_app.py              # Main Flask application (START HERE)
├── config/
│   ├── __init__.py
│   ├── config.py           # Configuration manager class
│   └── config.yaml         # Main configuration file
├── src/                    # Core RAG components
│   ├── __init__.py
│   ├── rag_pipeline.py     # Main RAG orchestrator
│   ├── document_processor.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py     # Qdrant vector store
│   ├── hybrid_retriever.py
│   ├── web_scraper.py      # Advanced web scraper (Selenium)
│   ├── reranker.py
│   ├── llm_handler.py
│   ├── chat_memory.py
│   ├── query_expansion.py
│   ├── evaluation.py
│   └── utils.py
├── templates/              # Web UI templates
│   ├── index.html
│   └── login.html
├── data/                   # Data directory
│   ├── pdf_files/          # Uploaded PDFs
│   └── ingest_manifest.json
├── logs/                   # Application logs
│   └── web.log
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For advanced web scraping with Selenium, ensure Chrome/Chromium is installed:
```bash
# Windows: Install Chrome from https://www.google.com/chrome/
# Linux: sudo apt-get install chromium-browser
# Mac: brew install --cask chromium
```

### 2. Start Qdrant Vector Database

Option A: Using Docker (recommended)
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Option B: Using Qdrant Cloud
- Create account at https://qdrant.tech/
- Update `config/config.yaml` with your cloud URL and API key

### 3. Configure API Keys

Edit `config/config.yaml`:
```yaml
openai:
  api_key: "sk-..."  # Your OpenAI API key
```

Or set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Run the Web App

```bash
cd C:\Users\hp\OneDrive\Desktop\rag
python web_app.py
```

The app will start on `http://localhost:8080`

## Features

### 📄 PDF Upload
- Upload PDFs directly from the UI
- Automatic chunking, embedding, and vector storage
- Full-text searchable

### 🌐 Web Scraping
- Paste URL in the Web Scraping panel
- Choose mode:
  - **Auto**: Try advanced (Selenium) first, fallback to lightweight if unavailable
  - **Force Advanced**: Uses Selenium + ChromeDriver for richer extraction (requires Chrome installed)
  - **Force Fallback**: Uses simple requests + BeautifulSoup parser
- Scraped content saved as PDF and ingested into vector store

### 🤖 RAG Chat
- Ask questions about your PDFs and web content
- Hybrid retrieval (dense + sparse search)
- Contextual answers with citations

### ⚙️ Configuration

Edit `config/config.yaml` to customize:
```yaml
# Model settings
embedding:
  model_name: "BAAI/bge-large-en-v1.5"   # Embedding model

llm:
  model: "gpt-4o-mini"                    # LLM model

# Vector store
vectorstore:
  collection_name: "pdf_documents"
  host: "localhost"
  port: 6333

# File storage
ingest:
  pdf_dir: "data/pdf_files"               # Where PDFs are stored
```

## Troubleshooting

### "No module named 'langchain_core'"
```bash
pip install langchain-core --upgrade
```

### "Chrome not found" (when using Advanced scraper)
- Install Google Chrome: https://www.google.com/chrome/
- ChromeDriver will be downloaded automatically via webdriver-manager

### "ConnectionRefusedError: [localhost:6333]"
- Make sure Qdrant is running: `docker ps | grep qdrant`
- Or start it: `docker run -d -p 6333:6333 qdrant/qdrant`

### PDF upload fails
- Check that `data/pdf_files/` directory exists
- Ensure file size is < 100MB (configurable in web_app.py)
- Verify PDF is valid: `pdfinfo <filename>`

### Web scraping timeout
- Increase timeout in config or web_app.py (currently 15 seconds)
- Check internet connection
- Some sites may block Selenium; try "Force Fallback" mode

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main web interface |
| `/files` | GET | List all PDF files |
| `/upload` | POST | Upload and ingest PDF |
| `/download/<filename>` | GET | Download PDF file |
| `/files/<filename>` | DELETE | Delete PDF file |
| `/scrape` | POST | Scrape URL and ingest |
| `/ask` | POST | Ask RAG question |

## Example cURL Commands

### Upload PDF
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8080/upload
```

### Scrape URL
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"url":"https://example.com", "mode":"auto"}' \
  http://localhost:8080/scrape
```

### Ask Question
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What is RAG?"}' \
  http://localhost:8080/ask
```

## Performance Tips

1. **Embeddings**: BAAI/bge-large-en-v1.5 is accurate but slower (1024 dims). For faster results, use BAAI/bge-small-en-v1.5.

2. **Chunk Size**: Smaller chunks (512) improve retrieval accuracy but require more vectors. Default is 1000.

3. **Reranking**: Enabled by default for better answer quality. Disable if speed is critical.

4. **Vector Store**: Qdrant is optimized for dense vectors. For large datasets (>1M vectors), use Qdrant Cloud.

## Logging

Logs are stored in `logs/web.log`. Monitor them to debug issues:
```bash
tail -f logs/web.log
```

## License

MIT

## Support

For issues or questions:
1. Check logs in `logs/web.log`
2. Verify config in `config/config.yaml`
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Test Qdrant connection: `curl http://localhost:6333/health`
