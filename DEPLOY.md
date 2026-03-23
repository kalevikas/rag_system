# 🚀 Deploying RAG Intelligence to Streamlit Cloud (Free)

This guide walks you through getting a public URL for your RAG project suitable for a LinkedIn profile link.

---

## Architecture for Cloud Deployment

```
[Browser] → [Streamlit Cloud (free)] → [OpenAI API] 
                                     ↕
                              [Qdrant Cloud (free 1 GB)]
```

---

## Step 1 — Get a Free Qdrant Cloud Cluster

1. Go to **https://cloud.qdrant.io** and sign up (free).
2. Create a new cluster → choose the **Free Tier** (1 GB, always free).
3. After creation, note:
   - **Cluster URL** — looks like `https://xxxx.us-east.aws.cloud.qdrant.io`
   - **API Key** — from the "API Keys" tab in Qdrant Cloud console.

---

## Step 2 — Update `config/config.yaml` for Cloud

Open `config/config.yaml` and set:

```yaml
embedding:
  model_name: BAAI/bge-base-en-v1.5   # lighter model for cloud (768-dim)
  batch_size: 32
  normalize: true

vectorstore:
  vector_size: 768                     # must match bge-base embedding dim
  use_cloud: false                     # env vars override this automatically
  cloud_url: null
  api_key: null
```

> ⚠️ **Important:** If you switch from `bge-large-en-v1.5` (1024-dim) to
> `bge-base-en-v1.5` (768-dim) you must recreate all Qdrant collections
> (the vector dimensions must match). Delete existing collections first via
> the Qdrant Cloud dashboard.

---

## Step 3 — Push Code to GitHub

```bash
git init
git add .
git commit -m "Initial RAG Intelligence commit"
git remote add origin https://github.com/YOUR_USERNAME/rag-intelligence.git
git push -u origin main
```

Make sure `.gitignore` contains:
```
.streamlit/secrets.toml
venv_gpt/
__pycache__/
*.pyc
logs/
data/pdf_files/
.env
config/config.yaml   # optional — keep your config private
```

---

## Step 4 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"New app"** → select your repo → set main file to `streamlit_app.py`.
3. Click **"Advanced settings"** → **"Secrets"** and add:

```toml
OPENAI_API_KEY   = "sk-..."
QDRANT_URL       = "https://your-cluster.qdrant.io"
QDRANT_API_KEY   = "your-qdrant-api-key"
```

4. Click **Deploy**. Your app gets a URL like:
   `https://your-app-name.streamlit.app`

---

## Step 5 — Add to LinkedIn

1. On your LinkedIn profile → **"Add profile section"** → **"Featured"** → **"Link"**.
2. Paste your Streamlit URL and give it a title like:
   *"RAG Intelligence — Multi-User AI Document Assistant"*
3. Add a description:
   > Built a production-grade RAG system using GPT-4o-mini, Qdrant vector DB, BGE embeddings, hybrid search (BM25 + dense), cross-encoder re-ranking, and Streamlit. Each user gets an isolated vector index for private document Q&A.

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars (or create .streamlit/secrets.toml — see secrets.toml.example)
set OPENAI_API_KEY=sk-...

# Make sure Qdrant is running locally
docker run -p 6333:6333 qdrant/qdrant

# Run the app
streamlit run streamlit_app.py
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ValueError: OpenAI API key is required` | Set `OPENAI_API_KEY` in Streamlit secrets or env var |
| `ConnectionRefusedError: Qdrant` | Make sure Qdrant Cloud URL/key are correct in secrets |
| Slow first load | Embedding model downloads on first run (~400 MB for bge-base). Subsequent runs use cached model. |
| `vector_size mismatch` | Delete the Qdrant collection and re-ingest. `bge-base` = 768, `bge-large` = 1024. |
| Upload limit exceeded | Streamlit Cloud free tier has a 200 MB file upload limit (already configured). |
