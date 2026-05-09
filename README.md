# Docsy 🧠 — Internal Docs Q&A Agent

Docsy is a self-hosted, AI-powered documentation assistant. Upload internal documents (PDF, DOCX, Markdown, TXT) and query them in plain English. Powered by FAISS vector search and OpenAI or local sentence-transformers.

---

## Features

- 📄 **Multi-format ingestion** — PDF, DOCX, Markdown, TXT
- 🔍 **Semantic search** — FAISS vector index with embedding-based retrieval
- 🤖 **LLM answers** — OpenAI GPT-3.5 or local sentence-transformers fallback
- 🖥️ **Premium dark UI** — Glassmorphism SPA with animated stats dashboard
- ♻️ **Duplicate detection** — file hash prevents re-indexing the same document
- 🗑️ **Document management** — delete docs and auto-rebuild index

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, Vanilla CSS, Vanilla JS |
| Backend | Python 3.9+, Flask, Flask-CORS |
| Embeddings | OpenAI `text-embedding-ada-002` OR `sentence-transformers` |
| Vector Search | FAISS (`faiss-cpu`) |
| Document Parsing | PyPDF2, python-docx, BeautifulSoup4 |
| Metadata Store | SQLite |
| LLM (optional) | OpenAI `gpt-3.5-turbo` |

---

## Quick Start

### 1. Clone & set up backend

```bash
git clone https://github.com/vidhiwhere/Docsy.git
cd Docsy

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set your OPENAI_API_KEY (or set NLP_MODE=local)
```

### 3. Run the backend

```bash
cd backend
python app.py
```

Backend runs at `http://localhost:5000`

### 4. Open the frontend

Open `frontend/index.html` in your browser — no build step needed.

---

## Project Structure

```
docsy/
├── backend/
│   ├── app.py               # Flask API
│   ├── config.py            # Environment config
│   ├── requirements.txt
│   ├── .env.example
│   └── modules/
│       ├── parser.py        # PDF/DOCX/MD/TXT parser + chunker
│       ├── embedder.py      # OpenAI or local embeddings
│       ├── indexer.py       # FAISS index + SQLite metadata
│       └── qa_engine.py     # Query → retrieve → answer pipeline
└── frontend/
    ├── index.html
    ├── css/style.css
    └── js/
        ├── app.js
        ├── query.js
        ├── upload.js
        └── dashboard.js
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + NLP mode |
| POST | `/api/upload` | Upload & index documents |
| POST | `/api/query` | Ask a question |
| GET | `/api/documents` | List all documents |
| DELETE | `/api/documents/<id>` | Remove a document |
| GET | `/api/stats` | Dashboard stats |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NLP_MODE` | `openai` | `openai` or `local` |
| `OPENAI_API_KEY` | — | Required for OpenAI mode |
| `OPENAI_EMBED_MODEL` | `text-embedding-ada-002` | Embedding model |
| `OPENAI_CHAT_MODEL` | `gpt-3.5-turbo` | Chat model |
| `TOP_K_RESULTS` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
