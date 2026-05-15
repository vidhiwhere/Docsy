"""
app.py — Docsy Flask API entry point.
"""

import os
import sys

# Ensure modules directory is importable
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import config
from modules.indexer import (
    init_db, add_document, add_chunks, get_document_by_hash,
    list_documents, delete_document, get_stats
)
from modules.parser import parse_file, file_hash
from modules.embedder import get_embeddings
from modules.qa_engine import answer_query, summarize_document

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

with app.app_context():
    init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": config.NLP_MODE})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return _error("No files field in request")

    uploaded = request.files.getlist("files")
    if not uploaded:
        return _error("No files selected")

    results = []
    for file in uploaded:
        if not file.filename:
            continue

        if not _allowed_file(file.filename):
            results.append({"filename": file.filename, "status": "error",
                             "message": "Unsupported file type"})
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_DIR, filename)
        file.save(filepath)

        # Duplicate detection via hash
        fhash = file_hash(filepath)
        existing = get_document_by_hash(fhash)
        if existing:
            os.remove(filepath)
            results.append({"filename": filename, "status": "skipped",
                             "message": "Document already indexed"})
            continue

        try:
            chunks = parse_file(filepath)
            if not chunks:
                results.append({"filename": filename, "status": "error",
                                 "message": "Could not extract text"})
                continue

            texts = [c["text"] for c in chunks]
            embeddings = get_embeddings(texts)

            page_count = max(c["page"] for c in chunks)
            doc_id = add_document(filename, filepath, fhash, page_count, len(chunks))
            add_chunks(doc_id, chunks, embeddings)

            results.append({
                "filename": filename,
                "status": "success",
                "chunks": len(chunks),
                "pages": page_count,
            })

        except Exception as e:
            results.append({"filename": filename, "status": "error",
                             "message": str(e)})

    return jsonify({"results": results})


@app.route("/api/query", methods=["POST"])
def query():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return _error("Question is required")
    if len(question) > 1000:
        return _error("Question too long (max 1000 characters)")

    try:
        result = answer_query(question)
        return jsonify(result)
    except Exception as e:
        return _error(f"Query failed: {str(e)}", 500)


@app.route("/api/documents", methods=["GET"])
def documents():
    docs = list_documents()
    return jsonify({"documents": docs})


@app.route("/api/summarize/<int:doc_id>", methods=["GET", "POST"])
def summarize(doc_id: int):
    try:
        result = summarize_document(doc_id)
        return jsonify(result)
    except Exception as e:
        return _error(f"Summarization failed: {str(e)}", 500)


@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
def delete_doc(doc_id: int):
    try:
        delete_document(doc_id)
        return jsonify({"status": "deleted", "id": doc_id})
    except Exception as e:
        return _error(str(e), 500)


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[Docsy] Starting in {config.NLP_MODE.upper()} mode on port {config.FLASK_PORT}")
    app.run(debug=config.FLASK_DEBUG, port=config.FLASK_PORT)
