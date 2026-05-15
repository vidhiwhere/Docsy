"""
indexer.py — FAISS vector index + SQLite metadata store.
Handles build, persist, load, search, and delete operations.
"""

import os
import json
import sqlite3
import numpy as np
import faiss
from config import INDEX_DIR, DB_PATH, TOP_K
from modules.embedder import get_embedding_dim

FAISS_PATH = os.path.join(INDEX_DIR, "docsy.index")
ID_MAP_PATH = os.path.join(INDEX_DIR, "id_map.json")

# In-memory state
_index: faiss.Index | None = None
_id_map: list[int] = []   # position → sqlite chunk id


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                filepath    TEXT NOT NULL,
                file_hash   TEXT NOT NULL,
                page_count  INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                page        INTEGER NOT NULL,
                text        TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                question   TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------

def _load_index():
    global _index, _id_map
    if os.path.exists(FAISS_PATH):
        _index = faiss.read_index(FAISS_PATH)
        with open(ID_MAP_PATH, "r") as f:
            _id_map = json.load(f)
    else:
        dim = get_embedding_dim()
        _index = faiss.IndexFlatL2(dim)
        _id_map = []


def _save_index():
    faiss.write_index(_index, FAISS_PATH)
    with open(ID_MAP_PATH, "w") as f:
        json.dump(_id_map, f)


def get_index():
    global _index
    if _index is None:
        _load_index()
    return _index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_chunks(doc_id: int, chunks: list[dict], embeddings: np.ndarray):
    """Insert chunks into SQLite and their embeddings into FAISS."""
    get_index()   # ensure loaded
    chunk_ids = []
    with _get_conn() as conn:
        for chunk in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, chunk_index, page, text) VALUES (?,?,?,?)",
                (doc_id, chunk["chunk_index"], chunk["page"], chunk["text"])
            )
            chunk_ids.append(cur.lastrowid)
        conn.commit()

    _index.add(embeddings)
    _id_map.extend(chunk_ids)
    _save_index()


def search(query_embedding: np.ndarray, k: int = TOP_K) -> list[dict]:
    """
    Return top-K chunks as dicts with text and source metadata.
    """
    get_index()
    if _index.ntotal == 0:
        return []

    k = min(k, _index.ntotal)
    distances, indices = _index.search(query_embedding.reshape(1, -1), k)

    results = []
    with _get_conn() as conn:
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_id_map):
                continue
            chunk_id = _id_map[idx]
            row = conn.execute(
                """SELECT c.text, c.page, c.chunk_index, d.filename
                   FROM chunks c
                   JOIN documents d ON c.doc_id = d.id
                   WHERE c.id = ?""",
                (chunk_id,)
            ).fetchone()
            if row:
                results.append({
                    "text": row["text"],
                    "page": row["page"],
                    "chunk_index": row["chunk_index"],
                    "source_file": row["filename"],
                    "score": float(dist),
                })
    return results


def add_document(filename: str, filepath: str, file_hash: str,
                 page_count: int, chunk_count: int) -> int:
    """Insert a document record and return its id."""
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO documents (filename, filepath, file_hash, page_count, chunk_count)
               VALUES (?,?,?,?,?)""",
            (filename, filepath, file_hash, page_count, chunk_count)
        )
        conn.commit()
        return cur.lastrowid


def get_document_by_hash(file_hash: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return dict(row) if row else None


def list_documents() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, filename, page_count, chunk_count, created_at FROM documents ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_chunks_for_doc(doc_id: int) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT text, page, chunk_index FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id: int):
    """
    Remove document and its chunks from SQLite.
    FAISS requires full rebuild since it doesn't support per-vector deletion.
    """
    from modules.embedder import get_embeddings

    with _get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()

    _rebuild_index()


def _rebuild_index():
    """Re-embed all remaining chunks and rebuild FAISS from scratch."""
    global _index, _id_map
    from modules.embedder import get_embeddings, get_embedding_dim

    dim = get_embedding_dim()
    _index = faiss.IndexFlatL2(dim)
    _id_map = []

    with _get_conn() as conn:
        rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()

    if not rows:
        _save_index()
        return

    texts = [r["text"] for r in rows]
    ids   = [r["id"]   for r in rows]
    embeddings = get_embeddings(texts)
    _index.add(embeddings)
    _id_map = ids
    _save_index()


def get_stats() -> dict:
    with _get_conn() as conn:
        doc_count   = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        query_count = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        today_queries = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE date(created_at) = date('now')"
        ).fetchone()[0]
        recent = conn.execute(
            "SELECT question, created_at FROM query_log ORDER BY created_at DESC LIMIT 5"
        ).fetchall()
    return {
        "doc_count":     doc_count,
        "chunk_count":   chunk_count,
        "query_count":   query_count,
        "today_queries": today_queries,
        "recent_queries": [dict(r) for r in recent],
        "index_vectors": get_index().ntotal if get_index() else 0,
    }


def log_query(question: str):
    with _get_conn() as conn:
        conn.execute("INSERT INTO query_log (question) VALUES (?)", (question,))
        conn.commit()
