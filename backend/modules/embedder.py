"""
embedder.py — Embedding generation.
Switches between OpenAI and local sentence-transformers based on config.
"""

import numpy as np
from config import NLP_MODE, OPENAI_API_KEY, OPENAI_EMBED

# Lazy-loaded models
_local_model = None
_openai_client = None


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts. Returns a 2D numpy array of shape (N, dim).
    """
    if not texts:
        return np.array([])

    if NLP_MODE == "openai":
        return _openai_embed(texts)
    else:
        return _local_embed(texts)


def get_embedding_dim() -> int:
    """Returns the embedding dimensionality for the current mode."""
    if NLP_MODE == "openai":
        return 1536  # ada-002
    else:
        model = _get_local_model()
        return model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _openai_embed(texts: list[str]) -> np.ndarray:
    client = _get_openai_client()
    # Batch in groups of 100
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=OPENAI_EMBED, input=batch)
        for item in response.data:
            all_embeddings.append(item.embedding)
    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Local backend (sentence-transformers)
# ---------------------------------------------------------------------------

def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print("[Docsy] Loading local embedding model (all-MiniLM-L6-v2)…")
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def _local_embed(texts: list[str]) -> np.ndarray:
    model = _get_local_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)
