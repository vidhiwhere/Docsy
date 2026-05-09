"""
qa_engine.py — Query pipeline: embed → retrieve → answer.
Uses OpenAI GPT for generation if NLP_MODE=openai, else returns raw chunks.
"""

from config import NLP_MODE, OPENAI_API_KEY, OPENAI_CHAT, TOP_K
from modules.embedder import get_embeddings
from modules.indexer import search, log_query


SYSTEM_PROMPT = """You are Docsy, an intelligent documentation assistant.
Answer the user's question strictly using the provided document context.
Be concise, accurate, and cite sources where appropriate.
If the answer is not in the provided context, say: "I couldn't find that in the available documents."
Do not make up information."""


def answer_query(question: str) -> dict:
    """
    Main Q&A pipeline.
    Returns: { answer, sources, mode }
    """
    log_query(question)

    # 1. Embed the question
    q_embedding = get_embeddings([question])[0]

    # 2. Retrieve top-K chunks
    chunks = search(q_embedding, k=TOP_K)

    if not chunks:
        return {
            "answer": "No documents have been indexed yet. Please upload some documents first.",
            "sources": [],
            "mode": NLP_MODE,
        }

    # 3. Build source references
    sources = [
        {
            "source_file": c["source_file"],
            "page": c["page"],
            "excerpt": c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
            "score": round(c["score"], 4),
        }
        for c in chunks
    ]

    # 4. Generate answer
    if NLP_MODE == "openai":
        answer = _openai_answer(question, chunks)
    else:
        answer = _local_answer(question, chunks)

    return {
        "answer": answer,
        "sources": sources,
        "mode": NLP_MODE,
    }


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------

def _openai_answer(question: str, chunks: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    context = "\n\n---\n\n".join(
        f"[Source: {c['source_file']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = client.chat.completions.create(
        model=OPENAI_CHAT,
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def _local_answer(question: str, chunks: list[dict]) -> str:
    """
    Without an LLM, return the most relevant chunk as a direct excerpt.
    """
    best = chunks[0]
    return (
        f"Based on **{best['source_file']}** (Page {best['page']}):\n\n"
        f"{best['text'][:800]}"
    )
