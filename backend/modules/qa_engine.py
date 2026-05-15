"""
qa_engine.py — Query pipeline: embed → retrieve → answer.
Uses OpenAI GPT for generation if NLP_MODE=openai, else returns raw chunks.
"""

from config import NLP_MODE, OPENAI_API_KEY, OPENAI_CHAT, TOP_K
from modules.embedder import get_embeddings
from modules.indexer import search, log_query, get_chunks_for_doc


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


def summarize_document(doc_id: int) -> dict:
    chunks = get_chunks_for_doc(doc_id)
    if not chunks:
        return {"summary": "Document has no text or could not be found."}

    # Limit to first ~5000 chars to avoid hitting token limits for large docs
    full_text = "\n\n".join(c["text"] for c in chunks)[:5000]

    if NLP_MODE == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Summarize the following document content concisely in 2-4 paragraphs. Highlight the key takeaways."},
            {"role": "user", "content": f"Document content:\n{full_text}"},
        ]
        try:
            response = client.chat.completions.create(
                model=OPENAI_CHAT,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            summary = f"Summarization failed: {str(e)}"
    else:
        # Local fallback just returns the beginning of the text
        summary = "*(OpenAI Mode is disabled. Here is an excerpt from the beginning of the document instead of a full summary:)*\n\n" + full_text[:800] + "..."

    return {"summary": summary}


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
