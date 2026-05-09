"""
parser.py — Multi-format document parser with chunking.
Supports: PDF, DOCX, Markdown, TXT
"""

import os
import re
import hashlib
import markdown
from bs4 import BeautifulSoup
from config import CHUNK_SIZE, CHUNK_OVERLAP

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(filepath: str) -> list[dict]:
    """
    Parse a file and return a list of chunk dicts:
    { text, chunk_index, page, source_file }
    """
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)

    if ext == ".pdf":
        raw_pages = _parse_pdf(filepath)
    elif ext == ".docx":
        raw_pages = _parse_docx(filepath)
    elif ext in (".md", ".markdown"):
        raw_pages = _parse_markdown(filepath)
    elif ext == ".txt":
        raw_pages = _parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = []
    chunk_idx = 0
    for page_num, page_text in raw_pages:
        page_chunks = _chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_text in page_chunks:
            chunks.append({
                "text": chunk_text.strip(),
                "chunk_index": chunk_idx,
                "page": page_num,
                "source_file": filename,
            })
            chunk_idx += 1

    return chunks


def file_hash(filepath: str) -> str:
    """MD5 hash of a file for change detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Format parsers — return list of (page_number, text)
# ---------------------------------------------------------------------------

def _parse_pdf(filepath: str) -> list[tuple[int, str]]:
    if not PDF_AVAILABLE:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    pages = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def _parse_docx(filepath: str) -> list[tuple[int, str]]:
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    doc = DocxDocument(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Treat every ~20 paragraphs as a "page"
    pages = []
    chunk_size = 20
    for i in range(0, len(paragraphs), chunk_size):
        page_num = (i // chunk_size) + 1
        page_text = "\n".join(paragraphs[i:i + chunk_size])
        pages.append((page_num, page_text))
    return pages


def _parse_markdown(filepath: str) -> list[tuple[int, str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        md_content = f.read()
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text(separator="\n")
    return [(1, plain_text)]


def _parse_txt(filepath: str) -> list[tuple[int, str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return [(1, content)]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window word chunker."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += size - overlap
    return chunks
