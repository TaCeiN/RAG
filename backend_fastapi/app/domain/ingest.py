from __future__ import annotations

import re
import zipfile
from html import unescape
from pathlib import Path
from xml.etree import ElementTree as ET

from app.core.config import settings


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix == ".docx":
        return _extract_docx_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    min_size = min(settings.chunk_min_size, max(1, chunk_size // 2))

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = _normalize_text(text)
    stripped = normalized.strip()
    if not stripped:
        return []

    blocks = _semantic_blocks(stripped)
    if len(blocks) <= 1:
        return _merge_small_chunks(_slice_with_overlap(stripped, chunk_size=chunk_size, overlap=overlap), min_size=min_size)

    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for block in blocks:
        block_size = _token_count(block)
        if block_size > chunk_size:
            if current:
                chunks.append(current)
                current = []
                current_size = 0
            chunks.extend([[part] for part in _slice_with_overlap(block, chunk_size=chunk_size, overlap=overlap)])
            continue

        if current and (current_size + block_size > chunk_size):
            # Reduce tiny tails: allow soft overflow up to 20% when current chunk is too short.
            if current_size < min_size and (current_size + block_size <= int(chunk_size * 1.2)):
                current.append(block)
                current_size += block_size
                continue
            chunks.append(current)
            current = [block]
            current_size = block_size
        else:
            current.append(block)
            current_size += block_size

    if current:
        chunks.append(current)

    rendered = ["\n\n".join(parts).strip() for parts in chunks if parts]
    return _merge_small_chunks(rendered, min_size=min_size)


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def _semantic_blocks(text: str) -> list[str]:
    lines = text.split("\n")
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(" ".join(part.strip() for part in paragraph if part.strip()).strip())
            paragraph.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_paragraph()
            continue
        if _looks_like_heading(line):
            flush_paragraph()
            blocks.append(line)
            continue
        paragraph.append(line)

    flush_paragraph()
    return [block for block in blocks if block]


def _looks_like_heading(line: str) -> bool:
    if line.startswith("#"):
        return True
    if re.match(r"^\d+(\.\d+)*[.)]\s+", line):
        return True
    if len(line) <= 80 and line.endswith(":"):
        return True
    letters = [ch for ch in line if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters) and len(letters) >= 12:
        return True
    return False


def _slice_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces = re.findall(r"\s*\S+", text, flags=re.UNICODE)
    if not pieces:
        return []
    if len(pieces) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(pieces):
        end = min(len(pieces), start + chunk_size)
        chunk = "".join(pieces[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(pieces):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _merge_small_chunks(chunks: list[str], min_size: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        clean = chunk.strip()
        if not clean:
            continue
        if merged and _token_count(clean) < min_size:
            merged[-1] = f"{merged[-1]}\n\n{clean}".strip()
        else:
            merged.append(clean)
    return merged


def _token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = root.find(f".//{namespace}body")
    text_parts: list[str] = []
    children = list(body) if body is not None else list(root)

    for element in children:
        if element.tag == f"{namespace}tbl":
            rows: list[str] = []
            for row in element.findall(f".//{namespace}tr"):
                cells: list[str] = []
                for cell in row.findall(f"{namespace}tc"):
                    cell_text = _element_text(cell)
                    if cell_text:
                        cells.append(cell_text)
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                text_parts.append("\n".join(rows))
            continue

        if element.tag == f"{namespace}p":
            paragraph = _element_text(element)
            if paragraph:
                text_parts.append(paragraph)
            continue

        fallback = _element_text(element)
        if fallback:
            text_parts.append(fallback)

    joined = "\n".join(text_parts)
    joined = unescape(joined)
    return _normalize_text(joined).strip()


def _element_text(element: ET.Element) -> str:
    parts = [node.text for node in element.iter() if node.tag.endswith("}t") and node.text]
    return _normalize_text(" ".join(parts)).strip()


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None

    if PdfReader is not None:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(text)
        if pages:
            return _normalize_text("\n\n".join(pages)).strip()

    # Fallback path for environments without pypdf.
    raw_text = path.read_bytes().decode("latin-1", errors="ignore")
    stream_chunks = re.findall(r"stream\s*(.*?)\s*endstream", raw_text, flags=re.S)
    source = "\n".join(stream_chunks) if stream_chunks else raw_text
    matches = re.findall(r"\(([^()]*)\)\s*T[Jj]", source, flags=re.S)
    if not matches:
        matches = re.findall(r"\(([^()]*)\)", source, flags=re.S)
    cleaned = [match.strip() for match in matches if match.strip()]
    return _normalize_text("\n".join(cleaned)).strip()
