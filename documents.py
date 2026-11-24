"""Load the knowledge corpus (markdown + YAML front matter) and chunk it heading-aware."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from surgeshield.config import path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def source_url(self) -> str:
        return str(self.meta.get("source_url", "")).split("#")[0].strip()


def load_documents(root: Path | None = None) -> list[Document]:
    root = root or path("knowledge")
    docs = []
    for p in sorted(root.rglob("*.md")):
        raw = p.read_text()
        meta: dict = {}
        if raw.startswith("---"):
            _, fm, body = raw.split("---", 2)
            meta = yaml.safe_load(fm) or {}
        else:
            body = raw
        meta.setdefault("doc_type", p.parent.name)
        docs.append(Document(p.stem, meta.get("title", p.stem), body.strip(), meta))
    return docs


def chunk_document(doc: Document, size: int = 700, overlap: int = 120) -> list[Chunk]:
    """Split on markdown headings, then window long sections; prefix each chunk with its heading path
    (contextual chunking) so retrieval works on short sections."""
    sections = re.split(r"(?m)^(#{1,3} .+)$", doc.text)
    chunks: list[Chunk] = []
    heading = doc.title
    buf: list[tuple[str, str]] = []
    if sections and not sections[0].startswith("#"):
        buf.append((heading, sections[0]))
        sections = sections[1:]
    for i in range(0, len(sections) - 1, 2):
        buf.append((sections[i].lstrip("# ").strip(), sections[i + 1]))
    for sec, body in buf:
        body = body.strip()
        if not body:
            continue
        start = 0
        while start < len(body):
            piece = body[start:start + size]
            cid = hashlib.md5(f"{doc.doc_id}:{sec}:{start}".encode()).hexdigest()[:10]
            chunks.append(Chunk(cid, doc.doc_id, doc.title, sec, f"{doc.title} > {sec}\n{piece}", doc.meta))
            if start + size >= len(body):
                break
            start += size - overlap
    return chunks
