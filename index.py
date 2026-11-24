"""Hybrid retrieval: BM25 (lexical) + dense embeddings, fused with Reciprocal Rank Fusion,
with metadata filters and MMR diversification.

Dense backends (auto-selected):
  - sentence-transformers (if installed and SURGESHIELD_EMBEDDER=st): all-MiniLM-L6-v2
  - LSA (TF-IDF -> TruncatedSVD) : zero-download fallback so the system runs fully offline
Swap in pgvector / Qdrant by implementing `VectorStore.search` - the interface is the same.
"""
from __future__ import annotations

import logging
import os
import pickle
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from surgeshield.config import get_settings, path
from surgeshield.rag.documents import Chunk, chunk_document, load_documents

log = logging.getLogger(__name__)
_tok = re.compile(r"[a-z0-9$.]+")


def tokenize(text: str) -> list[str]:
    return _tok.findall(text.lower())


class DenseEncoder:
    def __init__(self, texts: list[str]):
        self.kind = "lsa"
        if os.environ.get("SURGESHIELD_EMBEDDER") == "st":
            try:
                from sentence_transformers import SentenceTransformer
                self.st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.kind = "sentence-transformers"
                return
            except Exception as e:  # pragma: no cover
                log.warning("sentence-transformers unavailable (%s) - using LSA", e)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(96, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.kind == "sentence-transformers":
            return np.asarray(self.st.encode(texts, normalize_embeddings=True))
        v = self.svd.transform(self.tfidf.transform(texts))
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.encoder = DenseEncoder(texts)
        self.emb = self.encoder.encode(texts)

    def search(self, query: str, k: int = 5, filters: dict | None = None, rrf_k: int = 60,
               mmr_lambda: float = 0.7) -> list[Hit]:
        mask = np.ones(len(self.chunks), bool)
        for key, val in (filters or {}).items():
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            mask &= np.array([c.meta.get(key) in vals for c in self.chunks])
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return []
        bm = self.bm25.get_scores(tokenize(query))[idx]
        q = self.encoder.encode([query])[0]
        dn = self.emb[idx] @ q
        bm_rank = {int(idx[i]): r for r, i in enumerate(np.argsort(-bm))}
        dn_rank = {int(idx[i]): r for r, i in enumerate(np.argsort(-dn))}
        fused = {i: 1 / (rrf_k + bm_rank[i]) + 1 / (rrf_k + dn_rank[i]) for i in map(int, idx)}
        cand = sorted(fused, key=fused.get, reverse=True)[: max(k * 4, 10)]
        # MMR: balance relevance and diversity so one long doc doesn't crowd out others
        chosen: list[int] = []
        while cand and len(chosen) < k:
            def mmr(i: int) -> float:
                div = max((float(self.emb[i] @ self.emb[j]) for j in chosen), default=0.0)
                return mmr_lambda * fused[i] * 60 - (1 - mmr_lambda) * div
            best = max(cand, key=mmr)
            chosen.append(best)
            cand.remove(best)
        return [Hit(self.chunks[i], fused[i], bm_rank[i], dn_rank[i]) for i in chosen]

    def add_document(self, title: str, text: str, meta: dict) -> None:
        """Online learning loop: postmortems are appended to the index after each incident."""
        from surgeshield.rag.documents import Document
        doc = Document(meta.get("doc_id", title.lower().replace(" ", "_")), title, text, meta)
        cfg = get_settings().rag
        self.chunks.extend(chunk_document(doc, cfg.chunk_size, cfg.chunk_overlap))
        self.__init__(self.chunks)


def build_index(save: bool = True) -> HybridIndex:
    cfg = get_settings().rag
    chunks = [c for d in load_documents() for c in chunk_document(d, cfg.chunk_size, cfg.chunk_overlap)]
    idx = HybridIndex(chunks)
    if save:
        p = path(cfg.index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(idx, f)
    log.info("indexed %d chunks (dense=%s)", len(chunks), idx.encoder.kind)
    return idx


_index: HybridIndex | None = None


def get_index() -> HybridIndex:
    global _index
    if _index is None:
        p = path(get_settings().rag.index_path)
        _index = pickle.load(open(p, "rb")) if p.exists() else build_index()
    return _index
