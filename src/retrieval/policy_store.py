"""Load policy + EMR documents, chunk them, and expose a hybrid retriever.

Retrieval is BM25 (+ synonyms) first; dense embeddings are fused in only when
an embedder is available. This keeps the pipeline runnable with zero heavy deps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT
from .bm25_index import BM25Index
from .embedder import Embedder
from .hybrid import cosine, rrf_fuse

_POOL = 20  # candidate pool size for rank fusion (independent of requested k)


@dataclass
class Chunk:
    source_id: str
    source_type: str
    chunk_index: int
    text: str

    @property
    def citation_key(self) -> str:
        return f"{self.source_id}-{self.chunk_index}"


class HybridRetriever:
    def __init__(self, policy_dir: Path | None = None, emr_dir: Path | None = None):
        self.policy_dir = policy_dir or PROJECT_ROOT / "data" / "policy"
        self.emr_dir = emr_dir or PROJECT_ROOT / "data" / "synthetic" / "emr"
        self.chunks: list[Chunk] = []
        self._bm25: BM25Index | None = None
        self._embedder = Embedder()
        self._build()

    def _load(self, directory: Path, source_type: str) -> list[Chunk]:
        if not directory.exists():
            return []
        out = []
        for path in sorted(directory.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                out.append(Chunk(source_id=path.stem, source_type=source_type, chunk_index=i, text=para))
        return out

    def _build(self):
        self.chunks = self._load(self.policy_dir, "policy") + self._load(self.emr_dir, "emr")
        self._bm25 = BM25Index([c.text for c in self.chunks])
        # Dense embeddings are fused in a later phase; BM25 carries the demo today.
        self._vectors = self._embedder.embed([c.text for c in self.chunks]) if self._embedder.available else None

    @property
    def using_embeddings(self) -> bool:
        return self._vectors is not None

    def search(self, query: str, k: int = 5, source_type: str | None = None) -> list[dict]:
        bm25_hits = self._bm25.search(query, k=_POOL)
        if self._vectors is not None:
            qvec = self._embedder.embed([query])
            if qvec:
                sims = sorted(
                    ((cosine(qvec[0], v), i) for i, v in enumerate(self._vectors)),
                    key=lambda x: -x[0],
                )
                emb_hits = [(i, s) for s, i in sims if s > 0.0][:_POOL]
                hits = rrf_fuse([bm25_hits, emb_hits])
            else:
                hits = [(i, s) for i, s in bm25_hits]
        else:
            hits = [(i, s) for i, s in bm25_hits]
        out = []
        for idx, score in hits:
            c = self.chunks[idx]
            if source_type and c.source_type != source_type:
                continue
            out.append({
                "citation_key": c.citation_key,
                "source_id": c.source_id,
                "source_type": c.source_type,
                "text": c.text,
                "score": round(score, 4),
            })
            if len(out) >= k:
                break
        return out
