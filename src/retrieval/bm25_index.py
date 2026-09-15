"""BM25 keyword index with stopword removal + optional synonym expansion.

Pure-Python implementation (no external deps) so the retriever runs even
before heavy packages are installed. The `rank_bm25` library is equivalent.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "at", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "not", "did", "do", "does", "has", "have", "had", "when",
    "who", "what", "where", "why", "how", "if", "then", "than", "it", "its",
    "as", "from", "which", "after", "before", "during", "about", "into", "over",
    "under", "all", "any", "every", "itself", "given", "help", "work", "again",
}

SYNONYMS = {
    "sugar": ["glucose"], "glucose": ["sugar"],
    "diabetic": ["diabetes"], "diabetes": ["diabetic"],
    "shots": ["injections"], "injections": ["shots"],
    "ache": ["pain"], "pain": ["ache"], "discomfort": ["pain"],
    "pills": ["medicine", "anti-inflammatory"],
    "medicine": ["pills", "anti-inflammatory"],
    "operation": ["surgery"], "surgery": ["operation"],
    "doctor": ["professional", "physician"], "professional": ["doctor"],
    "training": ["therapy"], "stretching": ["physical therapy"],
    "device": ["monitor"],
}


def tokenize(text: str, expand_synonyms: bool = True) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
    if expand_synonyms:
        out = list(toks)
        for t in toks:
            out.extend(SYNONYMS.get(t, []))
        toks = out
    return toks


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.texts = texts
        self.tokens = [tokenize(t) for t in texts]
        self.k1 = k1
        self.b = b
        self.doc_len = [len(t) for t in self.tokens]
        self.avgdl = sum(self.doc_len) / len(self.tokens) if self.tokens else 1.0
        self.N = len(self.tokens)
        df = Counter()
        for t in self.tokens:
            for w in set(t):
                df[w] += 1
        self.idf = {w: 1 + math.log((self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def score(self, query_tokens: list[str]) -> list[float]:
        out = []
        for toks in self.tokens:
            freq = Counter(toks)
            dl = len(toks)
            s = 0.0
            for q in query_tokens:
                if q not in self.idf:
                    continue
                f = freq.get(q, 0)
                s += self.idf[q] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores = self.score(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in order[:k] if scores[i] > 0]
