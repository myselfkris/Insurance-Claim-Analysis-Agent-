"""Compare search modes on the 5 difficulty levels.

Modes:
  A. raw BM25              (no cleanup)
  B. BM25 + stopword removal
  C. BM25 + stopwords + small manual synonym map

This answers one question: do we NEED real embeddings (semantic search)
to pass the paraphrase level, or can cheaper fixes get us there?

Run:  python scripts/retrieval_compare.py
"""
import math
import re
from collections import Counter

from retrieval_levels import LEVELS
from retrieval_probe import load_chunks

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "at", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "not", "did", "do", "does", "has", "have", "had", "when",
    "who", "what", "where", "why", "how", "if", "then", "than", "it", "its",
    "as", "from", "which", "after", "before", "during", "about", "into", "over",
    "under", "all", "any", "every", "itself", "given", "help", "work", "again",
}

# A small, honest set of common medical synonyms. This is MANUAL — someone has
# to list every pairing. Real embeddings learn these automatically.
SYNONYMS = {
    "sugar": ["glucose"],
    "glucose": ["sugar"],
    "diabetic": ["diabetes"],
    "diabetes": ["diabetic"],
    "shots": ["injections"],
    "injections": ["shots"],
    "ache": ["pain"],
    "pain": ["ache"],
    "discomfort": ["pain"],
    "pills": ["medicine", "anti-inflammatory"],
    "medicine": ["pills", "anti-inflammatory"],
    "operation": ["surgery"],
    "surgery": ["operation"],
    "doctor": ["professional", "physician"],
    "professional": ["doctor"],
    "training": ["therapy"],
    "stretching": ["physical therapy"],
    "device": ["monitor"],
}


def prep(text: str, stopwords: bool, synonyms: bool) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if synonyms:
        out = list(toks)
        for t in toks:
            out.extend(SYNONYMS.get(t, []))
        toks = out
    return toks


class BM25:
    def __init__(self, doc_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.doc_tokens = doc_tokens
        self.k1 = k1
        self.b = b
        self.doc_len = [len(t) for t in doc_tokens]
        self.avgdl = sum(self.doc_len) / len(doc_tokens)
        self.N = len(doc_tokens)
        df = Counter()
        for t in doc_tokens:
            for w in set(t):
                df[w] += 1
        self.idf = {w: 1 + math.log((self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def score(self, q_tokens: list[str]) -> list[float]:
        out = []
        for toks in self.doc_tokens:
            freq = Counter(toks)
            dl = len(toks)
            s = 0.0
            for q in q_tokens:
                if q not in self.idf:
                    continue
                f = freq.get(q, 0)
                s += self.idf[q] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def run_mode(stopwords: bool, synonyms: bool) -> dict:
    chunks = load_chunks()
    bm25 = BM25([prep(c["text"], stopwords, synonyms) for c in chunks])
    per_level = {}
    for lv in LEVELS:
        ok = 0
        for q, expected in lv["cases"]:
            scores = bm25.score(prep(q, stopwords, synonyms))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            top = order[0]
            hit = chunks[top]["doc"] == expected and scores[top] > 0
            if hit:
                ok += 1
        per_level[lv["level"]] = (ok, len(lv["cases"]))
    return per_level


def main() -> None:
    print("Search-mode comparison on Levels 1-4 (top-1 must be the right doc).\n")
    modes = [
        ("A. raw BM25", False, False),
        ("B. + stopwords", True, False),
        ("C. + stopwords + synonyms", True, True),
    ]
    for name, sw, sy in modes:
        res = run_mode(sw, sy)
        cells = [f"L{lv}: {res[lv][0]}/{res[lv][1]}" for lv in (1, 2, 3, 4)]
        total_ok = sum(res[lv][0] for lv in (1, 2, 3, 4))
        total = sum(res[lv][1] for lv in (1, 2, 3, 4))
        print(f"{name:28s}  {'  '.join(cells)}   => {total_ok}/{total} total")
    print()


if __name__ == "__main__":
    main()
