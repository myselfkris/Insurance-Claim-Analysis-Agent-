"""Tiny retrieval probe — answers TWO questions:

    1. Does search find the right rule when words match?   (easy)
    2. Does it still find it when the words are different? (hard)

No LLM, no agents, no database. Just BM25 (plain Python) over the policy files.
Run:  python scripts/retrieval_probe.py
"""
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policy"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Standard BM25 ranking (same math the rank_bm25 library uses)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.tokens]
        self.avgdl = sum(self.doc_len) / len(self.docs)
        self.N = len(self.docs)

        self.df = Counter()
        for t in self.tokens:
            for w in set(t):
                self.df[w] += 1
        self.idf = {
            w: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for w, df in self.df.items()
        }

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for i, toks in enumerate(self.tokens):
            freq = Counter(toks)
            dl = self.doc_len[i]
            s = 0.0
            for q in query_tokens:
                if q not in self.idf:
                    continue
                f = freq.get(q, 0)
                idf = self.idf[q]
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append({"doc": path.stem, "chunk": i, "text": para})
    return chunks


EASY_QUERIES = [
    "knee MRI after physical therapy fails",
    "continuous glucose monitor type 1 diabetes insulin pump",
    "physical therapy after knee surgery doctor prescribed",
]

# Same meaning as the easy ones, but different words — tests meaning matching.
HARD_QUERIES = [
    "chronic knee discomfort that did not improve with pills and exercise",
    "sugar monitoring device for someone who takes insulin many times a day",
    "rehab sessions ordered by the doctor after a knee operation",
]


def main() -> None:
    chunks = load_chunks()
    bm25 = BM25([c["text"] for c in chunks])
    n_docs = len({c["doc"] for c in chunks})
    print(f"Loaded {len(chunks)} chunks from {n_docs} policy docs.\n")

    sections = [("EASY (words match)", EASY_QUERIES), ("HARD (different words)", HARD_QUERIES)]
    for label, queries in sections:
        print("#" * 70)
        print(f"# {label}")
        print("#" * 70)
        for q in queries:
            scores = bm25.score(tokenize(q))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            print("=" * 70)
            print(f"QUERY: {q}")
            for rank, idx in enumerate(order[:3], start=1):
                c = chunks[idx]
                print(f"\n  #{rank}  doc={c['doc']}  score={scores[idx]:.3f}")
                print(f"      {c['text'][:150]}")
            print()


if __name__ == "__main__":
    main()
