"""Test the hybrid retriever (BM25 + embeddings) on the 5 difficulty levels.

Run:  python scripts/test_hybrid.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_levels import LEVELS
from src.retrieval.policy_store import HybridRetriever


def main() -> None:
    r = HybridRetriever()
    print(f"Hybrid retriever built. embeddings active = {r.using_embeddings}\n")

    total_ok = 0
    total = 0
    for lv in LEVELS:
        ok = 0
        print(f"# LEVEL {lv['level']} — {lv['name']}")
        for q, expected in lv["cases"]:
            hits = r.search(q, k=1)
            top = hits[0]["source_id"] if hits else "NONE"
            good = top == expected
            ok += good
            total_ok += good
            total += 1
            print(f"   {'PASS' if good else 'FAIL'}  {q!r}  -> {top}")
        print(f"   => {ok}/{len(lv['cases'])}\n")

    # Level 5 recall
    q5 = "all the requirements to get a continuous glucose monitor"
    hits = r.search(q5, k=3)
    top3 = [h["source_id"] for h in hits]
    ok5 = all(d == "POL-CGM" for d in top3)
    print(f"# LEVEL 5 — recall: {q5!r} -> {top3}  {'PASS' if ok5 else 'FAIL'}\n")

    print("=" * 70)
    print(f"SUMMARY: {total_ok}/{total} top-1 correct (levels 1-4)")
    print("=" * 70)

    print("\n=== DETAIL: what the hybrid returns for the hard paraphrase queries ===\n")
    l3 = next(lv for lv in LEVELS if lv["level"] == 3)
    for q, expected in l3["cases"]:
        print(f"QUERY: {q}")
        for hit in r.search(q, k=3):
            print(f"   [{hit['citation_key']}] ({hit['score']:.4f}) {hit['text'][:90]}")
        print()


if __name__ == "__main__":
    main()
