"""5 difficulty levels of retrieval test cases.

Each level tests a different KIND of matching the search must handle:
  L1 exact words, L2 light rephrase, L3 full paraphrase,
  L4 negation/trick, L5 multi-criteria recall.

Run:  python scripts/retrieval_levels.py
"""
from retrieval_probe import BM25, load_chunks, tokenize

LEVELS = [
    {
        "level": 1,
        "name": "Exact words (copied straight from the rule)",
        "cases": [
            ("knee pain that limits daily activities", "POL-KNEE-MRI"),
            ("the patient has type 1 diabetes", "POL-CGM"),
            ("loss of function after surgery or injury", "POL-PT"),
        ],
    },
    {
        "level": 2,
        "name": "Light rephrase (key words still present)",
        "cases": [
            ("when is a knee MRI covered", "POL-KNEE-MRI"),
            ("coverage for a continuous glucose monitor", "POL-CGM"),
            ("physical therapy that a doctor prescribed with goals", "POL-PT"),
        ],
    },
    {
        "level": 3,
        "name": "Full paraphrase (same meaning, zero shared words)",
        "cases": [
            ("scan of the middle of the leg for a long ache that pills and stretching did not help",
             "POL-KNEE-MRI"),
            ("device that reads sugar by itself all day for someone who takes shots before meals",
             "POL-CGM"),
            ("movement training given by a professional after an operation to help the leg work again",
             "POL-PT"),
        ],
    },
    {
        "level": 4,
        "name": "Negation / exclusion (trick questions)",
        "cases": [
            ("is a knee MRI covered when the patient tried nothing before", "POL-KNEE-MRI"),
            ("glucose monitor is denied when the patient does not have diabetes", "POL-CGM"),
        ],
    },
]


def run() -> None:
    chunks = load_chunks()
    bm25 = BM25([c["text"] for c in chunks])
    print(f"Loaded {len(chunks)} chunks from {len({c['doc'] for c in chunks})} docs.\n")

    summary = []
    for lv in LEVELS:
        print("#" * 72)
        print(f"# LEVEL {lv['level']} — {lv['name']}")
        print("#" * 72)
        for q, expected in lv["cases"]:
            scores = bm25.score(tokenize(q))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            top_idx = order[0] if order else None
            top_doc = chunks[top_idx]["doc"] if top_idx is not None else "NONE"
            top_score = scores[top_idx] if top_idx is not None else 0.0
            # A score of 0 means "no word matched at all" -> not a real find.
            ok = top_doc == expected and top_score > 0
            summary.append(ok)
            print(f"\nQ: {q}")
            print(f"   expected: {expected} | found: {top_doc} (score={top_score:.3f}) -> "
                  f"{'PASS' if ok else 'FAIL'}")
            for rank, idx in enumerate(order[:2], start=1):
                c = chunks[idx]
                print(f"      #{rank} {c['doc']} | {c['text'][:80]}")

    # Level 5 — recall: does top-3 contain ALL parts of the one correct rule?
    print("\n" + "#" * 72)
    print("# LEVEL 5 — Multi-criteria recall (need ALL parts of one rule)")
    print("#" * 72)
    q5 = "all the requirements to get a continuous glucose monitor"
    scores = bm25.score(tokenize(q5))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    top3_docs = [chunks[i]["doc"] for i in order[:3]]
    ok5 = all(d == "POL-CGM" for d in top3_docs)
    print(f"\nQ: {q5}")
    print(f"   top-3 docs: {top3_docs} -> {'PASS (all 3 are CGM)' if ok5 else 'FAIL'}")
    for rank, idx in enumerate(order[:3], start=1):
        print(f"      #{rank} {chunks[idx]['text'][:80]}")

    # Tally
    total = len(summary)
    passed = sum(summary)
    print("\n" + "=" * 72)
    print("RESULT SUMMARY")
    print("=" * 72)
    print(f"  Levels 1-4 top-1 correct: {passed}/{total}")
    print(f"  Level 5 (recall): {'PASS' if ok5 else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    run()
