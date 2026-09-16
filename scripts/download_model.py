"""Download the semantic embedding model and sanity-check it.

Run:  python scripts/download_model.py
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings


def sim(model, a: str, b: str) -> float:
    va = model.encode([a])[0]
    vb = model.encode([b])[0]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    return dot / (na * nb)


def main() -> None:
    print(f"Loading model: {settings.embeddings_model} (HF_HOME={os.environ.get('HF_HOME')})")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embeddings_model)
    print("Model ready.\n")

    pairs = [
        ("knee pain", "chronic ache in the joint below the thigh"),
        ("knee pain", "continuous glucose monitor"),
        ("sugar monitoring device", "continuous glucose monitor"),
        ("physical therapy", "movement training by a professional after surgery"),
    ]
    for a, b in pairs:
        print(f"  similarity {sim(model, a, b):.3f}   {a!r}  vs  {b!r}")


if __name__ == "__main__":
    main()
