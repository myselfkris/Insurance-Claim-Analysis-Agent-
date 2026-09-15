"""Phase 2 walking skeleton: run the full pipeline end-to-end.

Run:  $env:LLM_PROVIDER='mock'; python scripts/run_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROJECT_ROOT
from src.graph import build_graph
from src import store


def main() -> None:
    denial_path = PROJECT_ROOT / "data" / "synthetic" / "denials" / "DENIAL-KNEE-001.txt"
    emr_path = PROJECT_ROOT / "data" / "synthetic" / "emr" / "EMR-KNEE-001.txt"
    denial = denial_path.read_text(encoding="utf-8") if denial_path.exists() else "No denial file found."
    case = emr_path.read_text(encoding="utf-8") if emr_path.exists() else "No case file found."

    graph = build_graph()
    final = graph.invoke({"denial_text": denial, "case_text": case})

    letter = (final.get("letter_draft") or {}).get("letter", "")
    verdict = final.get("verdict") or {}
    print("=" * 70)
    print("APPEAL LETTER (draft)")
    print("=" * 70)
    print(letter)
    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict.get('verdict')} | confidence: {final.get('confidence_score')}")
    print(f"DENIAL TYPE: {(final.get('plan') or {}).get('denial_type')}")
    print(f"EVIDENCE USED: {[e['citation_key'] for e in final.get('retrieved_evidence', [])]}")
    print("=" * 70)

    case_id = store.save_case(final)
    print(f"\nSaved as case #{case_id} with status 'pending_approval' (data/cases.db)")


if __name__ == "__main__":
    main()
