"""Run the honest-triage pipeline end-to-end.

Usage:
  python scripts/run_pipeline.py                 # appeal sample (all criteria MET)
  python scripts/run_pipeline.py uphold          # denial-correct sample (NOT MET)
  python scripts/run_pipeline.py more_info       # missing-doc sample (UNVERIFIED)

Run with LLM_PROVIDER=mock for no API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store
from src.config import PROJECT_ROOT
from src.graph import build_graph

SAMPLES = {
    "appeal": ("DENIAL-KNEE-001.txt", "EMR-KNEE-001.txt"),
    "uphold": ("DENIAL-KNEE-002.txt", "EMR-KNEE-002.txt"),
    "more_info": ("DENIAL-KNEE-003.txt", "EMR-KNEE-003.txt"),
}


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "appeal"
    denial_name, emr_name = SAMPLES.get(scenario, SAMPLES["appeal"])
    denial = (PROJECT_ROOT / "data" / "synthetic" / "denials" / denial_name).read_text(encoding="utf-8")
    case = (PROJECT_ROOT / "data" / "synthetic" / "emr" / emr_name).read_text(encoding="utf-8")

    graph = build_graph()
    final = graph.invoke({"denial_text": denial, "case_text": case})

    decision = final.get("decision")
    output = final.get("output") or {}
    verdict = final.get("verdict") or {}
    evaluation = final.get("evaluation") or {}
    spec = final.get("criteria_spec") or {}

    print("=" * 72)
    print(f"DECISION: {decision}     (expression: {spec.get('expression')})")
    print("=" * 72)
    for v in evaluation.get("criteria", []):
        print(f"  [{v['criterion_id']}] {v['status']:<11} {v['reasoning'][:90]}")
    print("\n" + "-" * 72)
    print("OUTPUT:")
    print(output.get("letter") or output.get("explanation") or output.get("request") or "(none)")
    print("\n" + "=" * 72)
    print(f"VERIFY: {verdict.get('verdict')} | reason: {verdict.get('reason')} | "
          f"confidence: {final.get('confidence_score')}")
    print("=" * 72)

    case_id = store.save_case(final)
    print(f"\nSaved as case #{case_id} (status: pending_approval)")


if __name__ == "__main__":
    main()
