"""Insurance Claim Analysis Agent — Streamlit UI (Analyze + Admin).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import store
from src.config import PROJECT_ROOT, settings
from src.graph import build_graph

st.set_page_config(page_title="Insurance Claim Analysis Agent", layout="wide")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def list_files(subdir: str) -> list[str]:
    d = PROJECT_ROOT / "data" / "synthetic" / subdir
    return sorted(p.name for p in d.glob("*.txt")) if d.exists() else []


def run_screen() -> None:
    st.header("Analyze a rejected claim")
    denials = list_files("denials")
    emrs = list_files("emr")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Denial")
        choice = st.selectbox("Sample denial", denials, index=0) if denials else None
        loaded = load_text(PROJECT_ROOT / "data" / "synthetic" / "denials" / choice) if choice else ""
        denial = st.text_area("Denial text", value=loaded, height=180, key=f"denial_{choice}")
    with c2:
        st.subheader("Patient records")
        choice2 = st.selectbox("Sample records", emrs, index=0) if emrs else None
        loaded2 = load_text(PROJECT_ROOT / "data" / "synthetic" / "emr" / choice2) if choice2 else ""
        case = st.text_area("Records text", value=loaded2, height=180, key=f"case_{choice2}")

    st.caption(f"LLM provider: **{settings.llm_provider}** (set LLM_PROVIDER in .env)")

    if st.button("Analyze", type="primary"):
        with st.spinner("research → evaluate → decide → verify …"):
            graph = build_graph()
            final = graph.invoke({"denial_text": denial, "case_text": case})
        case_id = store.save_case(final)
        st.session_state["last_final"] = final
        st.session_state["last_case_id"] = case_id
        st.success(f"Saved as case #{case_id} (pending approval)")

    final = st.session_state.get("last_final")
    if final:
        decision = final.get("decision")
        output = final.get("output") or {}
        verdict = final.get("verdict") or {}
        evaluation = final.get("evaluation") or {}
        spec = final.get("criteria_spec") or {}

        st.subheader(f"Decision: **{decision}**  (rule: {spec.get('expression', '')})")
        rows = [{"criterion": v["criterion_id"], "status": v["status"], "reasoning": v["reasoning"]}
                for v in evaluation.get("criteria", [])]
        st.table(rows)

        st.subheader("Output")
        text = output.get("letter") or output.get("explanation") or output.get("request") or ""
        st.text_area("Output", value=text, height=260, key="output_result")

        m1, m2 = st.columns(2)
        m1.metric("Verify", verdict.get("verdict"))
        m2.metric("Confidence", final.get("confidence_score"))

        with st.expander("Evidence used"):
            for e in final.get("retrieved_evidence", []):
                st.write(f"**[{e['citation_key']}]** ({e['source_type']}) — {e['text'][:200]}")


def admin_screen() -> None:
    st.header("Admin — review & approve")
    pending = store.list_cases(status="pending_approval")
    if not pending:
        st.info("No pending cases. Analyze a claim first.")
        return

    labels = {c["id"]: f"Case #{c['id']} ({c['decision']}, conf {round(c.get('confidence') or 0, 2)})"
              for c in pending}
    case_id = st.selectbox("Pending case", list(labels.keys()), format_func=lambda i: labels[i])
    case = next(c for c in pending if c["id"] == case_id)

    text = st.text_area("Output (editable)", value=case.get("output_text") or "", height=260, key=f"admin_{case_id}")

    b1, b2 = st.columns(2)
    if b1.button("Approve", type="primary"):
        store.approve(case_id, edited_text=text)
        st.success(f"Case #{case_id} approved.")
        st.rerun()
    if b2.button("Reject"):
        store.reject(case_id, edited_text=text)
        st.warning(f"Case #{case_id} rejected.")
        st.rerun()

    if st.checkbox("Show all cases"):
        st.table([{"id": c["id"], "decision": c["decision"], "status": c["status"],
                   "confidence": round(c.get("confidence") or 0, 2)} for c in store.list_cases()])


st.title("Insurance Claim Analysis Agent")
st.caption("Honest triage — appeal, uphold, or request more info. The AI prepares; a human decides.")
t1, t2 = st.tabs(["Analyze", "Admin"])
with t1:
    run_screen()
with t2:
    admin_screen()
