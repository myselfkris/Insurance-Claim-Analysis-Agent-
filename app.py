"""PriorAuth Crusher — Streamlit UI (Run + Admin).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

# Allow running as `streamlit run app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import store
from src.config import PROJECT_ROOT, settings
from src.graph import build_graph

st.set_page_config(page_title="PriorAuth Crusher", layout="wide")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def list_files(subdir: str) -> list[str]:
    d = PROJECT_ROOT / "data" / "synthetic" / subdir
    return sorted(p.name for p in d.glob("*.txt")) if d.exists() else []


def run_screen() -> None:
    st.header("Run an appeal")
    denials = list_files("denials")
    emrs = list_files("emr")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Denial (the 'no' letter)")
        denial_choice = st.selectbox("Sample denial", denials, index=0) if denials else None
        loaded_denial = load_text(PROJECT_ROOT / "data" / "synthetic" / "denials" / denial_choice) if denial_choice else ""
        denial = st.text_area("Denial text", value=loaded_denial, height=180, key=f"denial_{denial_choice}")

    with col2:
        st.subheader("Case / EMR (the patient's note)")
        emr_choice = st.selectbox("Sample case", emrs, index=0) if emrs else None
        loaded_case = load_text(PROJECT_ROOT / "data" / "synthetic" / "emr" / emr_choice) if emr_choice else ""
        case = st.text_area("Case text", value=loaded_case, height=180, key=f"case_{emr_choice}")

    st.caption(f"LLM provider: **{settings.llm_provider}**  (set LLM_PROVIDER in .env — use 'mock' for no key)")

    if st.button("Run agents", type="primary"):
        with st.spinner("research → draft → verify …"):
            graph = build_graph()
            final = graph.invoke({"denial_text": denial, "case_text": case})
        case_id = store.save_case(final)
        st.session_state["last_final"] = final
        st.session_state["last_case_id"] = case_id
        st.success(f"Saved as case #{case_id} (status: pending approval)")

    final = st.session_state.get("last_final")
    if final:
        letter = (final.get("letter_draft") or {}).get("letter", "")
        verdict = final.get("verdict") or {}
        st.subheader("Appeal letter (draft)")
        st.text_area("Letter", value=letter, height=300, key="letter_result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", verdict.get("verdict"))
        c2.metric("Confidence", final.get("confidence_score"))
        c3.metric("Denial type", (final.get("plan") or {}).get("denial_type"))

        with st.expander("Evidence used"):
            for e in final.get("retrieved_evidence", []):
                st.write(f"**[{e['citation_key']}]** ({e['source_type']}) — {e['text'][:220]}")
        with st.expander("Audit trail (sign-off proof)"):
            st.json(final.get("audit_trail", []))


def admin_screen() -> None:
    st.header("Admin — review & approve")
    pending = store.list_cases(status="pending_approval")
    if not pending:
        st.info("No pending cases. Run an appeal first.")
        return

    labels = {c["id"]: f"Case #{c['id']}  (confidence {round(c.get('confidence') or 0.0, 2)})" for c in pending}
    case_id = st.selectbox("Pending case", list(labels.keys()), format_func=lambda i: labels[i])
    case = next(c for c in pending if c["id"] == case_id)

    letter = st.text_area("Appeal letter (editable)", value=case.get("letter") or "", height=300, key=f"admin_{case_id}")

    b1, b2 = st.columns(2)
    if b1.button("Approve", type="primary"):
        store.approve(case_id, edited_letter=letter)
        st.success(f"Case #{case_id} approved.")
        st.rerun()
    if b2.button("Reject"):
        store.reject(case_id, edited_letter=letter)
        st.warning(f"Case #{case_id} rejected.")
        st.rerun()

    if st.checkbox("Show all cases (history)"):
        rows = [{"id": c["id"], "status": c["status"], "confidence": round(c.get("confidence") or 0.0, 2)}
                for c in store.list_cases()]
        st.table(rows)


st.title("PriorAuth Crusher")
st.caption("AI prepares a cited appeal letter; the human makes the final decision.")

tab_run, tab_admin = st.tabs(["Run", "Admin"])
with tab_run:
    run_screen()
with tab_admin:
    admin_screen()
