import os, json, tempfile
from pathlib import Path
import streamlit as st

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
import agentic

DISTRICTS = ["Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya", "Galle", "Matara",
             "Hambantota", "Jaffna", "Kilinochchi", "Mannar", "Vavuniya", "Mullaitivu", "Batticaloa",
             "Ampara", "Trincomalee", "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa",
             "Badulla", "Monaragala", "Ratnapura", "Kegalle"]

st.set_page_config(page_title="Solar PV Agentic Diagnostics", page_icon="☀️")
st.title("☀️ Solar PV Agentic Diagnostics")
st.caption("Upload anything. A Router agent inspects it and picks the right model on its own.")

with st.sidebar:
    st.header("Site")
    location = st.selectbox("District", DISTRICTS, index=DISTRICTS.index("Kilinochchi"))
    agentic.CONFIG["panels_in_series"] = st.number_input("Panels in series per string", 1, value=7)
    agentic.CONFIG["parallel_strings"] = st.number_input("Strings in parallel", 1, value=1)

files = st.file_uploader("Upload image(s) and/or CSV(s)", type=["png", "jpg", "jpeg", "csv"],
                         accept_multiple_files=True)

if st.button("Run agentic diagnosis", type="primary", disabled=not files):
    tmp = tempfile.mkdtemp()
    paths = []
    for f in files:
        p = Path(tmp) / f.name
        p.write_bytes(f.getbuffer()); paths.append(str(p))
    with st.spinner("Router agent profiling files and choosing a model..."):
        try:
            out = agentic.run_agentic(paths, location)
        except Exception as e:
            st.error(f"Agent pipeline failed: {e}"); st.stop()

    st.subheader("🧭 Routing decision")
    c1, c2, c3 = st.columns(3)
    c1.metric("Agent chose", out["agent_route"])
    c2.metric("Old if/else would pick", out["rule_route"])
    c3.metric("Agree", "✅" if out["agree"] else "❌")
    with st.expander("Agent tool-call trace", expanded=True):
        for i, t in enumerate(out["trace"], 1):
            st.markdown(f"**{i}. `{t['tool']}`**")
            st.code(json.dumps(t["args"]), language="json")
            st.code(t["result"], language="json")

    st.subheader("📝 Diagnostic ticket")
    st.markdown(out["ticket"])
