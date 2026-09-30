import os, tempfile
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
    series = st.number_input("Panels in series per string", 1, value=7)
    parallel = st.number_input("Strings in parallel", 1, value=1)

agentic.CONFIG["panels_in_series"] = int(series)
agentic.CONFIG["parallel_strings"] = int(parallel)

string_config = agentic.inf.resolve_string_config(int(parallel))
st.caption(f"Sensor CSVs use the **{string_config}** model "
           f"(trained on a {1 if string_config == '1-string' else 3}-parallel-string reference).")

with st.expander("Using a different panel model? (optional)"):
    st.caption("Leave any field at 0.0 to assume the reference panel "
               "(Voc 47.42V, Isc 15A, Vmp 39.51V, Imp 14.17A).")
    labels = {"Voc": "Voc (V)", "Isc": "Isc (A)", "Vmp": "Vmp (V)", "Imp": "Imp (A)"}
    specs, cols = {}, st.columns(4)
    for col, (field, label) in zip(cols, labels.items()):
        with col:
            val = st.number_input(label, min_value=0.0, value=0.0, format="%.2f", key=f"panel_{field}")
            if val > 0:
                specs[field] = val
agentic.CONFIG["panel_specs"] = specs

files = st.file_uploader("Upload image(s) and/or CSV(s)", type=["png", "jpg", "jpeg", "csv"],
                         accept_multiple_files=True)


def show_summary(summary):
    if not summary:
        st.warning("The agent did not run any model.")
        return
    r = summary["result"]
    st.subheader("🔎 Model result")
    st.markdown(f"**Model used:** {summary['model']}")
    if "error" in r:
        st.error(r["error"]); return

    if summary["tool"] == "run_fusion":
        c1, c2, c3 = st.columns(3)
        c1.metric("Fused finding", r.get("fused_label") or "No shared evidence",
                  f"{r['fused_confidence_pct']}%" if r.get("fused_confidence_pct") else None)
        c2.metric("Sensor (RF)", r["rf_detection"], f"Shading {r['rf_shading_probability']*100:.1f}%", delta_color="off")
        c3.metric("Thermal CNN", r["thermal_detection"],
                  f"Shadowing {r['thermal_shadowing_probability']*100:.1f}%", delta_color="off")
        st.caption(r.get("explanation", ""))
    elif summary["tool"] == "run_lstm_model":
        c1, c2, c3 = st.columns(3)
        c1.metric("Detection", r["detection"])
        c2.metric("Current health", r["current_health"], f"threshold {r['threshold']}", delta_color="off")
        c3.metric("Predicted failure", r["failure_date"] or "N/A")
        if r.get("failure_already_elapsed"):
            st.warning("The projected failure date has already passed. Inspect the array now.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Detection", r["detection"])
        conf = r["confidence"]
        c2.metric("Confidence", f"{conf}%" if isinstance(conf, (int, float)) else conf)


if st.button("Run agentic diagnosis", type="primary", disabled=not files):
    tmp = tempfile.mkdtemp()
    paths = []
    for f in files:
        p = Path(tmp) / f.name
        p.write_bytes(f.getbuffer()); paths.append(str(p))
    with st.spinner("Router agent choosing a model and writing the ticket..."):
        try:
            out = agentic.run_agentic(paths, location)
        except Exception as e:
            st.error(f"Agent pipeline failed: {e}"); st.stop()

    show_summary(out["summary"])
    st.subheader("📝 Diagnostic ticket")
    st.markdown(out["ticket"])
