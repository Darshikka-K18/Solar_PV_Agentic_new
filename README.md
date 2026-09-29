# Solar PV Agentic Diagnostics (standalone)
Separate from Solar_PV_App; nothing there is modified.

Setup
1. Copy (don't move) the `models/` folder from Solar_PV_App into this folder (or set MODELS_DIR).
2. pip install -r requirements.txt
3. Set GROQ_API_KEY (env var or .streamlit/secrets.toml)
4. streamlit run app.py

Each run appends agent-vs-old-rule routing to evaluation_log.csv (routing accuracy for the report).
