"""Agentic layer: an LLM Router agent profiles the upload(s) and chooses the
model tool itself (no if/else routing), then a Diagnostic agent writes the ticket."""
import os, json
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

import inference as inf
import fusion
from tools import get_coordinates_for_location, fetch_weather_data

CONFIG = {"panels_in_series": inf.REFERENCE_PANELS_IN_SERIES, "parallel_strings": 1, "panel_specs": None}
TRACE = []  # every tool call the agents make: {tool, args, result}

MODEL_TOOLS = {"run_rgb_model": "image", "run_thermal_model": "image",
               "run_rf_model": "rf", "run_lstm_model": "lstm", "run_fusion": "fusion"}


def _clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_debug")}

def _traced(name, args, fn):
    try:
        res = fn()
        out = json.dumps(_clean(res) if isinstance(res, dict) else res, default=str)
    except Exception as e:
        out = json.dumps({"error": str(e)})
    TRACE.append({"tool": name, "args": args, "result": out})
    return out


# ------------------------------- TOOLS -------------------------------------
@tool("profile_file")
def profile_file(file_path: str) -> str:
    """Inspect an uploaded file and return facts about it (no prediction).
    ALWAYS call this first for every file. Images: size, colour stats.
    CSVs: columns and row count."""
    def go():
        p = Path(file_path); ext = p.suffix.lower(); info = {"file": p.name, "extension": ext}
        if ext in (".jpg", ".jpeg", ".png"):
            img = Image.open(p).convert("RGB")
            hsv = np.array(img.convert("HSV")).astype(float) / 255
            info.update(kind="image", size=list(img.size),
                        mean_saturation=round(float(hsv[..., 1].mean()), 3),
                        hue_std=round(float(hsv[..., 0].std()), 3),
                        mean_brightness=round(float(hsv[..., 2].mean()), 3),
                        hint="Thermal/IR images are false-colour palettes (very high saturation, wide hue "
                             "spread) or near-grayscale; ordinary RGB panel photos have natural colours.")
        elif ext == ".csv":
            df = pd.read_csv(p)
            info.update(kind="csv", n_rows=len(df), columns=list(df.columns),
                        has_Health_Indicator="Health_Indicator" in df.columns,
                        has_timestamp="timestamp" in df.columns)
            if len(df) == 1:
                info["row"] = df.iloc[0].to_dict()
        else:
            info["kind"] = "unsupported"
        return info
    return _traced("profile_file", {"file_path": file_path}, go)

@tool("run_rgb_model")
def run_rgb_model(file_path: str) -> str:
    """Use ONLY for ordinary visible-light (RGB) photos of a panel. Classifies
    Bird_drop, Clean, Dusty, Electrical_damage, Physical_damage."""
    return _traced("run_rgb_model", {"file_path": file_path}, lambda: inf.run_cnn_inference(file_path))

@tool("run_thermal_model")
def run_thermal_model(file_path: str) -> str:
    """Use ONLY for infrared/thermal images (false-colour heat maps). Detects
    Cracking, Diode, Shadowing."""
    return _traced("run_thermal_model", {"file_path": file_path}, lambda: inf.run_thermal_cnn_inference(file_path))

@tool("run_rf_model")
def run_rf_model(file_path: str) -> str:
    """Use ONLY for a CSV with exactly ONE row of electrical readings (Voc_V, Isc_A,
    Vmp_V, Imp_A, Pmax_W, Temp_C, Irr_Wm2...). Detects Normal, Shading, Short,
    Connector, OC. Array wiring is applied automatically."""
    def go():
        c = CONFIG
        return inf.run_rf_inference(file_path, inf.resolve_string_config(c["parallel_strings"]),
                                    c["panels_in_series"], c["parallel_strings"], c["panel_specs"] or None)
    return _traced("run_rf_model", {"file_path": file_path}, go)

@tool("run_lstm_model")
def run_lstm_model(file_path: str) -> str:
    """Use ONLY for a CSV with MANY rows containing a Health_Indicator column (time
    series). Projects remaining useful life / failure date."""
    return _traced("run_lstm_model", {"file_path": file_path}, lambda: inf.run_lstm_inference(file_path))

@tool("run_fusion")
def run_fusion(image_path: str, csv_path: str) -> str:
    """Use when BOTH a thermal image AND a single-row sensor CSV of the same panel
    are provided. Runs both models and fuses the result."""
    def go():
        c = CONFIG; sc = inf.resolve_string_config(c["parallel_strings"])
        rf = inf.run_rf_inference(csv_path, sc, c["panels_in_series"], c["parallel_strings"], c["panel_specs"] or None)
        th = inf.run_thermal_cnn_inference(image_path)
        return fusion.fuse_predictions(rf, th, sc)
    return _traced("run_fusion", {"image_path": image_path, "csv_path": csv_path}, go)


# ------------------------------- CREW --------------------------------------
def get_llm():
    import crewai.llms.cache as _c   # CrewAI/Groq workaround (crewAI issue #5886)
    _c.mark_cache_breakpoint = lambda msg: msg
    return LLM(model=os.environ.get("AGENT_MODEL", "groq/openai/gpt-oss-20b"),
               api_key=os.environ["GROQ_API_KEY"])

def run_agentic(paths, location="Kilinochchi", llm=None):
    TRACE.clear(); llm = llm or get_llm()
    router = Agent(
        role="PV Data Triage Agent",
        goal="Look at what was uploaded, decide which diagnostic model fits it, and run it.",
        backstory="Solar reliability engineer who knows every model in the toolbox and never guesses "
                  "a data type without profiling the file first.",
        tools=[profile_file, run_rgb_model, run_thermal_model, run_rf_model, run_lstm_model, run_fusion],
        llm=llm, verbose=True)
    route_task = Task(
        description=f"Uploaded files: {json.dumps([str(p) for p in paths])}\n"
                    f"Array config: {json.dumps({k: v for k, v in CONFIG.items() if k != 'panel_specs'})}\n"
                    "1. Call profile_file on every file.\n"
                    "2. Choose the single most appropriate model tool from the profile (thermal vs RGB from "
                    "the colour stats; 1-row CSV vs time-series CSV; thermal image + 1-row CSV => run_fusion).\n"
                    "3. Run it, once. If a tool returns an error, fix the input choice and retry.\n"
                    "4. Reply with JSON: route (tool name), reason (one sentence), model_output.",
        expected_output="JSON with route, reason, model_output.", agent=router)

    expert = Agent(
        role="Senior PV Diagnostic Expert",
        goal="Turn a model result into a clear diagnostic ticket, using live weather only when it is relevant.",
        backstory="15+ years diagnosing tropical PV arrays; never attributes a multi-year trend to today's weather "
                  "and never forces a weather link onto a visible defect.",
        tools=[get_coordinates_for_location, fetch_weather_data], llm=llm, verbose=True)
    ticket_task = Task(
        description=f"Site: '{location}', Sri Lanka. Today: {datetime.now():%Y-%m-%d}.\n"
                    "Use the triage result; quote its numbers, never invent any.\n"
                    "WEATHER (tools give CURRENT conditions; assume data was captured now and say so):\n"
                    "- Sensor/fusion: fetch weather; cloud/low irradiance can mimic a fault.\n"
                    "- Thermal or RGB image: ALWAYS fetch weather. It cannot prove or rule out a defect, only affect "
                    "trust. Thermal needs irradiance of about 600 W/m2 or more, light wind, dry panel; otherwise contrast "
                    "drops and cloud shadows/rain cooling fake Shadowing or Diode patterns. RGB: overcast hides shadows, "
                    "rain adds glare and droplets that mimic Bird_drop/Dusty and washes off real dust. Rate capture "
                    "reliability HIGH/MEDIUM/LOW with cloud %, irradiance, rain mm, wind km/h; if LOW, lower confidence "
                    "and advise re-capture. If fallback_used is true, say live weather was unavailable and skip the rating.\n"
                    "- LSTM: skip weather; if failure_already_elapsed is true, say inspect now, never 'before' a past date.\n"
                    "PLAYBOOK (general guidance, use the matching entry):\n"
                    "Bird_drop: shaded cell -> hot spot, bypass diode conducts. Fix: soft-brush clean with low-mineral "
                    "water when cool, re-scan, bird spikes.\n"
                    "Dusty: soiling cuts current. Fix: cleaning schedule, compare performance ratio before/after.\n"
                    "Electrical_damage: burnt backsheet/junction box/connector -> arcing, shock/fire risk. Fix: isolate, "
                    "megger test, IV curve, inspect MC4, replace, warranty.\n"
                    "Physical_damage: broken glass/frame -> moisture, cracking, ground fault. Fix: isolate, insulation "
                    "test, EL imaging, replace, find cause (hail, impact, mounting, cleaning).\n"
                    "Cracking: cell cracks -> inactive area, local heating. Causes: load, handling, thermal cycling. Fix: "
                    "EL + IV curve, monitor, replace if hot-spotting.\n"
                    "Diode: failed bypass diode -> a third of module offline, IV step. Fix: test diode, replace junction "
                    "box or module.\n"
                    "Shadowing/Shading: bypass diodes activate, mismatch loss. Fix: shading survey, trim, re-string, "
                    "optimisers.\n"
                    "Short: low Voc (shorted cells/diode/ground fault). Fix: IR scan, insulation test.\n"
                    "Connector: high resistance -> Vmp/Pmax drop, Voc normal, joint heating. Fix: re-crimp MC4, contact "
                    "resistance, thermal scan under load.\n"
                    "OC: Isc near zero, Voc present. Fix: continuity test, isolator, fuses, cables, junction box.\n"
                    "LSTM trend: cell degradation (LID/PID, encapsulant, solder fatigue) or soiling. Fix: IV + EL on "
                    "weakest strings, compare string currents, inverter logs, plan replacement.\n"
                    "Clean/Normal: no visible defect; RGB cannot see microcracks, PID or diode faults.\n"
                    "TICKET (markdown, about 200-300 words): **PV Diagnostic Ticket** (site, date, model, detection, "
                    "confidence, key numbers), **Weather & Capture Conditions** (only if used, with reliability), "
                    "**Diagnosis** (status CONFIRMED_FAULT / POSSIBLE_WEATHER_INFLUENCE / UNCERTAIN / NOMINAL, confidence "
                    "0-1, 2-3 sentence justification), **Root Cause Analysis** (mechanism, 2-3 ranked causes, test that "
                    "separates them), **Technical Impact** (current/voltage/power, hot-spot or safety risk), **Corrective "
                    "Actions** (immediate; verification tests naming the instrument; repair/replace; prevention), "
                    "**Priority** (urgent/soon/routine + timeframe). If model confidence is under 60%, verify before "
                    "repair. Playbook content is general guidance, not a measured finding.",
        expected_output="A completed ticket in the structure above, no unresolved placeholders.",
        agent=expert, context=[route_task])

    result = Crew(agents=[router, expert], tasks=[route_task, ticket_task],
                  process=Process.sequential, verbose=True).kickoff()

    return {"ticket": str(result), "summary": summarize(), "trace": list(TRACE)}


MODEL_NAMES = {"run_rgb_model": "RGB image CNN", "run_thermal_model": "Thermal image CNN",
               "run_rf_model": "Random Forest (sensor snapshot)", "run_lstm_model": "LSTM (time series)",
               "run_fusion": "Sensor + thermal fusion"}

def summarize():
    """Key facts about the model the agent ended up running (for the UI)."""
    for t in reversed(TRACE):
        if t["tool"] in MODEL_NAMES:
            return {"tool": t["tool"], "model": MODEL_NAMES[t["tool"]], "result": json.loads(t["result"])}
    return None