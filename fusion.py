"""
fusion.py  -  Sensor (RF) + Thermal (MobileNetV2) fusion for partial-shading detection.

METHOD: learned fusion (logistic regression on the two models' probabilities)

    logit(p) = ln( p / (1 - p) )                    (p is clipped to [0.001, 0.999])
    z        = BIAS + W_RF * logit(p_rf) + W_THERMAL * logit(p_thermal)
    P        = 1 / (1 + e^(-z))                     P = fused probability of shading
    P >= 0.5  ->  shading CONFIRMED (cross-modal)

p_rf      = RF probability of class "Shading"
p_thermal = MobileNetV2 probability of class "Shadowing"

WHERE THE NUMBERS COME FROM (not hand-picked; see fit_fusion_weights.py):
For each final RF model used in the app (1-string: 20 trees / depth 15, 3-string: 15 trees /
depth 5), take its probability of Shading on its 30 % test file (method1_random split) (rows it has never seen), give every test row a thermal image (shaded row ->
a Shadowing image, other row -> a Cracking/Diode image) and read the thermal CNN's probability
for it, then fit a logistic regression on [logit(p_rf), logit(p_thermal)] -> shaded or not.
Repeated for 20 random pairings and averaged. The pairing is needed because no dataset has
sensor data and a thermal image of the same panel. Re-fit if you retrain a model.

NOTE: both RFs are almost perfect on their test files (1 and 0 mistakes), so the fit has
almost nothing to learn from; in particular the 3-string weights are weakly determined.

The function name, arguments and returned keys are the same as before, so agentic.py and
app.py do not need any change.
"""
import math

# (bias, weight on logit(p_rf), weight on logit(p_thermal))
WEIGHTS = {
    "1-string": (-0.631, 1.652, 0.505),
    "3-string": (-0.042, 1.904, 0.242),
}

FUSION_CONFIRM_THRESHOLD = 0.5
_EPS = 1e-3

RF_SHADING_LABEL = "Shading"
THERMAL_SHADOWING_LABEL = "Shadowing"


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def _extract_rf_shading_prob(rf_result: dict) -> float:
    """RF's probability specifically for the Shading class, even when it wasn't the top-1 pick.
    Falls back to 1.0/0.0 from the top label if class probabilities were not attached."""
    probs = rf_result.get("_debug_class_probabilities")
    if probs and RF_SHADING_LABEL in probs:
        return float(probs[RF_SHADING_LABEL])
    return 1.0 if rf_result.get("detection") == RF_SHADING_LABEL else 0.0


def _extract_thermal_shadowing_prob(thermal_result: dict) -> float:
    """Same idea for the thermal model's Shadowing probability."""
    probs = thermal_result.get("_debug_class_probabilities")
    if probs and THERMAL_SHADOWING_LABEL in probs:
        return float(probs[THERMAL_SHADOWING_LABEL])
    return 1.0 if thermal_result.get("detection") == THERMAL_SHADOWING_LABEL else 0.0


def fuse_predictions(rf_result: dict, thermal_result: dict, string_config: str = "1-string") -> dict:
    bias, w_rf, w_th = WEIGHTS.get(string_config, WEIGHTS["1-string"])

    p_rf = _extract_rf_shading_prob(rf_result)
    p_thermal = _extract_thermal_shadowing_prob(thermal_result)

    z = bias + w_rf * _logit(p_rf) + w_th * _logit(p_thermal)
    fused_p = 1.0 / (1.0 + math.exp(-z))

    result = {
        "rf_detection": rf_result.get("detection"),
        "rf_shading_probability": round(p_rf, 4),
        "thermal_detection": thermal_result.get("detection"),
        "thermal_shadowing_probability": round(p_thermal, 4),
        "combined_shading_score": round(fused_p, 4),
    }

    if fused_p >= FUSION_CONFIRM_THRESHOLD:
        result["fused_status"] = "CONFIRMED_CROSS_MODAL"
        result["fused_label"] = "Partial Shading"
        result["fused_confidence_pct"] = round(fused_p * 100, 2)
        result["explanation"] = (
            f"Sensor data ({p_rf*100:.1f}% Shading probability) and thermal "
            f"imagery ({p_thermal*100:.1f}% Shadowing probability) together point "
            f"to partial shading. Fused confidence: {fused_p*100:.1f}%."
        )
    else:
        result["fused_status"] = "NO_SHARED_EVIDENCE"
        result["fused_label"] = None
        result["fused_confidence_pct"] = None
        result["explanation"] = (
            f"Sensor top finding: {rf_result.get('detection')}. "
            f"Thermal top finding: {thermal_result.get('detection')}. "
            "These don't share enough combined evidence to confirm a single "
            "cross-modal diagnosis -- reporting both findings independently "
            "rather than forcing a match that isn't there."
        )

    return result


if __name__ == "__main__":
    print("Case 1: both models moderately confident, neither top-1 --")
    print(fuse_predictions(
        {"detection": "Normal", "_debug_class_probabilities": {"Normal": 0.55, "Shading": 0.71, "Short": 0.0, "Connector": 0.0, "OC": 0.0}},
        {"detection": "Diode", "_debug_class_probabilities": {"Cracking": 0.1, "Diode": 0.44, "Shadowing": 0.46}}))

    print("\nCase 2: no shared evidence --")
    print(fuse_predictions(
        {"detection": "Connector", "_debug_class_probabilities": {"Normal": 0.02, "Shading": 0.03, "Short": 0.05, "Connector": 0.88, "OC": 0.02}},
        {"detection": "Cracking", "_debug_class_probabilities": {"Cracking": 0.81, "Diode": 0.1, "Shadowing": 0.09}}))

    print("\nCase 3: strong agreement --")
    print(fuse_predictions(
        {"detection": "Shading", "_debug_class_probabilities": {"Normal": 0.02, "Shading": 0.9, "Short": 0.03, "Connector": 0.03, "OC": 0.02}},
        {"detection": "Shadowing", "_debug_class_probabilities": {"Cracking": 0.05, "Diode": 0.15, "Shadowing": 0.8}}))
