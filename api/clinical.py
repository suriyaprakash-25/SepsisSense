"""
Clinical reference data used by the web UI: units, adult reference ranges,
clinician-facing names for model drivers, bedside scores and guideline prompts.

Reference ranges are typical adult values for display only. They are not used
by the model.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# name, unit, low, high, decimals
MEASUREMENTS: Dict[str, Dict[str, Any]] = {
    # Vitals
    "HR": dict(label="Heart rate", short="HR", unit="bpm", low=60, high=100, dp=0, kind="vital"),
    "MAP": dict(label="Mean arterial pressure", short="MAP", unit="mmHg", low=65, high=110, dp=0, kind="vital"),
    "SBP": dict(label="Systolic BP", short="SBP", unit="mmHg", low=90, high=140, dp=0, kind="vital"),
    "DBP": dict(label="Diastolic BP", short="DBP", unit="mmHg", low=60, high=90, dp=0, kind="vital"),
    "Resp": dict(label="Respiratory rate", short="RR", unit="/min", low=12, high=20, dp=0, kind="vital"),
    "Temp": dict(label="Temperature", short="Temp", unit="°C", low=36.0, high=38.0, dp=1, kind="vital"),
    "O2Sat": dict(label="SpO₂", short="SpO₂", unit="%", low=94, high=100, dp=0, kind="vital"),
    "EtCO2": dict(label="End-tidal CO₂", short="EtCO₂", unit="mmHg", low=35, high=45, dp=0, kind="vital"),
    # Sepsis-relevant labs
    "Lactate": dict(label="Lactate", unit="mmol/L", low=0.5, high=2.0, dp=1, kind="lab", group="Perfusion"),
    "WBC": dict(label="WBC", unit="×10³/µL", low=4.0, high=12.0, dp=1, kind="lab", group="Inflammation"),
    "Platelets": dict(label="Platelets", unit="×10³/µL", low=150, high=400, dp=0, kind="lab", group="Coagulation"),
    "Creatinine": dict(label="Creatinine", unit="mg/dL", low=0.6, high=1.2, dp=2, kind="lab", group="Renal"),
    "BUN": dict(label="BUN", unit="mg/dL", low=7, high=20, dp=0, kind="lab", group="Renal"),
    "Bilirubin_total": dict(label="Bilirubin (total)", unit="mg/dL", low=0.3, high=1.2, dp=1, kind="lab", group="Hepatic"),
    "Bilirubin_direct": dict(label="Bilirubin (direct)", unit="mg/dL", low=0.0, high=0.3, dp=1, kind="lab", group="Hepatic"),
    "AST": dict(label="AST", unit="U/L", low=10, high=40, dp=0, kind="lab", group="Hepatic"),
    "Alkalinephos": dict(label="Alk phos", unit="U/L", low=40, high=130, dp=0, kind="lab", group="Hepatic"),
    "pH": dict(label="pH", unit="", low=7.35, high=7.45, dp=2, kind="lab", group="Blood gas"),
    "HCO3": dict(label="HCO₃⁻", unit="mmol/L", low=22, high=28, dp=0, kind="lab", group="Blood gas"),
    "BaseExcess": dict(label="Base excess", unit="mmol/L", low=-2, high=2, dp=0, kind="lab", group="Blood gas"),
    "PaCO2": dict(label="PaCO₂", unit="mmHg", low=35, high=45, dp=0, kind="lab", group="Blood gas"),
    "SaO2": dict(label="SaO₂", unit="%", low=94, high=100, dp=0, kind="lab", group="Blood gas"),
    "FiO2": dict(label="FiO₂", unit="", low=0.21, high=0.40, dp=2, kind="lab", group="Blood gas"),
    "Glucose": dict(label="Glucose", unit="mg/dL", low=70, high=180, dp=0, kind="lab", group="Chemistry"),
    "Potassium": dict(label="Potassium", unit="mmol/L", low=3.5, high=5.0, dp=1, kind="lab", group="Chemistry"),
    "Chloride": dict(label="Chloride", unit="mmol/L", low=98, high=107, dp=0, kind="lab", group="Chemistry"),
    "Calcium": dict(label="Calcium", unit="mg/dL", low=8.5, high=10.5, dp=1, kind="lab", group="Chemistry"),
    "Magnesium": dict(label="Magnesium", unit="mg/dL", low=1.7, high=2.2, dp=1, kind="lab", group="Chemistry"),
    "Phosphate": dict(label="Phosphate", unit="mg/dL", low=2.5, high=4.5, dp=1, kind="lab", group="Chemistry"),
    "Hgb": dict(label="Hemoglobin", unit="g/dL", low=12, high=17, dp=1, kind="lab", group="Hematology"),
    "Hct": dict(label="Hematocrit", unit="%", low=36, high=50, dp=0, kind="lab", group="Hematology"),
    "PTT": dict(label="PTT", unit="s", low=25, high=35, dp=0, kind="lab", group="Coagulation"),
    "Fibrinogen": dict(label="Fibrinogen", unit="mg/dL", low=200, high=400, dp=0, kind="lab", group="Coagulation"),
    "TroponinI": dict(label="Troponin I", unit="ng/mL", low=0.0, high=0.04, dp=2, kind="lab", group="Cardiac"),
}

# Clinical concepts used to group the 67 model features into explanations a
# clinician can act on. "primary" is the measurement shown next to the driver.
FEATURE_CONCEPT: Dict[str, Dict[str, Any]] = {
    "heart_rate": dict(label="Heart rate", primary="HR"),
    "blood_pressure": dict(label="Blood pressure", primary="MAP"),
    "shock_index": dict(label="Shock index (HR / BP)", primary=None),
    "respiratory": dict(label="Respiratory rate", primary="Resp"),
    "qsofa": dict(label="qSOFA criteria (RR, SBP)", primary=None),
    "oxygenation": dict(label="Oxygenation / ventilation", primary="O2Sat"),
    "temperature": dict(label="Temperature", primary="Temp"),
    "lactate": dict(label="Lactate", primary="Lactate"),
    "inflammation": dict(label="WBC / inflammatory response", primary="WBC"),
    "renal": dict(label="Renal function", primary="Creatinine"),
    "hepatic": dict(label="Liver function", primary="Bilirubin_total"),
    "coagulation": dict(label="Platelets / coagulation", primary="Platelets"),
    "acid_base": dict(label="Acid–base status", primary="pH"),
    "chemistry": dict(label="Electrolytes / glucose", primary=None),
    "hematology": dict(label="Hemoglobin / hematocrit", primary="Hgb"),
    "cardiac": dict(label="Troponin", primary="TroponinI"),
    "lab_pattern": dict(label="Recent lab testing", primary=None),
    "time": dict(label="Time in ICU / hospital", primary=None),
    "demographics": dict(label="Age / sex / unit", primary=None),
}

_EXPLICIT = {
    "HR": "heart_rate", "HR_roll6h": "heart_rate", "hr_trend_3h": "heart_rate",
    "SBP": "blood_pressure", "MAP": "blood_pressure", "DBP": "blood_pressure",
    "MAP_roll6h": "blood_pressure", "map_trend_3h": "blood_pressure",
    "shock_index": "shock_index", "modified_shock_index": "shock_index",
    "Resp": "respiratory", "Resp_roll6h": "respiratory",
    "qsofa_score": "qsofa",
    "O2Sat": "oxygenation", "O2Sat_roll6h": "oxygenation", "SaO2": "oxygenation",
    "FiO2": "oxygenation", "PaCO2": "oxygenation", "EtCO2": "oxygenation",
    "Temp": "temperature", "Temp_roll6h": "temperature",
    "Lactate": "lactate", "lactate_creatinine_ratio": "lactate",
    "WBC": "inflammation", "wbc_temp_interaction": "inflammation",
    "Creatinine": "renal", "BUN": "renal",
    "AST": "hepatic", "Alkalinephos": "hepatic", "Bilirubin_direct": "hepatic", "Bilirubin_total": "hepatic",
    "Platelets": "coagulation", "PTT": "coagulation", "Fibrinogen": "coagulation",
    "pH": "acid_base", "HCO3": "acid_base", "BaseExcess": "acid_base",
    "Potassium": "chemistry", "Chloride": "chemistry", "Calcium": "chemistry",
    "Magnesium": "chemistry", "Phosphate": "chemistry", "Glucose": "chemistry",
    "Hct": "hematology", "Hgb": "hematology",
    "TroponinI": "cardiac",
    "ICULOS": "time", "HospAdmTime": "time",
    "Age": "demographics", "Gender": "demographics", "Unit1": "demographics", "Unit2": "demographics",
}

FEATURE_LABELS = {
    "HR_roll6h": "HR, 6 h mean", "hr_trend_3h": "HR change over 3 h",
    "MAP_roll6h": "MAP, 6 h mean", "map_trend_3h": "MAP change over 3 h",
    "Temp_roll6h": "Temp, 6 h mean", "Resp_roll6h": "RR, 6 h mean", "O2Sat_roll6h": "SpO₂, 6 h mean",
    "shock_index": "Shock index (HR/SBP)", "modified_shock_index": "Modified shock index (HR/MAP)",
    "qsofa_score": "qSOFA (RR ≥ 22, SBP ≤ 100)", "lactate_creatinine_ratio": "Lactate / creatinine ratio",
    "wbc_temp_interaction": "WBC × temperature", "diagnostic_intensity_6h": "Key labs drawn in the last 6 h",
    "ICULOS": "Hours in ICU", "HospAdmTime": "Hours from hospital to ICU admission",
    "Unit1": "Medical ICU", "Unit2": "Surgical ICU", "Gender": "Sex",
}


def concept_for(feature: str) -> str:
    if feature in _EXPLICIT:
        return _EXPLICIT[feature]
    if feature.endswith("_measured_mask") or feature.endswith("_hours_since_meas") or feature == "diagnostic_intensity_6h":
        return "lab_pattern"
    return "chemistry"


def feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.endswith("_measured_mask"):
        base = feature.replace("_measured_mask", "")
        return f"{MEASUREMENTS.get(base, {}).get('label', base)} drawn this hour (1 = yes)"
    if feature.endswith("_hours_since_meas"):
        base = feature.replace("_hours_since_meas", "")
        return f"Hours since last {MEASUREMENTS.get(base, {}).get('label', base)} (99+ = not yet drawn)"
    return MEASUREMENTS.get(feature, {}).get("label", feature)


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def flag(name: str, value: Optional[float]) -> Optional[str]:
    m = MEASUREMENTS.get(name)
    if m is None or value is None:
        return None
    if value < m["low"]:
        return "low"
    if value > m["high"]:
        return "high"
    return None


# --------------------------------------------------------------- bedside scores

def sirs_criteria(hr, resp, temp, wbc, paco2=None) -> List[str]:
    met = []
    if hr is not None and hr > 90:
        met.append("HR > 90")
    if (resp is not None and resp > 20) or (paco2 is not None and paco2 < 32):
        met.append("RR > 20")
    if temp is not None and (temp > 38.0 or temp < 36.0):
        met.append("Temp > 38 or < 36 °C")
    if wbc is not None and (wbc > 12 or wbc < 4):
        met.append("WBC > 12 or < 4")
    return met


def qsofa_criteria(resp, sbp) -> List[str]:
    met = []
    if resp is not None and resp >= 22:
        met.append("RR ≥ 22")
    if sbp is not None and sbp <= 100:
        met.append("SBP ≤ 100")
    return met


# ---------------------------------------------------------- guideline prompts

def bundle_prompts(latest: Dict[str, Dict[str, Any]], risk: float, threshold: float, alert: bool) -> List[Dict[str, Any]]:
    """
    Surviving Sepsis Campaign 2021 Hour-1 bundle items, annotated with what the
    chart currently shows. These are rule-based reminders, not model output.
    """
    lac = latest.get("Lactate", {})
    lac_v, lac_age = lac.get("value"), lac.get("hours_ago")
    mapv = latest.get("MAP", {}).get("value")
    sbp = latest.get("SBP", {}).get("value")
    hypotension = (mapv is not None and mapv < 65) or (sbp is not None and sbp < 90)

    if lac_v is None:
        lac_ctx, lac_pri = "No lactate charted this admission.", "high"
    elif lac_age is not None and lac_age > 6:
        lac_ctx, lac_pri = f"Last lactate {lac_v:.1f} mmol/L, {int(lac_age)} h ago.", "high" if lac_v >= 2 else "medium"
    else:
        when = "this hour" if not lac_age else f"{int(lac_age)} h ago"
        lac_ctx = f"Lactate {lac_v:.1f} mmol/L, {when}."
        lac_pri = "high" if lac_v >= 2 else "low"
    if lac_v is not None and lac_v >= 2:
        lac_ctx += " Elevated: remeasure within 2–4 h."

    fluids_ctx = []
    if mapv is not None and mapv < 65:
        fluids_ctx.append(f"MAP {mapv:.0f} mmHg")
    if sbp is not None and sbp < 90:
        fluids_ctx.append(f"SBP {sbp:.0f} mmHg")
    if lac_v is not None and lac_v >= 4:
        fluids_ctx.append(f"lactate {lac_v:.1f} mmol/L")

    return [
        dict(id="lactate", title="Measure lactate",
             detail="Remeasure if initial lactate is > 2 mmol/L.",
             context=lac_ctx, priority=lac_pri),
        dict(id="cultures", title="Obtain blood cultures before antibiotics",
             detail="At least two sets; do not delay antibiotics to obtain them.",
             context="Culture orders are not captured in this dataset.", priority="medium" if alert else "low"),
        dict(id="antibiotics", title="Administer broad-spectrum antibiotics",
             detail="Within 1 h if sepsis is probable or shock is present; within 3 h if sepsis is possible without shock.",
             context="Medication administration is not captured in this dataset.", priority="medium" if alert else "low"),
        dict(id="fluids", title="30 mL/kg crystalloid for hypotension or lactate ≥ 4",
             detail="Reassess fluid responsiveness frequently.",
             context=("Criteria present: " + ", ".join(fluids_ctx) + ".") if fluids_ctx else "No hypotension or lactate ≥ 4 currently charted.",
             priority="high" if fluids_ctx else "low"),
        dict(id="vasopressors", title="Vasopressors to maintain MAP ≥ 65 mmHg",
             detail="If hypotensive during or after fluid resuscitation. Norepinephrine first line.",
             context=f"Current MAP {mapv:.0f} mmHg." if mapv is not None else "MAP not charted.",
             priority="high" if hypotension else "low"),
    ]
