"""
Manual assessment: score vitals and labs entered by a clinician.

The model was trained on hourly time series (6 h rolling means, 3 h trends,
lab-availability features), so a clinician can enter readings at several time
points. Entries are placed on an hourly grid ending "now", then run through the
same feature steps as training.

Features come from src/features_v2.py (the code used in training) and missing
values are imputed with the exact training medians, so a manually entered
patient is treated exactly like a training patient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import engine
import features_v2 as fv
from data_harmonization import PHYSIOLOGICAL_BOUNDS, UniversalDataHarmonizer

log = logging.getLogger("sepsissense.assess")

MAX_HOURS_BACK = 24

# Fields a clinician can enter. (low, high) are hard input limits; values outside
# are rejected rather than silently dropped by the harmonizer.
INPUT_FIELDS: Dict[str, Tuple[float, float]] = {
    "HR": (20, 260), "MAP": (20, 200), "SBP": (30, 280), "DBP": (10, 180),
    "Resp": (4, 70), "Temp": (28, 44), "O2Sat": (40, 100), "EtCO2": (0, 100),
    "Lactate": (0.1, 30), "WBC": (0.1, 150), "Platelets": (1, 1500),
    "Creatinine": (0.1, 25), "BUN": (1, 300), "Bilirubin_total": (0.1, 60),
    "pH": (6.7, 7.8), "HCO3": (1, 60), "BaseExcess": (-40, 40), "PaCO2": (5, 150),
    "FiO2": (0.21, 1.0), "Glucose": (10, 1200), "Potassium": (1, 10),
    "Hgb": (2, 25), "PTT": (10, 200),
}
for _k, (_lo, _hi) in PHYSIOLOGICAL_BOUNDS.items():
    if _k in INPUT_FIELDS:
        INPUT_FIELDS[_k] = (max(INPUT_FIELDS[_k][0], _lo), min(INPUT_FIELDS[_k][1], _hi))


# ----------------------------------------------------------------- engine

@dataclass
class Assessor:
    ens: Any
    medians: Dict[str, float]
    feature_names: List[str]
    threshold: float

    @classmethod
    def load(cls) -> "Assessor":
        ens = engine.load_model()
        return cls(ens=ens, medians=fv.load_medians(engine.MEDIANS_PATH), feature_names=list(ens.feature_names),
                   threshold=float(ens.optimizer.optimal_threshold))

    def score(self, raw: pd.DataFrame) -> Dict[str, Any]:
        feats = fv.impute(fv.engineer(raw), self.medians)
        X = feats[self.feature_names].to_numpy(dtype=np.float32)
        raw_score, risk = engine.model_score(self.ens, X)
        return {
            "raw_score": raw_score,
            "risk": risk,
            "members": {k: m.predict_proba(X)[:, 1] for k, m in self.ens.models.items()},
            "contrib": engine.tree_contributions(self.ens, X),
            "features": feats[self.feature_names].reset_index(drop=True),
        }


def build_grid(
    readings: List[Dict[str, Any]],
    age: float,
    sex: str,
    unit: str,
    icu_hours: int,
    hosp_to_icu_hours: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Place readings on an hourly grid that ends now. Returns (harmonized frame, warnings)."""
    warnings: List[str] = []
    span = max([int(r["hours_ago"]) for r in readings] + [0])
    n = span + 1
    rows = []
    for k in range(n):
        ago = span - k
        rows.append({
            "Patient": "assessment",
            "DatasetSource": "manual",
            "SepsisLabel": 0,
            "Age": float(age),
            "Gender": 1 if sex == "M" else 0,
            "Unit1": 1.0 if unit == "MICU" else 0.0,
            "Unit2": 1.0 if unit == "SICU" else 0.0,
            "HospAdmTime": -float(hosp_to_icu_hours),
            "ICULOS": float(max(1, icu_hours - ago)),
        })
    df = pd.DataFrame(rows)
    for r in readings:
        idx = span - int(r["hours_ago"])
        for name, val in r["values"].items():
            if val is None:
                continue
            df.loc[idx, name] = float(val)
    harmonized = UniversalDataHarmonizer(cache_dir=str(engine.CACHE_PATH.parent)).harmonize_dataframe(df, "manual")

    entered = {k for r in readings for k, v in r["values"].items() if v is not None}
    if n == 1:
        warnings.append(
            "Only one time point entered. The model's 3 h trend features are zero and 6 h averages equal the "
            "current values, so the score is less reliable. Add earlier readings if you have them."
        )
    elif span < 3:
        warnings.append("Readings span less than 3 hours, so the 3 h trend features are partly missing.")
    missing_core = [m for m in ("HR", "MAP", "Resp", "Temp", "O2Sat") if m not in entered]
    if missing_core:
        warnings.append(f"Not entered: {', '.join(missing_core)}. The model fills these with typical ICU values.")
    if "Lactate" not in entered:
        warnings.append("No lactate entered. The model treats it as not yet drawn, which itself changes the score.")
    if icu_hours < span + 1:
        warnings.append("Earliest reading is before ICU admission; its ICU hour was set to 1.")
    return harmonized, warnings
