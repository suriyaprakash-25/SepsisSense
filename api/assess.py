"""
Manual assessment: score vitals and labs entered by a clinician.

The model was trained on hourly time series (6 h rolling means, 3 h trends,
lab-availability features), so a clinician can enter readings at several time
points. Entries are placed on an hourly grid ending "now", then run through the
same feature steps as training.

Feature engineering mirrors ``train_precision_model.extract_features`` step for
step, with one deliberate difference: the final median imputation uses medians
from training-era stays instead of medians of this single patient (a lone
patient's medians are either their own values or missing, which training never saw).
"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import engine
from data_harmonization import PHYSIOLOGICAL_BOUNDS, UniversalDataHarmonizer
from precision_engineering import extract_clinical_intent_features

log = logging.getLogger("sepsissense.assess")

MEDIANS_PATH = engine.CACHE_PATH.parent / "reference_medians_v1.json"
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


# --------------------------------------------------------------- features

def engineer_features(df: pd.DataFrame, medians: Optional[Dict[str, float]]) -> pd.DataFrame:
    """Same steps as train_precision_model.extract_features; imputation uses `medians`."""
    df_out = df.copy()
    pc = "Patient"
    numeric_cols = [c for c in df_out.columns if c not in [pc, "DatasetSource", "SepsisLabel"]]
    df_out[numeric_cols] = df_out.groupby(pc)[numeric_cols].ffill()

    for vital in ["HR", "MAP", "Temp", "Resp", "O2Sat"]:
        col = f"{vital}_roll6h"
        if vital in df_out.columns:
            df_out[col] = df_out.groupby(pc)[vital].transform(lambda s: s.rolling(6, min_periods=1).mean())
        else:
            df_out[col] = 0.0

    hr = df_out.get("HR", 0.0).fillna(80.0)
    sbp = df_out.get("SBP", 0.0).fillna(120.0)
    map_val = df_out.get("MAP", 0.0).fillna(85.0)
    resp = df_out.get("Resp", 0.0).fillna(18.0)
    lac = df_out.get("Lactate", 0.0).fillna(1.5)
    creat = df_out.get("Creatinine", 0.0).fillna(1.0)
    wbc = df_out.get("WBC", 0.0).fillna(8.0)
    temp = df_out.get("Temp", 0.0).fillna(37.0)

    df_out["shock_index"] = hr / (np.abs(sbp) + 1e-6)
    df_out["modified_shock_index"] = hr / (np.abs(map_val) + 1e-6)
    df_out["qsofa_score"] = (resp >= 22).astype(float) + (sbp <= 100).astype(float)
    df_out["hr_trend_3h"] = df_out.groupby(pc)["HR"].diff(3).fillna(0.0) if "HR" in df_out else 0.0
    df_out["map_trend_3h"] = df_out.groupby(pc)["MAP"].diff(3).fillna(0.0) if "MAP" in df_out else 0.0
    df_out["lactate_creatinine_ratio"] = lac / (np.abs(creat) + 1e-6)
    df_out["wbc_temp_interaction"] = wbc * temp
    if "EtCO2" not in df_out.columns:
        df_out["EtCO2"] = 0.0

    # pandas returns a wide frame from groupby().apply() when there is a single
    # group, which breaks the training helper; a one-row padding patient avoids it.
    single = df_out[pc].nunique() == 1
    if single:
        pad = pd.DataFrame([{pc: "__pad__"}])
        df_out = pd.concat([df_out, pad], ignore_index=True)
    df_out = extract_clinical_intent_features(df_out, patient_col=pc)
    if single:
        df_out = df_out[df_out[pc] != "__pad__"].reset_index(drop=True)

    if medians is not None:
        for col in df_out.columns:
            if col in (pc, "DatasetSource", "SepsisLabel"):
                continue
            df_out[col] = df_out[col].fillna(medians.get(col, 0.0))
    return df_out


def _load_reference_medians(feature_names: List[str]) -> Dict[str, float]:
    if MEDIANS_PATH.exists():
        return json.loads(MEDIANS_PATH.read_text(encoding="utf-8"))
    log.info("Computing reference medians from %d training-era stays...", engine.N_REFERENCE)
    files = sorted(glob.glob(str(engine.SET_A / "*.psv")))[: engine.N_REFERENCE]
    ref = pd.concat([engine._read_psv(f) for f in files], ignore_index=True)
    ref = UniversalDataHarmonizer(cache_dir=str(MEDIANS_PATH.parent)).harmonize_dataframe(ref, "PhysioNet_SetA")
    feats = engineer_features(ref, medians=None)
    med = {}
    for col in feats.columns:
        if col in ("Patient", "DatasetSource", "SepsisLabel"):
            continue
        v = pd.to_numeric(feats[col], errors="coerce").median()
        med[col] = float(v) if pd.notna(v) else 0.0
    MEDIANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEDIANS_PATH.write_text(json.dumps(med, indent=1), encoding="utf-8")
    return med


# ----------------------------------------------------------------- engine

@dataclass
class Assessor:
    ens: Any
    medians: Dict[str, float]
    feature_names: List[str]
    threshold: float

    @classmethod
    def load(cls) -> "Assessor":
        ens = joblib.load(engine.MODEL_PATH)
        for m in ens.models.values():
            try:
                m.set_params(device="cpu")
            except Exception:
                pass
        fnames = list(ens.feature_names)
        return cls(ens=ens, medians=_load_reference_medians(fnames), feature_names=fnames,
                   threshold=float(ens.optimizer.optimal_threshold))

    def score(self, raw: pd.DataFrame) -> Dict[str, Any]:
        feats = engineer_features(raw, self.medians)
        X = feats.reindex(columns=self.feature_names, fill_value=0.0).to_numpy(dtype=np.float32)
        raw_score = self.ens._raw_predict_proba(X)[:, 1]
        risk = np.clip(self.ens.optimizer.calibrate(raw_score), 0.0, 1.0)
        return {
            "raw_score": raw_score,
            "risk": risk,
            "members": {k: m.predict_proba(X)[:, 1] for k, m in self.ens.models.items()},
            "contrib": engine._tree_contributions(self.ens, X),
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
    harmonized = UniversalDataHarmonizer(cache_dir=str(MEDIANS_PATH.parent)).harmonize_dataframe(df, "manual")

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
