"""
SepsisSense feature engineering, version 2.

Single source of truth for model inputs, shared by training (src/train_full.py),
the ward replay engine (api/engine.py) and manual assessment (api/assess.py).

Fixes relative to train_precision_model.extract_features (v1):
  * Lab-availability features are computed from the raw chart, BEFORE forward
    fill. In v1 they were computed after forward fill, so "lab measured this
    hour" really meant "lab ever measured".
  * Missing values are imputed with medians from the training split
    (saved to models/feature_medians.json), not medians of whatever frame is
    being processed. A single patient therefore gets the same treatment at
    inference as in training.
  * Fully vectorised (groupby rolling / cumcount), so 1.5M rows run in minutes.

The 67 feature names are unchanged from v1 so explanations in the UI still map.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROLL_VITALS = ["HR", "MAP", "Temp", "Resp", "O2Sat"]
KEY_LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "BUN", "pH", "Glucose"]

RAW_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
    "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
    "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]
DERIVED_FEATURES = [
    "HR_roll6h", "MAP_roll6h", "Temp_roll6h", "Resp_roll6h", "O2Sat_roll6h",
    "shock_index", "modified_shock_index", "qsofa_score", "hr_trend_3h", "map_trend_3h",
    "lactate_creatinine_ratio", "wbc_temp_interaction",
]
INTENT_FEATURES = [f for lab in KEY_LABS for f in (f"{lab}_measured_mask", f"{lab}_hours_since_meas")] + [
    "diagnostic_intensity_6h"
]
FEATURES: List[str] = RAW_FEATURES + DERIVED_FEATURES + INTENT_FEATURES  # 67


def engineer(df: pd.DataFrame, patient_col: str = "Patient") -> pd.DataFrame:
    """
    Build the 67 model features from a harmonized hourly frame (rows ordered by
    patient, then time). Returns float32 features with NaN where a value is
    still missing; call `impute` before scoring.
    """
    df = df.reset_index(drop=True)
    pid = df[patient_col].to_numpy()
    out: Dict[str, np.ndarray] = {}

    # ---- lab availability, from the raw chart (before forward fill) ----
    g = df.groupby(patient_col, sort=False)
    measured_count = np.zeros(len(df), dtype=np.float32)
    for lab in KEY_LABS:
        m = df[lab].notna()
        measured_count += m.to_numpy(dtype=np.float32)
        out[f"{lab}_measured_mask"] = m.to_numpy(dtype=np.float32)
        seen = m.astype(np.int32).groupby(pid, sort=False).cumsum()
        run = df.groupby([pid, seen.to_numpy()], sort=False).cumcount().to_numpy()
        # 0 in the hour it was drawn, +1 each hour after; 99, 100, ... before the first draw (as in v1)
        out[f"{lab}_hours_since_meas"] = np.where(seen.to_numpy() > 0, run, 99 + run).astype(np.float32)
    out["diagnostic_intensity_6h"] = (
        pd.Series(measured_count).groupby(pid, sort=False).rolling(6, min_periods=1).sum()
        .reset_index(level=0, drop=True).sort_index().to_numpy(dtype=np.float32)
    )

    # ---- forward fill within patient (chart persistence) ----
    filled = g[RAW_FEATURES].ffill()
    for c in RAW_FEATURES:
        out[c] = filled[c].to_numpy(dtype=np.float32)

    # ---- rolling means and trends ----
    for v in ROLL_VITALS:
        out[f"{v}_roll6h"] = (
            filled[v].groupby(pid, sort=False).rolling(6, min_periods=1).mean()
            .reset_index(level=0, drop=True).sort_index().to_numpy(dtype=np.float32)
        )
    out["hr_trend_3h"] = filled["HR"].groupby(pid, sort=False).diff(3).fillna(0.0).to_numpy(dtype=np.float32)
    out["map_trend_3h"] = filled["MAP"].groupby(pid, sort=False).diff(3).fillna(0.0).to_numpy(dtype=np.float32)

    # ---- interactions (same defaults as v1 for missing values) ----
    hr = filled["HR"].fillna(80.0).to_numpy()
    sbp = filled["SBP"].fillna(120.0).to_numpy()
    mapv = filled["MAP"].fillna(85.0).to_numpy()
    resp = filled["Resp"].fillna(18.0).to_numpy()
    lac = filled["Lactate"].fillna(1.5).to_numpy()
    creat = filled["Creatinine"].fillna(1.0).to_numpy()
    wbc = filled["WBC"].fillna(8.0).to_numpy()
    temp = filled["Temp"].fillna(37.0).to_numpy()
    out["shock_index"] = (hr / (np.abs(sbp) + 1e-6)).astype(np.float32)
    out["modified_shock_index"] = (hr / (np.abs(mapv) + 1e-6)).astype(np.float32)
    out["qsofa_score"] = ((resp >= 22).astype(np.float32) + (sbp <= 100).astype(np.float32))
    out["lactate_creatinine_ratio"] = (lac / (np.abs(creat) + 1e-6)).astype(np.float32)
    out["wbc_temp_interaction"] = (wbc * temp).astype(np.float32)

    return pd.DataFrame({f: out[f] for f in FEATURES})


def compute_medians(features: pd.DataFrame) -> Dict[str, float]:
    med = {}
    for c in FEATURES:
        v = np.nanmedian(features[c].to_numpy(dtype=np.float64)) if features[c].notna().any() else 0.0
        med[c] = float(v) if np.isfinite(v) else 0.0
    return med


def impute(features: pd.DataFrame, medians: Dict[str, float]) -> pd.DataFrame:
    return features.fillna({c: medians.get(c, 0.0) for c in FEATURES})


def load_medians(path: Path) -> Dict[str, float]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------- PhysioNet utility

def utility_terms(labels: np.ndarray, patients: np.ndarray,
                  dt_early: int = -12, dt_optimal: int = -6, dt_late: int = 3,
                  max_u_tp: float = 1.0, min_u_fn: float = -2.0, u_fp: float = -0.05):
    """
    Per-row utility if the model predicts 1 (u_pos) or 0 (u_neg), following the
    official PhysioNet/CinC 2019 scorer (evaluate_sepsis_score.py). Onset is the
    first positive label + 6 h. Utility of any prediction vector is then
    sum(where(pred, u_pos, u_neg)), which makes threshold scans cheap.
    Also returns best/inaction totals for normalisation.
    """
    labels = np.asarray(labels).astype(int)
    n = len(labels)
    u_pos = np.zeros(n)
    u_neg = np.zeros(n)
    best_pred = np.zeros(n, dtype=bool)
    m1 = max_u_tp / (dt_optimal - dt_early); b1 = -m1 * dt_early
    m2 = -max_u_tp / (dt_late - dt_optimal); b2 = -m2 * dt_late
    m3 = min_u_fn / (dt_late - dt_optimal); b3 = -m3 * dt_optimal  # falls from 0 to min_u_fn after onset
    # patient boundaries (rows are contiguous per patient)
    change = np.r_[True, patients[1:] != patients[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        lab = labels[s:e]
        t = np.arange(e - s)
        if lab.any():
            ts = int(np.argmax(lab)) - dt_optimal
            d = t - ts
            in_scope = t <= ts + dt_late
            early = d <= dt_optimal
            up = np.where(early, np.maximum(m1 * d + b1, u_fp), m2 * d + b2)
            un = np.where(early, 0.0, m3 * d + b3)
            u_pos[s:e] = np.where(in_scope, up, 0.0)
            u_neg[s:e] = np.where(in_scope, un, 0.0)
            best_pred[s:e] = (t >= max(0, ts + dt_early)) & (t < min(ts + dt_late + 1, e - s))
        else:
            u_pos[s:e] = u_fp
    best = float(np.where(best_pred, u_pos, u_neg).sum())
    inaction = float(u_neg.sum())
    return u_pos, u_neg, best, inaction


def normalized_utility(pred: np.ndarray, u_pos: np.ndarray, u_neg: np.ndarray, best: float, inaction: float) -> float:
    obs = float(np.where(pred, u_pos, u_neg).sum())
    return (obs - inaction) / (best - inaction) if best != inaction else 0.0


def causal_alerts(score: np.ndarray, patients: np.ndarray, threshold: float) -> np.ndarray:
    """
    2-hour persistence filter, causal: alert if score >= t and the previous hour
    (same patient) was >= 0.85 t, or the score is extreme (>= min(1.3 t, 0.9)).
    """
    score = np.asarray(score, dtype=np.float64)
    prev = np.r_[-np.inf, score[:-1]]
    first = np.r_[True, patients[1:] != patients[:-1]]
    prev[first] = -np.inf
    return (score >= threshold) & ((prev >= 0.85 * threshold) | (score >= min(threshold * 1.3, 0.9)))
