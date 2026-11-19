"""
SepsisSense feature engineering, version 3 (research-informed).

Adds to the 67 v2 features (src/features_v2.py) the feature families used by
top PhysioNet/CinC 2019 entries:

  * Measurement process (Du et al. 2019; Yang et al. 2019): for every clinical
    variable, the cumulative number of measurements so far and hours since the
    last measurement. Informative missingness: what clinicians chose to measure.
  * Vital-sign dynamics (Du et al.: deltas + variance; Morrill et al.: rolling
    min/max over 6 h): 6 h std, 12 h min/max/mean, 1 h and 6 h change, and the
    deviation from the patient's own running mean.
  * Lab dynamics: change since the previous different measurement.
  * Clinical scores (Morrill et al.: PartialSOFA, BUN/Cr): partial SOFA from the
    components available in the dataset (coagulation, liver, cardiovascular,
    renal), its rise over the previous 24 h, SIRS count, a NEWS-like score, and
    BUN/creatinine ratio.

New feature names use "<variable>__<stat>" so the UI can group explanations by
clinical concept. All computation is vectorised per patient; no future rows are
used (every feature at hour t depends only on hours <= t).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

import features_v2 as fv

CLINICAL_VARS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
    "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DYN_VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]
DYN_LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "BUN", "Bilirubin_total", "pH", "HCO3", "Glucose", "Hgb"]

EXTRA_FEATURES: List[str] = (
    [f"{v}__n_meas" for v in CLINICAL_VARS]
    + [f"{v}__hours_since" for v in CLINICAL_VARS if v not in fv.KEY_LABS]
    + [f"{v}__{s}" for v in DYN_VITALS for s in ("std6", "min12", "max12", "mean12", "diff1", "diff6", "dev_mean")]
    + [f"{v}__delta_prev" for v in DYN_LABS]
    + ["score__partial_sofa", "score__partial_sofa_rise24", "score__sirs", "score__news", "score__bun_cr"]
)
FEATURES: List[str] = fv.FEATURES + EXTRA_FEATURES


def _roll(series: pd.Series, pid: np.ndarray, window: int, how: str) -> np.ndarray:
    r = series.groupby(pid, sort=False).rolling(window, min_periods=1)
    r = getattr(r, how)()
    return r.reset_index(level=0, drop=True).sort_index().to_numpy(dtype=np.float32)


def engineer(df: pd.DataFrame, patient_col: str = "Patient") -> pd.DataFrame:
    """v2 features + v3 extras, NaN where missing (impute with training medians afterwards)."""
    df = df.reset_index(drop=True)
    base = fv.engineer(df, patient_col=patient_col)
    pid = df[patient_col].to_numpy()
    g = df.groupby(patient_col, sort=False)
    out: Dict[str, np.ndarray] = {}

    # ---- measurement process ----
    for v in CLINICAL_VARS:
        m = df[v].notna()
        n = m.astype(np.int32).groupby(pid, sort=False).cumsum().to_numpy()
        out[f"{v}__n_meas"] = n.astype(np.float32)
        if v not in fv.KEY_LABS:
            run = df.groupby([pid, n], sort=False).cumcount().to_numpy()
            out[f"{v}__hours_since"] = np.where(n > 0, run, 99 + run).astype(np.float32)

    # ---- vital dynamics (on forward-filled values) ----
    filled = g[list(dict.fromkeys(DYN_VITALS + DYN_LABS))].ffill()
    for v in DYN_VITALS:
        s = filled[v]
        out[f"{v}__std6"] = _roll(s, pid, 6, "std")
        out[f"{v}__min12"] = _roll(s, pid, 12, "min")
        out[f"{v}__max12"] = _roll(s, pid, 12, "max")
        out[f"{v}__mean12"] = _roll(s, pid, 12, "mean")
        out[f"{v}__diff1"] = s.groupby(pid, sort=False).diff(1).to_numpy(dtype=np.float32)
        out[f"{v}__diff6"] = s.groupby(pid, sort=False).diff(6).to_numpy(dtype=np.float32)
        expanding = s.groupby(pid, sort=False).expanding().mean().reset_index(level=0, drop=True).sort_index()
        out[f"{v}__dev_mean"] = (s - expanding).to_numpy(dtype=np.float32)

    # ---- lab dynamics: change since the previous different measurement ----
    for v in DYN_LABS:
        raw = df[v]
        measured = raw.notna().to_numpy()
        prev_val = raw.groupby(pid, sort=False).ffill().groupby(pid, sort=False).shift(1)
        delta = (raw - prev_val).where(measured)
        out[f"{v}__delta_prev"] = delta.groupby(pid, sort=False).ffill().to_numpy(dtype=np.float32)

    # ---- clinical scores ----
    plt = filled["Platelets"].to_numpy()
    bili = filled["Bilirubin_total"].to_numpy()
    mapv = filled["MAP"].to_numpy()
    cr = filled["Creatinine"].to_numpy()
    bun = filled["BUN"].to_numpy()

    def bands(x, cuts, reverse=False):
        x = np.nan_to_num(x, nan=np.nan)
        score = np.zeros(len(x), dtype=np.float32)
        for c in cuts:
            score += (x < c) if reverse else (x >= c)
        score[np.isnan(x)] = 0
        return score

    sofa = (
        bands(plt, [150, 100, 50, 20], reverse=True)
        + bands(bili, [1.2, 2.0, 6.0, 12.0])
        + np.where(np.isnan(mapv), 0, (mapv < 70)).astype(np.float32)
        + bands(cr, [1.2, 2.0, 3.5, 5.0])
    )
    out["score__partial_sofa"] = sofa
    sofa_min24 = pd.Series(sofa).groupby(pid, sort=False).rolling(24, min_periods=1).min() \
        .reset_index(level=0, drop=True).sort_index().to_numpy(dtype=np.float32)
    out["score__partial_sofa_rise24"] = sofa - sofa_min24

    hr = base["HR"].to_numpy(); rr = base["Resp"].to_numpy(); tmp = base["Temp"].to_numpy()
    wbc = base["WBC"].to_numpy(); paco2 = base["PaCO2"].to_numpy(); spo2 = base["O2Sat"].to_numpy()
    sbp = base["SBP"].to_numpy()
    nz = lambda a: np.nan_to_num(a, nan=np.nan)  # noqa: E731
    sirs = (
        (nz(hr) > 90).astype(np.float32)
        + ((nz(rr) > 20) | (nz(paco2) < 32)).astype(np.float32)
        + ((nz(tmp) > 38) | (nz(tmp) < 36)).astype(np.float32)
        + ((nz(wbc) > 12) | (nz(wbc) < 4)).astype(np.float32)
    )
    out["score__sirs"] = sirs
    news = (
        np.select([rr <= 8, rr <= 11, rr <= 20, rr <= 24, rr > 24], [3, 1, 0, 2, 3], 0)
        + np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], 0)
        + np.select([sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219, sbp > 219], [3, 2, 1, 0, 3], 0)
        + np.select([hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130, hr > 130], [3, 1, 0, 1, 2, 3], 0)
        + np.select([tmp <= 35, tmp <= 36, tmp <= 38, tmp <= 39, tmp > 39], [3, 1, 0, 1, 2], 0)
    ).astype(np.float32)
    out["score__news"] = news
    out["score__bun_cr"] = (bun / np.where(np.abs(cr) < 1e-6, np.nan, cr)).astype(np.float32)

    extra = pd.DataFrame({f: out[f] for f in EXTRA_FEATURES})
    return pd.concat([base, extra], axis=1)
