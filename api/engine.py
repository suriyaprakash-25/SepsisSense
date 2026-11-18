"""
SepsisSense inference engine for the clinician web UI.

Replays real, de-identified ICU stays from the public PhysioNet/CinC 2019
Challenge (training set A) through the trained precision ensemble.

Design notes
------------
* Only stays p010000+ are used. Training read the first 1,500 files of each
  set, so these stays were never seen by the model.
* Feature engineering reuses ``train_precision_model.extract_features``. Missing
  values are median-imputed in training, so a reference sample of training-era
  stays is processed alongside the pool to keep imputation medians close to
  the training distribution.
* The saved alert threshold (0.75) was tuned on the *uncalibrated* ensemble
  score, while the isotonic calibrator tops out at ~0.39. Alerts therefore use
  the raw score (the scale the threshold was tuned on) and the UI displays the
  calibrated probability as the patient's risk.
* Alerts use a *causal* version of the repository's 2-hour persistence filter
  (the original peeks one hour into the future, which is impossible at the bedside).
* Ward composition: the first 7 septic and first 9 non-septic eligible stays
  from the unseen pool (file order), plus up to 2 stays chosen *because the model
  alerted* (one septic, one not) so the alert workflow can be demonstrated.
  Every patient carries its selection reason.
* Risk drivers are TreeSHAP contributions from the XGBoost and LightGBM members
  (log-odds, averaged). The final score also passes through the random forest
  and isotonic calibration, so drivers approximate *why*; they are not an exact
  decomposition of the displayed probability.
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore")

from data_harmonization import UniversalDataHarmonizer  # noqa: E402
from train_precision_model import extract_features  # noqa: E402

from clinical import FEATURE_CONCEPT, concept_for, feature_label  # noqa: E402

log = logging.getLogger("sepsissense.engine")

SET_A = ROOT / "physionet.org" / "files" / "challenge-2019" / "1.0.0" / "training" / "training_setA"
MODEL_PATH = ROOT / "models" / "sepsissense_precision_ensemble.pkl"
METRICS_PATH = ROOT / "models" / "precision_metrics.json"
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "cohort_v4.pkl"

VITALS = ["HR", "MAP", "SBP", "DBP", "Resp", "Temp", "O2Sat", "EtCO2"]
LABS = [
    "Lactate", "WBC", "Platelets", "Creatinine", "BUN", "Bilirubin_total", "pH",
    "HCO3", "BaseExcess", "PaCO2", "FiO2", "SaO2", "Glucose", "Potassium",
    "Chloride", "Calcium", "Magnesium", "Phosphate", "Hgb", "Hct", "PTT",
    "Fibrinogen", "AST", "Alkalinephos", "Bilirubin_direct", "TroponinI",
]

N_SEPTIC = 7
N_NON_SEPTIC = 9
N_REFERENCE = 400
POOL_SIZE = 1500
SCAN_START = 10_000  # first unseen file index (p010000)


@dataclass
class PatientRecord:
    pid: str
    bed: str
    unit: str
    age: float
    sex: str
    hosp_adm_hours: float
    n_hours: int
    base_hour: int
    sepsis_label_hour: Optional[int]  # first hour with SepsisLabel = 1 (onset - 6 h); research view only
    selection_reason: str
    raw: pd.DataFrame  # harmonized measurements; NaN = not charted that hour
    risk: np.ndarray  # calibrated probability
    raw_score: np.ndarray  # uncalibrated ensemble score (alert scale)
    member_risk: Dict[str, np.ndarray]
    alert: np.ndarray
    contrib: np.ndarray  # (hours, n_features) log-odds contributions
    feature_values: pd.DataFrame  # model inputs per hour (after ffill / imputation)


@dataclass
class Cohort:
    patients: Dict[str, PatientRecord] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    threshold: float = 0.75
    base_rate: float = 0.02
    pool_stats: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- io

def _read_psv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df["Patient"] = os.path.splitext(os.path.basename(path))[0]
    df["DatasetSource"] = "training_setA"
    return df


# ------------------------------------------------------------------- scoring

def causal_persistence_alert(score: np.ndarray, t: float) -> np.ndarray:
    """Alert if score >= t and the previous hour was >= 0.85 t, or the score is extreme."""
    out = np.zeros(len(score), dtype=int)
    for i, p in enumerate(score):
        if p < t:
            continue
        if p >= min(t * 1.30, 0.90) or (i > 0 and score[i - 1] >= 0.85 * t):
            out[i] = 1
    return out


def _tree_contributions(ens: Any, X: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    parts = []
    if "xgboost" in ens.models:
        booster = ens.models["xgboost"].get_booster()
        c = booster.predict(xgb.DMatrix(X, feature_names=booster.feature_names), pred_contribs=True)
        parts.append(c[:, :-1])
    if "lightgbm" in ens.models:
        c = ens.models["lightgbm"].predict(X, pred_contrib=True)
        parts.append(np.asarray(c)[:, :-1])
    if not parts:
        return np.zeros_like(X)
    return sum(parts) / len(parts)


def _pool_stats(feats_pool: pd.DataFrame, risk: np.ndarray, alert: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = feats_pool["SepsisLabel"].to_numpy()
    df = pd.DataFrame({"p": feats_pool["Patient"].to_numpy(), "y": y, "a": alert})
    per_pt = df.groupby("p").agg(septic=("y", "max"), alerted=("a", "max"))
    tp = int(((alert == 1) & (y == 1)).sum())
    fp = int(((alert == 1) & (y == 0)).sum())
    return {
        "patients": int(len(per_pt)),
        "hours": int(len(y)),
        "septic_patients": int(per_pt["septic"].sum()),
        "hour_roc_auc": float(roc_auc_score(y, risk)),
        "hour_pr_auc": float(average_precision_score(y, risk)),
        "positive_hour_rate": float(y.mean()),
        "alert_hours": int(alert.sum()),
        "alert_precision": float(tp / (tp + fp)) if tp + fp else None,
        "alert_recall": float(tp / max(1, int(y.sum()))),
        "septic_patients_alerted": int(per_pt.loc[per_pt["septic"] == 1, "alerted"].sum()),
        "non_septic_patients_alerted": int(per_pt.loc[per_pt["septic"] == 0, "alerted"].sum()),
    }


def build_cohort() -> Cohort:
    files = sorted(glob.glob(str(SET_A / "*.psv")))
    if len(files) < SCAN_START + POOL_SIZE:
        raise FileNotFoundError(
            f"PhysioNet 2019 training set A not found under {SET_A}. See DATA_DOWNLOAD.md."
        )

    log.info("Reading %d unseen stays and %d reference stays...", POOL_SIZE, N_REFERENCE)
    pool = [_read_psv(fp) for fp in files[SCAN_START:SCAN_START + POOL_SIZE]]
    reference = [_read_psv(fp) for fp in files[:N_REFERENCE]]
    pool_ids = [d["Patient"].iloc[0] for d in pool]

    harmonizer = UniversalDataHarmonizer(cache_dir=str(CACHE_PATH.parent))
    combined = harmonizer.harmonize_dataframe(
        pd.concat(reference + pool, ignore_index=True), source_name="PhysioNet_SetA"
    )
    log.info("Engineering features...")
    feats = extract_features(combined)

    ens = joblib.load(MODEL_PATH)
    for m in ens.models.values():
        try:
            m.set_params(device="cpu")
        except Exception:
            pass
    fnames = list(ens.feature_names)
    t = float(ens.optimizer.optimal_threshold)

    in_pool = feats["Patient"].isin(pool_ids).to_numpy()
    feats_p = feats.loc[in_pool].reset_index(drop=True)
    raw_all = combined.loc[in_pool].reset_index(drop=True)
    X = feats_p.reindex(columns=fnames, fill_value=0.0).to_numpy(dtype=np.float32)

    log.info("Scoring %d hours...", len(X))
    raw_score = ens._raw_predict_proba(X)[:, 1]
    risk = np.clip(ens.optimizer.calibrate(raw_score), 0.0, 1.0)
    labels_all = feats_p["SepsisLabel"].to_numpy()
    pid_arr = feats_p["Patient"].to_numpy()
    groups = {pid: np.flatnonzero(pid_arr == pid) for pid in pool_ids}
    alert = np.zeros(len(X), dtype=int)
    for idx in groups.values():
        alert[idx] = causal_persistence_alert(raw_score[idx], t)

    # ---- ward selection (deterministic, file order) ----
    chosen: Dict[str, str] = {}
    n_s = n_n = 0
    for pid in pool_ids:
        labels = labels_all[groups[pid]]
        n = len(labels)
        if labels.max() == 1:
            first = int(labels.argmax())
            if n_s < N_SEPTIC and 12 <= first <= 70 and n >= first + 8:
                chosen[pid] = "consecutive_septic"
                n_s += 1
        elif n_n < N_NON_SEPTIC and 36 <= n <= 120:
            chosen[pid] = "consecutive_non_septic"
            n_n += 1
    want = {"septic": True, "non_septic": True}
    for pid in pool_ids:
        if pid in chosen or not any(want.values()):
            continue
        idx = groups[pid]
        if alert[idx].max() == 0 or len(idx) < 24 or int(np.flatnonzero(alert[idx])[0]) < 8:
            continue
        key = "septic" if labels_all[idx].max() == 1 else "non_septic"
        if want[key]:
            chosen[pid] = "alert_example"
            want[key] = False

    members_all = {k: m.predict_proba(X)[:, 1] for k, m in ens.models.items()}
    sel_rows = np.concatenate([groups[p] for p in chosen])
    contrib_sel = _tree_contributions(ens, X[sel_rows])
    contrib_lookup, off = {}, 0
    for p in chosen:
        k = len(groups[p])
        contrib_lookup[p] = contrib_sel[off:off + k]
        off += k

    rng = np.random.default_rng(7)
    septic_offsets = [-8, -5, -2, 0, 2, 4, 6]
    s_i = 0
    recs: List[dict] = []
    micu = sicu = 0
    for pid, reason in chosen.items():
        idx = groups[pid]
        raw_p = raw_all.iloc[idx].reset_index(drop=True)
        labels = labels_all[idx]
        n = len(idx)
        label_hour = int(labels.argmax()) if labels.max() == 1 else None
        a = alert[idx]
        if reason == "alert_example":
            base = int(np.flatnonzero(a)[0]) + 1
        elif label_hour is not None:
            base = label_hour + septic_offsets[s_i % len(septic_offsets)]
            s_i += 1
        else:
            base = int(rng.integers(12, max(13, n - 14)))
        unit = "SICU" if raw_p["Unit2"].fillna(0).max() == 1 else "MICU"
        if unit == "MICU":
            micu += 1
            bed = f"MICU-{micu:02d}"
        else:
            sicu += 1
            bed = f"SICU-{sicu:02d}"
        adm = raw_p["HospAdmTime"].iloc[0]
        recs.append(dict(
            pid=pid, bed=bed, unit=unit,
            age=float(raw_p["Age"].iloc[0]),
            sex="M" if raw_p["Gender"].iloc[0] == 1 else "F",
            hosp_adm_hours=float(-adm) if pd.notna(adm) else 0.0,
            n_hours=n,
            base_hour=int(np.clip(base, 6, n - 1)),
            sepsis_label_hour=label_hour,
            selection_reason=reason,
            raw=raw_p,
            risk=risk[idx],
            raw_score=raw_score[idx],
            member_risk={k: v[idx] for k, v in members_all.items()},
            alert=a,
            contrib=contrib_lookup[pid],
            feature_values=feats_p.iloc[idx][fnames].reset_index(drop=True),
        ))

    cohort = Cohort(feature_names=fnames, threshold=t)
    cohort.base_rate = float(labels_all.mean())
    cohort.pool_stats = _pool_stats(feats_p, risk, alert)
    for r in sorted(recs, key=lambda r: (r["unit"] != "MICU", r["bed"])):
        cohort.patients[r["pid"]] = PatientRecord(**r)
    return cohort


def load_cohort(force: bool = False) -> Cohort:
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    cohort = build_cohort()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cohort, f)
    return cohort


# ------------------------------------------------------------- driver grouping

def grouped_drivers(rec: PatientRecord, hour: int, feature_names: List[str], top: int = 6) -> List[Dict[str, Any]]:
    c = rec.contrib[hour]
    fv = rec.feature_values.iloc[hour]
    groups: Dict[str, float] = {}
    members: Dict[str, List[Dict[str, Any]]] = {}
    for j, f in enumerate(feature_names):
        concept = concept_for(f)
        groups[concept] = groups.get(concept, 0.0) + float(c[j])
        members.setdefault(concept, []).append({
            "feature": f,
            "label": feature_label(f),
            "value": float(fv[f]),
            "contribution": float(c[j]),
        })
    ranked = sorted(groups.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
    out = []
    for concept, val in ranked:
        feats = sorted(members[concept], key=lambda m: abs(m["contribution"]), reverse=True)[:4]
        out.append({
            "concept": concept,
            "label": FEATURE_CONCEPT[concept]["label"],
            "primary": FEATURE_CONCEPT[concept]["primary"],
            "contribution": val,
            "features": feats,
        })
    return out
